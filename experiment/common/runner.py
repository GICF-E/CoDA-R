"""格子运行器：配置哈希定 ID、按格子续跑、可选多进程并行。

一个格子 = 一个可 JSON 序列化的 payload 加一个顶层函数 fn(payload, cell_dir) -> summary。
格子 ID 含 payload 的哈希，配置有任何改动时续跑都不会误用旧结果（代码改动不在哈希里，需要不带 --resume 重跑）。
cells.jsonl 每完成一个格子追加一行状态；同一 ID 只认最后一行。
并行时每个格子独占一个子进程：子进程异常退出只重跑这一个格子，不影响其它格子。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import _ConnectionBase, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, TypeAlias

import torch

from experiment.common.io_util import append_jsonl, load_jsonl, log, repair_jsonl_tail

CellFn = Callable[[dict[str, Any], Path], dict[str, Any]]
# Pipe() 在 Windows 上返回 PipeConnection、其它平台返回 Connection，两者的公共基类是 _ConnectionBase
ResultConn: TypeAlias = "_ConnectionBase[dict[str, Any], dict[str, Any]]"
# 一个格子的子进程异常退出（崩溃、被系统杀掉）时最多尝试的总次数
MAX_CELL_ATTEMPTS = 3


@dataclass(frozen = True)
class RunOptions:
    out: Path
    smoke: bool = False
    resume: bool = False
    max_cells: int | None = None
    workers: int = 1
    # 每个格子子进程的线程数（torch 与 OpenMP）；> 1 时即使 workers = 1 也在子进程里运行
    cell_threads: int = 1


@dataclass(frozen = True)
class CellSpec:
    # 可读前缀，例如 leduc_poker_coda_r_s100
    name: str
    payload: dict[str, Any]

    @property
    def cell_id(self) -> str:
        digest = hashlib.sha1(json.dumps(self.payload, sort_keys = True, default = str).encode("utf-8")).hexdigest()
        return f"{self.name}_{digest[:8]}"


def prepare_output(options: RunOptions) -> Path:
    """不续跑时清空格子记录（格子目录在重跑时各自覆盖）；续跑时只修复中断留下的半行。"""
    out = options.out
    out.mkdir(parents = True, exist_ok = True)
    if options.resume:
        repair_jsonl_tail(out / "cells.jsonl")
    else:
        (out / "cells.jsonl").unlink(missing_ok = True)
    return out


def latest_records(out: Path) -> dict[str, dict[str, Any]]:
    return {r["cell_id"]: r for r in load_jsonl(out / "cells.jsonl")}


def _execute(fn: CellFn, payload: dict[str, Any], cell_dir: Path, redirect: bool) -> dict[str, Any]:
    cell_dir.mkdir(parents = True, exist_ok = True)
    t0 = time.time()
    with contextlib.ExitStack() as stack:
        if redirect:
            handle = stack.enter_context((cell_dir / "log.txt").open("w", encoding = "utf-8"))
            stack.enter_context(contextlib.redirect_stdout(handle))
            stack.enter_context(contextlib.redirect_stderr(handle))
        try:
            summary = fn(payload, cell_dir)
            return dict(status = "done", summary = summary, elapsed_s = time.time() - t0)
        except Exception as exc:
            traceback.print_exc()
            return dict(status = "error", error = f"{type(exc).__name__}: {exc}", elapsed_s = time.time() - t0)


def _cell_process(fn: CellFn, payload: dict[str, Any], cell_dir: Path, conn: ResultConn, threads: int) -> None:
    torch.set_num_threads(threads)
    conn.send(_execute(fn, payload, cell_dir, redirect = True))
    conn.close()


@dataclass
class _Running:
    spec: CellSpec
    attempt: int
    process: BaseProcess
    conn: ResultConn
    started: float


def _run_parallel(todo: list[CellSpec], fn: CellFn, options: RunOptions,
                  finish: Callable[[CellSpec, dict[str, Any]], None]) -> None:
    # 子进程在导入 numpy / torch 之前就继承这些变量，避免多进程互相抢线程
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(options.cell_threads)
    context = get_context("spawn")
    pending: deque[tuple[CellSpec, int]] = deque((spec, 1) for spec in todo)
    running: list[_Running] = []
    try:
        while pending or running:
            while pending and len(running) < options.workers:
                spec, attempt = pending.popleft()
                receiver, sender = context.Pipe(duplex = False)
                process = context.Process(target = _cell_process,
                                          args = (fn, spec.payload, options.out / spec.cell_id, sender, options.cell_threads))
                process.start()
                # 父进程不再持有写端，子进程一旦退出，读端立即可读（得到结果或 EOF）
                sender.close()
                running.append(_Running(spec, attempt, process, receiver, time.time()))
            ready = set(wait([r.conn for r in running] + [r.process.sentinel for r in running]))
            for r in [r for r in running if r.conn in ready or r.process.sentinel in ready]:
                result: dict[str, Any] | None = None
                # 子进程没发结果就退出时，Windows 上 poll 本身会抛 BrokenPipeError
                with contextlib.suppress(EOFError, OSError):
                    if r.conn.poll():
                        result = r.conn.recv()
                r.process.join()
                r.conn.close()
                running.remove(r)
                if result is not None:
                    finish(r.spec, result)
                    continue
                code = r.process.exitcode
                if r.attempt < MAX_CELL_ATTEMPTS:
                    log(f"{r.spec.cell_id}: worker exited abnormally (exit code {code}); "
                        f"retry {r.attempt + 1}/{MAX_CELL_ATTEMPTS}")
                    pending.appendleft((r.spec, r.attempt + 1))
                else:
                    finish(r.spec, dict(status = "error", elapsed_s = time.time() - r.started,
                                        error = f"worker exited abnormally {MAX_CELL_ATTEMPTS} times (last exit code {code})"))
    finally:
        for r in running:
            r.process.terminate()
            r.process.join()


def run_cells(specs: list[CellSpec], fn: CellFn, options: RunOptions) -> dict[str, dict[str, Any]]:
    """运行尚未完成的格子，返回这批格子的最新记录（含以前已完成的）。fn 必须是模块顶层函数。"""
    out = options.out
    done = {cid for cid, r in latest_records(out).items() if r.get("status") == "done"}
    todo = [spec for spec in specs if spec.cell_id not in done]
    n_done = len(specs) - len(todo)
    if options.max_cells is not None:
        todo = todo[:options.max_cells]
    log(f"{out.name}: {len(specs)} cells, {n_done} already done, running {len(todo)} with {options.workers} worker(s)"
        + (f" of {options.cell_threads} threads" if options.cell_threads > 1 else ""))
    n_finished = 0

    def finish(spec: CellSpec, result: dict[str, Any]) -> None:
        nonlocal n_finished
        n_finished += 1
        append_jsonl(out / "cells.jsonl", dict(cell_id = spec.cell_id, name = spec.name, payload = spec.payload, **result))
        state = "done" if result["status"] == "done" else f"FAILED ({result['error']})"
        log(f"[{n_finished}/{len(todo)}] {spec.cell_id}: {state} in {result['elapsed_s'] / 60:.1f} min")

    if options.workers <= 1 and options.cell_threads <= 1:
        for i, spec in enumerate(todo, start = 1):
            log(f"[{i}/{len(todo)}] start {spec.cell_id}")
            finish(spec, _execute(fn, spec.payload, out / spec.cell_id, redirect = False))
    else:
        _run_parallel(todo, fn, options, finish)
    records = latest_records(out)
    return {spec.cell_id: records[spec.cell_id] for spec in specs if spec.cell_id in records}


def cell_status(out: Path) -> dict[str, Any]:
    """进度查询：目录是否存在，以及最新记录中完成与失败的格子数。"""
    records = latest_records(out) if out.is_dir() else {}
    statuses = [r.get("status") for r in records.values()]
    return dict(exists = out.is_dir(), done = statuses.count("done"), error = statuses.count("error"))


def done_records(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records.values() if r.get("status") == "done"]


def require_complete(specs: list[CellSpec], records: dict[str, dict[str, Any]]) -> None:
    """后续阶段依赖本阶段全部格子时调用；有缺失或失败就停下，提示用 --resume 重跑。"""
    missing = [s.cell_id for s in specs if records.get(s.cell_id, {}).get("status") != "done"]
    if missing:
        raise RuntimeError(f"{len(missing)} cells are missing or failed (first: {missing[0]}); rerun with --resume")
