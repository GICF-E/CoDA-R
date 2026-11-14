"""两阶段调参。

阶段 1：每种方法 9 个配置 x 3 个调参 seed，预算 B / 4（熵按 B / 4 自洽退火），按 tail_mean 的 seed 均值排序。
阶段 2：前 2 名在同样的调参 seed 上用完整预算 B 复核，取更好的一个写入 selected.json。
tune-a 调不依赖其它方法的方法；tune-b 的带池方法沿用 tune-a 中 R-NaD-PPO 选出的 eta，所以必须在 tune-a 之后运行。
调参 seed 不进入主实验统计。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiment.common.io_util import load_json, write_json, write_summary
from experiment.common.runner import CellSpec, RunOptions, prepare_output, require_complete, run_cells
from experiment.experiments.cells import run_train_cell, train_spec
from experiment.experiments.methods import (
    INHERITS_ETA,
    TUNE_STAGE_A,
    TUNE_STAGE_B,
    build_config,
    deployment,
    grid,
    params_label,
    title,
)
from experiment.experiments.protocol import MAIN_BUDGET, MAIN_GAMES, TUNE_BUDGET, TUNE_SEEDS
from experiment.experiments.report import CellResult, group, load_results

TOP_K = 2
SELECTED = "selected.json"
STAGES = {"a": TUNE_STAGE_A, "b": TUNE_STAGE_B}


def selected_path(out_root: Path, stage: str) -> Path:
    return out_root / f"tune-{stage}" / SELECTED


def load_selected(out_root: Path, smoke: bool) -> dict[str, dict[str, dict[str, Any]]]:
    """合并 tune-a 与 tune-b 的选择：{env: {method: params}}。冒烟时缺失则退回各方法网格的第一个配置。"""
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for stage, methods in STAGES.items():
        path = selected_path(out_root, stage)
        if path.is_file():
            for env, by_method in load_json(path)["selected"].items():
                merged.setdefault(env, {}).update(by_method)
        elif smoke:
            for env in MAIN_GAMES:
                for method in methods:
                    merged.setdefault(env, {})[method] = grid(method, env)[0]
        else:
            raise FileNotFoundError(f"{path} is missing; run tune-{stage} first")
    return merged


def _inherited(stage: str, out_root: Path, smoke: bool) -> dict[str, float]:
    """tune-b 的带池方法沿用的 eta：{env: eta}。"""
    if stage != "b":
        return {}
    path = selected_path(out_root, "a")
    if not path.is_file():
        if smoke:
            return {env: grid("rnad", env)[0]["eta"] for env in MAIN_GAMES}
        raise FileNotFoundError(f"{path} is missing; run tune-a first")
    return {env: float(by_method["rnad"]["eta"]) for env, by_method in load_json(path)["selected"].items()}


def _spec(method: str, env: str, seed: int, budget: int, params: dict[str, Any], stage: int, smoke: bool) -> CellSpec:
    label = f"{title(method)} [{params_label(params)}]"
    cfg = build_config(method, env, seed, budget, params, label = label)
    return train_spec(cfg, smoke, method = method, tune_stage = stage, params = params, deployment = deployment(method))


def _seeds(smoke: bool) -> tuple[int, ...]:
    # 冒烟只验证两阶段流程能走通：一个 seed、每种方法两个配置
    return TUNE_SEEDS[:1] if smoke else TUNE_SEEDS


def _stage1_specs(stage: str, eta: dict[str, float], smoke: bool) -> list[CellSpec]:
    specs = []
    for env in MAIN_GAMES:
        for method in STAGES[stage]:
            for params in grid(method, env)[:2] if smoke else grid(method, env):
                full = {**params, "eta": eta[env]} if method in INHERITS_ETA else params
                specs += [_spec(method, env, seed, TUNE_BUDGET, full, 1, smoke) for seed in _seeds(smoke)]
    return specs


def _ranking(results: list[CellResult]) -> dict[tuple[str, str], list[tuple[float, dict[str, Any]]]]:
    """{(env, method): [(tail_mean 的 seed 均值, params), ...]}，从好到差；NaN 排在最后。"""
    by_config = group(results, lambda r: (r.cfg.env, r.meta["method"], params_label(r.meta["params"])))
    ranking: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = {}
    for (env, method, _label), cells in by_config.items():
        values = [c.metrics()["tail_mean"] for c in cells]
        score = float(np.mean(values)) if all(np.isfinite(values)) else float("inf")
        ranking.setdefault((env, method), []).append((score, cells[0].meta["params"]))
    for rows in ranking.values():
        rows.sort(key = lambda row: row[0])
    return ranking


def _write_report(out: Path, stage: str, first: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]],
                  second: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]], selected: dict[str, Any]) -> None:
    lines = [f"# tune-{stage}", "",
             f"阶段 1：预算 {TUNE_BUDGET}，seed {list(TUNE_SEEDS)}；阶段 2：前 {TOP_K} 名在预算 {MAIN_BUDGET} 上复核。",
             "分数是 tail_mean 的 seed 均值（越小越好）；这些表同时是各参数的敏感性分析。", ""]
    for (env, method), rows in sorted(first.items()):
        lines += [f"## {env} / {title(method)}", "", "| 阶段 1 排名 | 参数 | 阶段 1 分数 |", "|---:|---|---:|"]
        lines += [f"| {i + 1} | {params_label(p)} | {s:.4g} |" for i, (s, p) in enumerate(rows)]
        confirm = second.get((env, method), [])
        lines += ["", "复核（完整预算）：" + "；".join(f"{params_label(p)} -> {s:.4g}" for s, p in confirm),
                  f"选定：{params_label(selected[env][method])}", ""]
    write_summary(out, "\n".join(lines))


def run_tuning(stage: str, options: RunOptions) -> Path:
    out = prepare_output(options)
    eta = _inherited(stage, out.parent, options.smoke)
    specs1 = _stage1_specs(stage, eta, options.smoke)
    records1 = run_cells(specs1, run_train_cell, options)
    require_complete(specs1, records1)
    first = _ranking(load_results(out, records1))
    specs2 = [_spec(method, env, seed, MAIN_BUDGET, params, 2, options.smoke)
              for (env, method), rows in first.items() for _score, params in rows[:TOP_K] for seed in _seeds(options.smoke)]
    records2 = run_cells(specs2, run_train_cell, options)
    require_complete(specs2, records2)
    second = _ranking(load_results(out, records2))
    selected: dict[str, dict[str, Any]] = {}
    for (env, method), rows in second.items():
        selected.setdefault(env, {})[method] = rows[0][1]
    write_json(out / SELECTED, dict(stage = stage, selected = selected))
    _write_report(out, stage, first, second, selected)
    return out


def planned_cells(stage: str) -> int:
    n_configs = sum(len(grid(method, env)) for env in MAIN_GAMES for method in STAGES[stage])
    n_methods = len(MAIN_GAMES) * len(STAGES[stage])
    return (n_configs + TOP_K * n_methods) * len(TUNE_SEEDS)
