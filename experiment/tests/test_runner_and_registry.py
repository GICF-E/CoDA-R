import os
from pathlib import Path

import pytest
import torch

from experiment.common.io_util import load_jsonl
from experiment.common.runner import MAX_CELL_ATTEMPTS, CellSpec, RunOptions, prepare_output, run_cells
from experiment.experiments import ablations, diagnostics, main, mechanisms
from experiment.experiments.methods import CODA_C, METHODS, grid
from experiment.experiments.protocol import MAIN_GAMES
from experiment.registry import BY_ID, EXPERIMENTS, GROUPS, expand


def _ok(payload: dict, cell_dir: Path) -> dict:
    if payload.get("fail"):
        raise ValueError("boom")
    return dict(value = payload["x"] * 2)


def test_cell_id_depends_on_payload():
    assert CellSpec("a", dict(x = 1)).cell_id != CellSpec("a", dict(x = 2)).cell_id
    assert CellSpec("a", dict(x = 1)).cell_id == CellSpec("a", dict(x = 1)).cell_id


def test_run_cells_records_failures_and_resumes(tmp_path: Path):
    options = RunOptions(out = tmp_path)
    prepare_output(options)
    specs = [CellSpec("ok", dict(x = 3)), CellSpec("bad", dict(x = 1, fail = True))]
    records = run_cells(specs, _ok, options)
    assert records[specs[0].cell_id]["summary"]["value"] == 6
    assert records[specs[1].cell_id]["status"] == "error"
    resumed = RunOptions(out = tmp_path, resume = True)
    prepare_output(resumed)
    run_cells(specs, _ok, resumed)
    lines = load_jsonl(tmp_path / "cells.jsonl")
    # 完成的格子不重跑，失败的格子重跑一次
    assert sum(r["cell_id"] == specs[0].cell_id for r in lines) == 1
    assert sum(r["cell_id"] == specs[1].cell_id for r in lines) == 2


def _crash_or_ok(payload: dict, cell_dir: Path) -> dict:
    if payload.get("crash"):
        # 模拟原生代码里的崩溃：进程直接退出，不经过 Python 异常处理
        os._exit(3)
    return dict(value = payload["x"] * 2)


def test_parallel_worker_crash_only_affects_its_cell(tmp_path: Path):
    options = RunOptions(out = tmp_path, workers = 2)
    prepare_output(options)
    specs = [CellSpec("ok", dict(x = 3)), CellSpec("crash", dict(x = 1, crash = True)), CellSpec("ok2", dict(x = 5))]
    records = run_cells(specs, _crash_or_ok, options)
    assert records[specs[0].cell_id]["summary"]["value"] == 6
    assert records[specs[2].cell_id]["summary"]["value"] == 10
    assert records[specs[1].cell_id]["status"] == "error"
    assert f"{MAX_CELL_ATTEMPTS} times" in records[specs[1].cell_id]["error"]


def _report_threads(payload: dict, cell_dir: Path) -> dict:
    return dict(pid = os.getpid(), omp = os.environ.get("OMP_NUM_THREADS"), torch = torch.get_num_threads())


def test_cell_threads_runs_in_a_child_with_that_many_threads(tmp_path: Path):
    options = RunOptions(out = tmp_path, workers = 1, cell_threads = 2)
    prepare_output(options)
    spec = CellSpec("threads", dict(x = 0))
    summary = run_cells([spec], _report_threads, options)[spec.cell_id]["summary"]
    assert summary["pid"] != os.getpid()
    assert summary["omp"] == "2" and summary["torch"] == 2


def test_every_tuned_method_has_nine_configs():
    for env in MAIN_GAMES:
        for method in METHODS:
            assert len(grid(method, env)) == 9, (env, method)
        assert {p["coda_c"] for p in grid("coda_r", env)} == set(CODA_C[env])


def test_registry_ids_and_groups():
    assert len(BY_ID) == len(EXPERIMENTS)
    assert expand("theory") == GROUPS["theory"] and expand("main") == ["main"]
    with pytest.raises(ValueError):
        expand("long-run")
    assert main.planned_cells() == 3 * 12 * 10
    assert ablations.planned_cells() == 2 * 18 * 6
    assert diagnostics.planned_cells() == 2 * 2 * 6 * 6
    assert mechanisms.planned_vuln_cells() == 2 * 6 * (1 + 4 * 7)
    assert mechanisms.planned_hole_cells() == 6 * ((1 + 4 * 3) + (1 + 2 * 3))
    assert mechanisms.planned_forgetting_cells() == 2 * 6 * (1 + 5 + 2)
    assert "forgetting" in GROUPS["mechanisms"] and "scale-history" in GROUPS["scale"]
