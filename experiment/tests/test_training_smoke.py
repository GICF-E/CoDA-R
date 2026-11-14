from pathlib import Path

import numpy as np

from experiment.common.io_util import load_jsonl
from experiment.experiments.methods import build_config
from experiment.neural.config import RunConfig
from experiment.neural.psro import run_psro
from experiment.neural.train import run_training


def _evals(cell: Path) -> list[dict]:
    return [h for h in load_jsonl(cell / "rollouts.jsonl") if h.get("evaluation")]


def test_coda_r_run_stops_exactly_at_budget_and_inserts_oracles(tmp_path: Path):
    cfg = build_config("coda_r", "kuhn_poker", 0, 4_000_000, dict(ent_final = 0.01, eta = 0.1)).shrunk()
    summary = run_training(cfg, tmp_path)
    assert summary["games"] == cfg.budget
    assert summary["n_oracles"] >= 1
    evals = _evals(tmp_path)
    assert evals[0]["games"] == 0 and evals[-1]["games"] == cfg.budget
    assert all(np.isfinite(h["exploitability"]) for h in evals)
    assert any("mix_average_expl" in h for h in load_jsonl(tmp_path / "rollouts.jsonl"))
    assert (tmp_path / "oracles.jsonl").is_file()


def test_exact_estimator_and_sweep_paths(tmp_path: Path):
    for i, extra in enumerate((dict(loss_estimator = "exact"), dict(detection = "sweep"), dict(rule = "pfsp"))):
        cfg = build_config("coda_r", "kuhn_poker", 0, 4_000_000, dict(ent_final = 0.01, eta = 0.1), **extra).shrunk()
        assert run_training(cfg, tmp_path / str(i))["games"] == cfg.budget


def test_psro_spends_the_whole_budget(tmp_path: Path):
    cfg = RunConfig(env = "kuhn_poker", label = "psro", seed = 0, budget = 4_000_000, algorithm = "psro", oracle = "ppo").shrunk()
    summary = run_psro(cfg, tmp_path)
    assert summary["games"] == cfg.budget and summary["n_members"] >= 2
