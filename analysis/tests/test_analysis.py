from pathlib import Path
from typing import Any

import numpy as np
import pytest

from analysis.data import Outputs
from analysis.figures import FIGURES
from analysis.figures.mechanisms import ABLATION_DISPLAY
from analysis.figures.tuning import edge_flags
from analysis.plots import held_matrix
from analysis.style import MIXTURE_METHODS, SINGLE_METHODS, UPPER_BOUND_METHODS, method_style
from analysis.tables import write_tables
from experiment.common.io_util import append_jsonl
from experiment.common.runner import CellSpec, RunOptions, prepare_output, run_cells
from experiment.experiments.ablations import VARIANTS
from experiment.neural.config import RunConfig

BUDGET = 1000


def _fake_training(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    """按 payload 里给定的评估序列写一份 rollouts.jsonl，格式与训练循环相同。"""
    for i, (games, expl) in enumerate(payload["evals"]):
        append_jsonl(cell_dir / "rollouts.jsonl", dict(roll = i - 1, games = games, evaluation = True, exploitability = expl,
                                                       entropy = 0.5, mix_average_expl = expl / 2))
    return dict(games = BUDGET)


def _write_main(root: Path) -> None:
    out = root / "main"
    prepare_output(RunOptions(out = out))
    specs = []
    for method, scale in (("coda_r", 0.5), ("pfsp", 1.0), ("psro", 0.8), ("nfsp", 0.9)):
        for seed in (100, 101, 102):
            cfg = RunConfig(env = "leduc_poker", label = method_style(method).label, seed = seed, budget = BUDGET)
            evals = [(0, 2.0), (100, scale), (800, scale * 1.5 + 0.01 * seed), (BUDGET, scale * 1.4)]
            specs.append(CellSpec(f"{method}_s{seed}", dict(config = cfg.as_dict(), method = method, evals = evals)))
    prep = RunConfig(env = "leduc_poker", label = "base", seed = 100, budget = BUDGET)
    specs.append(CellSpec("prep", dict(config = prep.as_dict(), role = "prep", method = "rnad", evals = [(0, 1.0)])))
    run_cells(specs, _fake_training, RunOptions(out = out))


def test_methods_drawn_together_have_distinct_styles():
    methods = (*SINGLE_METHODS, *MIXTURE_METHODS, *UPPER_BOUND_METHODS)
    looks = [(method_style(m).color, method_style(m).linestyle) for m in methods]
    assert len(set(looks)) == len(looks)
    assert len({method_style(m).label for m in methods}) == len(methods)


def test_ablation_display_names_cover_existing_variants_without_collisions():
    assert set(ABLATION_DISPLAY) <= set(VARIANTS)
    shown = [ABLATION_DISPLAY.get(label, label) for label in VARIANTS]
    assert len(set(shown)) == len(shown)


def test_outputs_skip_prep_cells_and_hold_values_forward(tmp_path: Path):
    _write_main(tmp_path)
    out = Outputs(tmp_path)
    groups = out.by_env_method("main")
    assert sorted(m for _env, m in groups) == ["coda_r", "nfsp", "pfsp", "psro"]
    cells = groups[("leduc_poker", "coda_r")]
    matrix = held_matrix(cells, "exploitability", np.array([50.0, 100.0, 900.0]))
    assert matrix[:, 0] == pytest.approx([2.0] * 3)
    assert matrix[:, 1] == pytest.approx([0.5] * 3)
    assert out.neural("missing") == []


def test_edge_flags_mark_grid_boundaries():
    flags = edge_flags("coda_r", "leduc_poker", dict(coda_c = 48.0, ent_final = 0.003, eta = 0.1))
    assert flags == dict(ent_final = "inner", coda_c = "max")
    assert edge_flags("nfsp", "goofspiel4", dict(nfsp_anticipatory = 0.05, nfsp_lr = 0.01)) == \
        dict(nfsp_lr = "max", nfsp_anticipatory = "min")


def test_main_figures_and_tables_build_from_outputs(tmp_path: Path):
    _write_main(tmp_path / "outputs")
    out = Outputs(tmp_path / "outputs")
    dest = tmp_path / "figures"
    for name in ("main_curves", "main_tail", "main_paired", "main_degradation"):
        assert FIGURES[name](out, dest) == [name]
        assert (dest / f"{name}.pdf").is_file()
    # 没有对应实验输出的图直接跳过
    assert FIGURES["scale_curves"](out, dest) == []
    text = write_tables(out, dest).read_text(encoding = "utf-8")
    assert "CoDA-R - PFSP" in text and r"CoDA-R time average $\bar{q}$ - NFSP" in text
