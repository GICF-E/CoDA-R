"""表格参考线：CFR+ 平均策略的 exploitability 随迭代的变化，只画参考，不参与同成本比较。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from open_spiel.python.algorithms import cfr

from experiment.common.io_util import append_jsonl, log, write_json
from experiment.neural.envs import load_game, score_span
from experiment.neural.evals import exploitability_of_table


def run_cfr_plus(env: str, iterations: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents = True, exist_ok = True)
    game = load_game(env)
    span = score_span(game)
    solver = cfr.CFRPlusSolver(game)
    log_path = out_dir / "rollouts.jsonl"
    log_path.unlink(missing_ok = True)
    checkpoints = set(np.unique(np.geomspace(1, iterations, 30).astype(int)).tolist())
    expl = float("nan")
    for it in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if it in checkpoints:
            expl = exploitability_of_table(game, solver.average_policy())
            append_jsonl(log_path, dict(iteration = it, exploitability = expl, norm_exploitability = expl / span))
            log(f"CFR+ {env} iter={it} expl={expl:.4g}")
    summary = dict(env = env, iterations = iterations, final_exploitability = expl)
    write_json(out_dir / "run_summary.json", summary)
    return summary
