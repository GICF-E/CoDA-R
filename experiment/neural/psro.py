"""神经 PSRO：对元策略训练 BR，部署对象是成员的整局混合。

每代成本 = BR 训练 + 新成员与全部现有成员的元矩阵对局；下一代放不进预算就停止，
剩余预算用于加密元矩阵估计，使运行恰好停在预算上。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiment.common.io_util import append_jsonl, log, write_json
from experiment.neural.budget import current_ledger, game_kind, measured_run
from experiment.neural.config import RunConfig
from experiment.neural.envs import FULL_TREE_INTRACTABLE, load_game, make_scorer, obs_and_action_size, score_span
from experiment.neural.evals import TableCache, aggregate, exact_br_from_table, exploitability_of_table, meta_nash_weights
from experiment.neural.mlp import SmallMLP
from experiment.neural.oracle import train_ppo_br
from experiment.neural.play import play_match
from experiment.neural.policies import MixtureActor, TorchActor, frozen_actor
from experiment.neural.schedule import EvalSchedule, exploiter_rollouts


class _MetaMatrix:
    """成员对成员的得分估计；冻结成员之间的结果不变，只补新对局。"""

    def __init__(self) -> None:
        self.sums: dict[tuple[int, int], float] = {}
        self.counts: dict[tuple[int, int], int] = {}

    def play(self, game: Any, members: list[Any], i: int, j: int, n: int, rng: np.random.Generator, scorer: Any) -> None:
        score = play_match(game, members[i], members[j], rng, n, scorer)
        self.sums[(i, j)] = self.sums.get((i, j), 0.0) + score * n
        self.counts[(i, j)] = self.counts.get((i, j), 0) + n

    def payoff(self, n_members: int) -> np.ndarray:
        out = np.full((n_members, n_members), 0.5, dtype = np.float64)
        for (i, j), total in self.sums.items():
            out[i, j] = total / self.counts[(i, j)]
            out[j, i] = 1.0 - out[i, j]
        return out


class PSRORun:
    def __init__(self, cfg: RunConfig, out_dir: Path):
        self.cfg = cfg
        self.out_dir = out_dir
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.game = load_game(cfg.env)
        self.scorer = make_scorer(self.game)
        self.span = score_span(self.game)
        self.exact_env = cfg.env not in FULL_TREE_INTRACTABLE
        self.ledger = current_ledger()
        self.schedule = EvalSchedule(cfg)
        self.cache = TableCache(self.game)
        obs_dim, n_actions = obs_and_action_size(self.game)
        self.members: list[Any] = [frozen_actor(SmallMLP(obs_dim, n_actions, cfg.hidden), cfg.device)]
        self.meta = _MetaMatrix()
        self.weights = np.ones(1, dtype = np.float64)
        self.log_path = out_dir / "rollouts.jsonl"
        self.log_path.unlink(missing_ok = True)

    def _br_cost(self) -> int:
        return 0 if self.cfg.oracle == "exact_br" else exploiter_rollouts(self.cfg) * self.cfg.games_per_rollout

    def _resolve_meta(self) -> None:
        self.weights = meta_nash_weights(self.meta.payoff(len(self.members)))

    def _new_best_response(self) -> Any:
        mixture = MixtureActor(self.members, self.weights)
        if self.cfg.oracle == "exact_br":
            # 与 exact_br_actor(game, mixture) 逐位相同：冻结成员的表格取自缓存，只聚合一次
            tables = [self.cache.get(str(i), m) for i, m in enumerate(self.members)]
            return exact_br_from_table(self.game, aggregate(self.game, tables, mixture.weights))
        # 与 OpenSpiel PSRO 一致：按元策略抽一个成员，抽到已训练的 BR 就继承它；第 0 个成员是随机网络，从零开始
        parent = int(self.rng.choice(len(self.members), p = self.weights))
        source = self.members[parent]
        warm = source.net if parent > 0 and isinstance(source, TorchActor) else None
        with game_kind("br_train"):
            return train_ppo_br(self.cfg, self.game, mixture, self.rng, self.scorer, warm_from = warm)

    def _evaluate(self, gen: int) -> None:
        rec: dict[str, Any] = dict(gen = gen, games = self.ledger.algorithmic_total, total_games = self.ledger.total,
                                   n_members = len(self.members), support = int(np.sum(self.weights > 1e-12)),
                                   evaluation = True)
        if self.exact_env:
            tables = [self.cache.get(str(i), m) for i, m in enumerate(self.members)]
            expl = exploitability_of_table(self.game, aggregate(self.game, tables, self.weights))
            rec.update(exploitability = expl, norm_exploitability = expl / self.span)
        else:
            self._save_mixture()
            rec.update(exploitability = float("nan"), norm_exploitability = float("nan"))
        append_jsonl(self.log_path, rec)

    def _save_mixture(self) -> None:
        ckpt_dir = self.out_dir / "ckpts"
        ckpt_dir.mkdir(parents = True, exist_ok = True)
        states = [m.net.state_dict() for m in self.members]
        torch.save(dict(states = states, weights = self.weights.tolist()),
                   ckpt_dir / f"games_{self.ledger.algorithmic_total:010d}.pt")

    def _refine(self) -> None:
        """把剩余预算按轮转分给已有成员对，加密元矩阵估计。"""
        pairs = sorted(self.meta.counts)
        k = 0
        while pairs and self.ledger.algorithmic_total < self.cfg.budget:
            left = self.cfg.budget - self.ledger.algorithmic_total
            n = min(max(left // len(pairs), 1), left)
            i, j = pairs[k % len(pairs)]
            with game_kind("meta_matrix"):
                self.meta.play(self.game, self.members, i, j, n, self.rng, self.scorer)
            k += 1

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        t0 = time.time()
        log(f"start PSRO {cfg.env} {cfg.label} seed={cfg.seed} budget={cfg.budget}")
        self._evaluate(-1)
        gen = 0
        while True:
            games_before = self.ledger.algorithmic_total
            next_cost = self._br_cost() + cfg.meta_games * len(self.members)
            if games_before + next_cost > cfg.budget:
                break
            self.members.append(self._new_best_response())
            with game_kind("meta_matrix"):
                for i in range(len(self.members) - 1):
                    self.meta.play(self.game, self.members, i, len(self.members) - 1, cfg.meta_games, self.rng, self.scorer)
            self._resolve_meta()
            games_after = self.ledger.algorithmic_total
            if self.schedule.crossed(games_before, games_after) or self.schedule.in_tail(games_after):
                self._evaluate(gen)
            gen += 1
        self._refine()
        self._resolve_meta()
        self._evaluate(gen)
        summary = dict(generations = gen, n_members = len(self.members), games = self.ledger.algorithmic_total,
                       total_games = self.ledger.total, breakdown = dict(self.ledger.counts), elapsed_s = time.time() - t0)
        write_json(self.out_dir / "run_summary.json", summary)
        return summary


@measured_run
def run_psro(cfg: RunConfig, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents = True, exist_ok = True)
    return PSRORun(cfg, out_dir).run()
