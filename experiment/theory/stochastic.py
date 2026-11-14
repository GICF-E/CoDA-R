"""T9：带采样的有限维 CoDA-R，用真实的池、估计器与采样分布代码，在真值随时可算的矩阵博弈上比较采样基。

每个 rollout 打 G 局：按池的采样分布抽一个臂，再从该臂的混合策略里抽一个纯策略 j，
梯度估计取 iw * S e_j（期望等于精确场），胜负按败率掷伯努利并送进估计器；之后做一次近端投影步。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment.core.estimator import LossEstimator
from experiment.core.pool import OpponentPool, SamplingConfig
from experiment.core.pool_rules import CoDARRule
from experiment.core.simplex import simplex_proj
from experiment.theory.games import best_response, exploitability, make_game


@dataclass(frozen = True)
class StochasticSetup:
    m: int
    c: float
    games: int
    base: str
    seed: int
    n_pool: int = 4
    eta: float = 0.1
    step_size: float = 0.05
    outer_steps: int = 128
    rollouts_per_outer: int = 8
    insert_every: int = 16
    match_games: int = 64


def _true_losses(s: np.ndarray, members: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 0.5 + (s @ members).T @ x


def run_stochastic(setup: StochasticSetup) -> dict[str, Any]:
    game = make_game("random", setup.m, setup.n_pool, setup.seed)
    s, rng = game.s, game.rng
    # 只比较采样基：关闭探测，零权重成员只靠 eps 被看见
    sampling = SamplingConfig(base = setup.base, detection = "none")
    pool = OpponentPool(CoDARRule(setup.c), LossEstimator(), sampling, capacity = 64)
    for i in range(setup.n_pool):
        pool.add(f"init_{i}", game.pool[:, i], "external")
    x = rng.dirichlet(np.ones(setup.m))
    anchor = x.copy()
    errors, ess, pool_share, expl, repair = [], [], [], [], []
    pending_repair: tuple[int, np.ndarray] | None = None
    roll = 0
    for outer in range(setup.outer_steps):
        for _ in range(setup.rollouts_per_outer):
            pool.begin_rollout(setup.games)
            grad = np.zeros(setup.m)
            for _ in range(setup.games):
                member, arm_id, iw = pool.sample(rng)
                opponent = x if member is None else member.policy
                j = int(rng.choice(setup.m, p = opponent))
                grad += iw * s[:, j]
                a = int(rng.choice(setup.m, p = x))
                pool.record(arm_id, float(rng.random() < 0.5 + s[a, j]))
            x = simplex_proj(x - setup.step_size * (grad / setup.games + setup.eta * (x - anchor)))
            stats = pool.end_rollout()
            members = np.column_stack([m.policy for m in pool.members])
            truth = _true_losses(s, members, x)
            field = pool.output().field[:-1]
            weighted = field > 0.01
            errors.append(float(np.max(np.abs(pool.estimator.losses(pool.ids) - truth)[weighted])) if weighted.any() else 0.0)
            ess.append(stats["ess_fraction"])
            pool_share.append(stats["pool_sample_share"])
            expl.append(exploitability(s, x))
            if pending_repair is not None and float(0.5 + pending_repair[1] @ s.T @ x) <= 0.51:
                repair.append(roll - pending_repair[0])
                pending_repair = None
            roll += 1
        anchor = x.copy()
        pool.end_outer_step()
        if (outer + 1) % setup.insert_every == 0:
            br = best_response(s, x)
            measured = float(np.mean(rng.random(setup.match_games) < 0.5 + float(x @ s @ br)))
            pool.add(f"br_{outer}", br, "oracle", initial_loss = measured)
            pending_repair = (roll, br)
    tail = expl[len(expl) * 3 // 4:]
    return dict(mean_weighted_error = float(np.mean(errors)), mean_ess = float(np.mean(ess)),
                mean_pool_share = float(np.mean(pool_share)), final_expl = expl[-1], tail_mean_expl = float(np.mean(tail)),
                mean_repair_rollouts = float(np.mean(repair)) if repair else float("nan"), n_repairs = len(repair))
