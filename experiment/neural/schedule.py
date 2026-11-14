"""按预算比例定义的日程：熵系数、评估点与 oracle 周期。所有方法共用同一套定义。"""
from __future__ import annotations

import numpy as np

from experiment.neural.config import RunConfig

# 状态空间大、只有胜负反馈的环境，exploiter 需要更多训练轮数；对所有方法的 exploiter 一视同仁
ORACLE_ROLLOUT_MULT = {"leduc_poker": 3, "liars_dice": 2, "abrupt_dark_hex": 3, "abrupt_phantom_ttt": 3}


def entropy_coef(cfg: RunConfig, games_done: int) -> float:
    """g <= anneal_frac * B 时线性退火，之后保持 ent_final，末段指标因此在同一个熵终值下测得。"""
    frac = min(1.0, games_done / (cfg.ent_anneal_frac * cfg.budget))
    return cfg.ent_start + frac * (cfg.ent_final - cfg.ent_start)


def learning_rate(cfg: RunConfig, games_done: int) -> float:
    """learner 的学习率在整个预算上从 lr 线性降到 lr * lr_final_frac；lr_final_frac = 1 时恒定。"""
    frac = min(1.0, games_done / cfg.budget)
    return cfg.lr * (1.0 + frac * (cfg.lr_final_frac - 1.0))


def exploiter_rollouts(cfg: RunConfig) -> int:
    return cfg.oracle_rollouts * ORACLE_ROLLOUT_MULT.get(cfg.env, 1)


def oracle_cost(cfg: RunConfig) -> int:
    """一次 PPO oracle 的算法性对局：exploiter 训练加一场入池评估。"""
    return exploiter_rollouts(cfg) * cfg.games_per_rollout + cfg.oracle_match_games


def oracle_period(cfg: RunConfig) -> int:
    """oracle 每隔多少个外环触发一次，使 PPO oracle 的累计占比约为 oracle_share；精确 BR 用同一周期。"""
    if cfg.oracle == "none" or cfg.oracle_share <= 0.0:
        return 0
    games_per_outer = cfg.games_per_rollout * cfg.outer_rollouts
    return max(1, int(round(oracle_cost(cfg) / (cfg.oracle_share * games_per_outer))))


class EvalSchedule:
    """末段之前在 [B / 1000, anneal_frac * B] 上按对数间隔取点，末段每 tail_eval_rollouts 个 rollout 一次，结束时一次。"""

    def __init__(self, cfg: RunConfig):
        self.budget = cfg.budget
        self.tail_start = cfg.ent_anneal_frac * cfg.budget
        self.tail_every = max(cfg.tail_eval_rollouts, 1)
        low = max(cfg.budget / 1000.0, float(cfg.games_per_rollout))
        self.points = np.geomspace(low, self.tail_start, max(cfg.eval_points, 2)) if low < self.tail_start else np.array([])

    def crossed(self, games_before: int, games_after: int) -> bool:
        """这一步是否越过了某个对数评估点或预算终点。"""
        if games_after >= self.budget:
            return True
        return bool(np.any((self.points > games_before) & (self.points <= games_after)))

    def in_tail(self, games: int) -> bool:
        return games >= self.tail_start

    def due(self, games_before: int, games_after: int, roll: int) -> bool:
        if self.crossed(games_before, games_after):
            return True
        return self.in_tail(games_after) and (roll + 1) % self.tail_every == 0
