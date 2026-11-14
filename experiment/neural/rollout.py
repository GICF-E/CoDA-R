"""一个 rollout 的采样：按池的采样分布抽对手打完若干局，记录败率并给每个样本挂上重要性比。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment.core.pool import OpponentPool
from experiment.neural.envs import Scorer
from experiment.neural.play import play_episode
from experiment.neural.ppo import PPOLearner, Step


@dataclass
class RolloutData:
    steps: list[Step]
    scores: list[float]
    # 本 rollout 训练场的总质量（自博弈份额加池系数之和）
    field_mass: float


def collect_rollout(
    game: Any,
    learner: PPOLearner,
    pool: OpponentPool | None,
    mirror_snapshot: Any,
    rng: np.random.Generator,
    scorer: Scorer,
    games: int,
) -> RolloutData:
    """
    pool 为 None 时是纯自博弈。mirror_snapshot 为 None 表示镜像对手直接用 learner 当前参数
    （rollout 内参数不变，等价于冻结副本）；否则用给定的周期快照。
    调用方在 PPO 更新之后负责 pool.end_rollout()。
    """
    steps: list[Step] = []
    scores: list[float] = []
    if pool is None:
        for _ in range(games):
            episode, score = play_episode(game, learner, mirror_snapshot, rng, 1.0, scorer, mirror = True)
            steps.extend(episode)
            scores.append(score)
        return RolloutData(steps, scores, 1.0)
    plan = pool.begin_rollout(games)
    for _ in range(games):
        member, arm_id, iw = pool.sample(rng)
        if member is None:
            episode, score = play_episode(game, learner, mirror_snapshot, rng, iw, scorer, mirror = True)
        else:
            episode, score = play_episode(game, learner, member.policy, rng, iw, scorer, mirror = False)
        # 败率估计只吃原始得分；塑形收益之后才写进 steps
        pool.record(arm_id, 1.0 - score)
        steps.extend(episode)
        scores.append(score)
    return RolloutData(steps, scores, plan.field_mass)
