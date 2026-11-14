"""池权重规则：给定各成员的败率估计，返回训练场系数和对偶权重。

所有规则共用同一个池、估计器、采样修正和原侧近端项，只在这里不同，
因此 CoDA-R 与 PFSP 等方法的对比只隔离了"池权重怎么定"这一项。

约定：长度 N + 1 的向量里前 N 个是池成员，最后一个是自博弈（镜像臂）。
- field：原侧训练场的系数，即 (P) 里各对手的份额，总和不必为 1。
- dual：长度 N + 1、和为 1 的对偶权重。CoDA-R 是投影 (w_P, w_m)；其它规则取归一化的 field。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from experiment.core.simplex import simplex_proj


@dataclass(frozen = True)
class RuleOutput:
    field: np.ndarray
    dual: np.ndarray

    @property
    def pool_mass(self) -> float:
        # 对偶侧的池总权重 a = 1 - w_m
        return float(1.0 - self.dual[-1])

    @property
    def dual_distance(self) -> float:
        # ||w - e_m||，CoDA-R 证书中的对偶项
        anchor = np.zeros_like(self.dual)
        anchor[-1] = 1.0
        return float(np.linalg.norm(self.dual - anchor))


def _self_play_only() -> RuleOutput:
    one = np.ones(1, dtype = np.float64)
    return RuleOutput(field = one, dual = one.copy())


def _normalized(field: np.ndarray) -> RuleOutput:
    return RuleOutput(field = field, dual = field / float(field.sum()))


class PoolRule(ABC):
    """无状态规则只需实现 output；有状态规则在两个结账钩子里更新内部状态。"""

    name: str = "base"

    @abstractmethod
    def output(self, ids: list[str], losses: np.ndarray) -> RuleOutput:
        ...

    def end_rollout(self, ids: list[str], losses: np.ndarray) -> None:
        return

    def end_outer_step(self, ids: list[str], losses: np.ndarray) -> None:
        return


class CoDARRule(PoolRule):
    """CoDA-R：w = Pi(e_m + c * (l_1, ..., l_N, 1/2))，自博弈份额恒为 1。"""

    name = "coda_r"

    def __init__(self, c: float):
        self.c = float(c)

    def output(self, ids: list[str], losses: np.ndarray) -> RuleOutput:
        if not ids:
            return _self_play_only()
        dual = simplex_proj(np.append(self.c * losses, 1.0 + 0.5 * self.c))
        return RuleOutput(field = np.append(dual[:-1], 1.0), dual = dual)


class MovingAnchorRule(PoolRule):
    """消融用：对偶锚 rho 在每个外环结账时移到当前 w，其余同 CoDA-R。"""

    name = "moving_anchor"

    def __init__(self, c: float):
        self.c = float(c)
        self.anchor: dict[str, float] = {}
        self.mirror_anchor = 1.0

    def _anchor_vector(self, ids: list[str]) -> np.ndarray:
        # 新成员的锚分量为 0；离开池的成员带走它的锚质量
        return np.array([self.anchor.get(i, 0.0) for i in ids] + [self.mirror_anchor], dtype = np.float64)

    def output(self, ids: list[str], losses: np.ndarray) -> RuleOutput:
        if not ids:
            return _self_play_only()
        dual = simplex_proj(self._anchor_vector(ids) + self.c * np.append(losses, 0.5))
        return RuleOutput(field = np.append(dual[:-1], 1.0), dual = dual)

    def end_outer_step(self, ids: list[str], losses: np.ndarray) -> None:
        dual = self.output(ids, losses).dual
        self.anchor = {i: float(w) for i, w in zip(ids, dual[:-1])}
        self.mirror_anchor = float(dual[-1])


class PFSPRule(PoolRule):
    """AlphaStar 的 PFSP：池内权重正比于 f(p) = (1 - p)^k，p 为 latest 的胜率，即 l^k；自博弈份额固定。"""

    name = "pfsp"

    def __init__(self, exponent: float, self_share: float):
        self.exponent = float(exponent)
        self.self_share = float(self_share)

    def output(self, ids: list[str], losses: np.ndarray) -> RuleOutput:
        if not ids:
            return _self_play_only()
        priority = np.power(np.clip(losses, 0.0, 1.0), self.exponent)
        total = float(priority.sum())
        share = priority / total if total > 0.0 else np.full(len(ids), 1.0 / len(ids))
        return _normalized(np.append((1.0 - self.self_share) * share, self.self_share))


class UniformRule(PoolRule):
    """均匀 FSP：池内均分，自博弈份额固定（OpenAI Five 的 80/20 即 self_share = 0.8）。"""

    name = "uniform"

    def __init__(self, self_share: float):
        self.self_share = float(self_share)

    def output(self, ids: list[str], losses: np.ndarray) -> RuleOutput:
        if not ids:
            return _self_play_only()
        share = (1.0 - self.self_share) / float(len(ids))
        return _normalized(np.append(np.full(len(ids), share), self.self_share))


class HedgeRule(PoolRule):
    """Hedge / MWU：累计领先 s_i += beta * (l_i - 1/2)，池内权重正比于 exp(s_i)；自博弈份额固定。"""

    name = "hedge"

    def __init__(self, beta: float, self_share: float):
        self.beta = float(beta)
        self.self_share = float(self_share)
        self.scores: dict[str, float] = {}

    def output(self, ids: list[str], losses: np.ndarray) -> RuleOutput:
        if not ids:
            return _self_play_only()
        scores = np.array([self.scores.get(i, 0.0) for i in ids], dtype = np.float64)
        weights = np.exp(scores - scores.max())
        share = weights / float(weights.sum())
        return _normalized(np.append((1.0 - self.self_share) * share, self.self_share))

    def end_rollout(self, ids: list[str], losses: np.ndarray) -> None:
        for member_id, loss in zip(ids, losses):
            self.scores[member_id] = self.scores.get(member_id, 0.0) + self.beta * (float(loss) - 0.5)


class FixedRule(PoolRule):
    """固定比例：按成员 ID 给定份额，自博弈取剩余；池里不在表中的成员份额为 0。"""

    name = "fixed"

    def __init__(self, member_shares: Mapping[str, float]):
        self.member_shares = dict(member_shares)
        if sum(self.member_shares.values()) >= 1.0:
            raise ValueError("fixed member shares must leave a positive self-play share")

    def output(self, ids: list[str], losses: np.ndarray) -> RuleOutput:
        shares = np.array([self.member_shares.get(i, 0.0) for i in ids], dtype = np.float64)
        return _normalized(np.append(shares, 1.0 - float(shares.sum())))
