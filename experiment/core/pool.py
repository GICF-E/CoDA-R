"""对手池：成员管理、按规则出权重、采样分布与重要性比、败率结账。

每个 rollout 的调用顺序：begin_rollout → sample / record 若干局 → end_rollout；
每个外环结账时调用 end_outer_step，之后才能 add / remove 成员。

采样分布 r 只影响方差和每个成员分到的局数；每局样本乘 iw = field_i / r_i，
期望更新恒等于训练场，与 r 无关（论文 §3.3）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiment.core.estimator import LossEstimator
from experiment.core.pool_rules import PoolRule, RuleOutput

# 镜像臂（latest 自己）的保留 ID，池成员不能使用
MIRROR_ID = "self"
SAMPLING_BASES = ("dual", "field", "half")
DETECTIONS = ("probe", "none", "sweep")


@dataclass(frozen = True)
class SamplingConfig:
    # 混入的均匀探索比例
    eps: float = 0.05
    # 自博弈臂的采样下限，重要性比 1 / r_m 不超过它的倒数
    mirror_floor: float = 0.1
    # 零权重成员的发现机制：probe = 保底局数加按上置信界分配的探测预算；none = 只靠 eps；
    # sweep = 由训练循环在外环结账时对每个成员打固定局数（不进梯度）
    detection: str = "probe"
    member_floor_games: int = 8
    probe_share: float = 0.25
    probe_z: float = 2.0
    # 采样基：dual = 对偶权重 (w_P, w_m)；field = 归一化训练场；half = 两者各半
    base: str = "dual"

    def __post_init__(self) -> None:
        if self.base not in SAMPLING_BASES:
            raise ValueError(f"unknown sampling base {self.base!r}")
        if self.detection not in DETECTIONS:
            raise ValueError(f"unknown detection {self.detection!r}")
        if not 0.0 < self.mirror_floor < 1.0:
            raise ValueError("mirror_floor must lie in (0, 1)")
        if not 0.0 <= self.probe_share < 1.0 - self.mirror_floor:
            raise ValueError("probe_share plus mirror_floor must be below 1")


@dataclass
class Member:
    id: str
    policy: Any
    # snapshot = 历史快照；oracle = exploiter；external = 实验预先放入的外部对手
    kind: str


@dataclass(frozen = True)
class RolloutPlan:
    """一个 rollout 内冻结的训练场与采样分布；最后一维是镜像臂。"""
    ids: list[str]
    field: np.ndarray
    dual: np.ndarray
    probs: np.ndarray

    @property
    def field_mass(self) -> float:
        return float(self.field.sum())

    def importance_ratio(self, arm: int) -> float:
        return float(self.field[arm] / self.probs[arm])


@dataclass
class _Pending:
    counts: dict[str, int] = field(default_factory = dict)
    loss_sums: dict[str, float] = field(default_factory = dict)
    iw_sum: float = 0.0
    iw_sq_sum: float = 0.0
    n_games: int = 0


class OpponentPool:
    def __init__(
        self,
        rule: PoolRule,
        estimator: LossEstimator,
        sampling: SamplingConfig,
        capacity: int,
        protect_recent: int = 2,
        freeze_field: bool = False,
    ):
        self.rule = rule
        self.estimator = estimator
        self.sampling = sampling
        self.capacity = int(capacity)
        self.protect_recent = int(protect_recent)
        # 消融：训练场系数在整个外环内冻结为外环开始时的值，采样仍按当前对偶权重
        self.freeze_field = bool(freeze_field)
        self.members: list[Member] = []
        self._frozen_field: dict[str, float] | None = None
        self._plan: RolloutPlan | None = None
        self._pending = _Pending()

    # -------- 成员 --------
    @property
    def ids(self) -> list[str]:
        return [m.id for m in self.members]

    def member(self, member_id: str) -> Member:
        return next(m for m in self.members if m.id == member_id)

    def add(self, member_id: str, policy: Any, kind: str, initial_loss: float = 0.5) -> list[str]:
        """加入成员；满池时先淘汰一个，返回被淘汰的 ID。"""
        if self._plan is not None:
            raise RuntimeError("members can only change between rollouts")
        if member_id == MIRROR_ID or member_id in self.ids:
            raise ValueError(f"member id {member_id!r} is reserved or already in the pool")
        evicted = []
        if len(self.members) >= self.capacity:
            evicted.append(self._evict())
        self.members.append(Member(member_id, policy, kind))
        self.estimator.add(member_id, initial_loss)
        return evicted

    def remove(self, member_id: str) -> None:
        if self._plan is not None:
            raise RuntimeError("members can only change between rollouts")
        self.members = [m for m in self.members if m.id != member_id]
        self.estimator.remove(member_id)

    def _evict(self) -> str:
        """
        淘汰训练场系数最小的成员（同值时先淘汰败率估计更低、更早入池的），最近入池的 protect_recent 个受保护。
        理论允许任意增删成员（推论 3.4），所以满池时也可以淘汰正权重成员。
        """
        out = self.output()
        losses = self.estimator.losses(self.ids)
        n_open = max(len(self.members) - self.protect_recent, 1)
        victim = min(range(n_open), key = lambda i: (float(out.field[i]), float(losses[i]), i))
        victim_id = self.members[victim].id
        self.remove(victim_id)
        return victim_id

    # -------- 权重 --------
    def output(self) -> RuleOutput:
        ids = self.ids
        return self.rule.output(ids, self.estimator.losses(ids))

    def _training_field(self, out: RuleOutput) -> np.ndarray:
        if not self.freeze_field:
            return out.field
        if self._frozen_field is None:
            self._frozen_field = {i: float(w) for i, w in zip(self.ids, out.field[:-1])}
            self._frozen_field[MIRROR_ID] = float(out.field[-1])
        frozen = self._frozen_field
        return np.array([frozen.get(i, 0.0) for i in self.ids] + [frozen[MIRROR_ID]], dtype = np.float64)

    def _member_floors(self, games: int) -> np.ndarray:
        n = len(self.members)
        if n == 0 or self.sampling.detection != "probe":
            return np.zeros(n, dtype = np.float64)
        cfg = self.sampling
        base = min(cfg.member_floor_games / float(games), cfg.probe_share / n)
        floors = np.full(n, base, dtype = np.float64)
        remainder = cfg.probe_share - base * n
        priority = self.estimator.probe_priority(self.ids, cfg.probe_z)
        if remainder > 0.0 and float(priority.sum()) > 0.0:
            floors += remainder * priority / float(priority.sum())
        return floors

    def _sampling_probs(self, out: RuleOutput, train_field: np.ndarray, games: int) -> np.ndarray:
        """采样基混入 eps 均匀探索，再抬到各臂下限，多出的质量按高于下限的部分等比例扣回。"""
        cfg = self.sampling
        normalized_field = train_field / float(train_field.sum())
        base = {"dual": out.dual, "field": normalized_field, "half": 0.5 * (out.dual + normalized_field)}[cfg.base]
        r = (1.0 - cfg.eps) * base + cfg.eps / float(base.size)
        floors = np.append(self._member_floors(games), cfg.mirror_floor)
        r = np.maximum(r, floors)
        extra = float(r.sum()) - 1.0
        if extra > 0.0:
            slack = r - floors
            r = r - extra * slack / float(slack.sum())
        r = np.maximum(r, 0.0)
        r = r / float(r.sum())
        if np.any((train_field > 0.0) & (r <= 0.0)):
            raise RuntimeError(f"sampling distribution does not cover the training field: field={train_field}, r={r}")
        return r

    # -------- rollout --------
    def begin_rollout(self, games: int) -> RolloutPlan:
        if self._plan is not None:
            raise RuntimeError("previous rollout was not finished")
        out = self.output()
        train_field = self._training_field(out)
        probs = self._sampling_probs(out, train_field, games)
        self._plan = RolloutPlan(ids = self.ids, field = train_field, dual = out.dual, probs = probs)
        self._pending = _Pending()
        return self._plan

    def sample(self, rng: np.random.Generator) -> tuple[Member | None, str, float]:
        """返回 (对手成员，镜像臂时为 None；臂 ID；重要性比)。"""
        plan = self._plan
        if plan is None:
            raise RuntimeError("sample called outside a rollout")
        arm = int(rng.choice(plan.probs.size, p = plan.probs))
        iw = plan.importance_ratio(arm)
        self._pending.iw_sum += iw
        self._pending.iw_sq_sum += iw * iw
        self._pending.n_games += 1
        if arm == len(plan.ids):
            return None, MIRROR_ID, iw
        member_id = plan.ids[arm]
        return self.member(member_id), member_id, iw

    def record(self, member_id: str, loss: float) -> None:
        if member_id == MIRROR_ID:
            return
        pending = self._pending
        pending.counts[member_id] = pending.counts.get(member_id, 0) + 1
        pending.loss_sums[member_id] = pending.loss_sums.get(member_id, 0.0) + float(loss)

    def end_rollout(self, exact_losses: dict[str, float] | None = None) -> dict[str, float]:
        """把本 rollout 的对局并入估计（或用精确败率覆盖），更新有状态规则，返回本 rollout 的诊断量。"""
        plan = self._plan
        if plan is None:
            raise RuntimeError("end_rollout called outside a rollout")
        pending = self._pending
        resets = 0
        for member_id, count in pending.counts.items():
            resets += int(self.estimator.update(member_id, count, pending.loss_sums[member_id] / count))
        for member_id, loss in (exact_losses or {}).items():
            self.estimator.set_exact(member_id, loss)
        ids = self.ids
        self.rule.end_rollout(ids, self.estimator.losses(ids))
        self._plan = None
        n = max(pending.n_games, 1)
        zero_field = plan.field[:-1] == 0.0
        return dict(
            pool_mass = float(1.0 - plan.dual[-1]),
            field_mass = plan.field_mass,
            pool_sample_share = float(1.0 - plan.probs[-1]),
            # 采样到训练场系数为 0 的成员的份额：这些局 iw = 0，只用于估计，不进梯度
            no_gradient_share = float(plan.probs[:-1][zero_field].sum()),
            ess_fraction = float(pending.iw_sum ** 2 / (n * pending.iw_sq_sum)) if pending.iw_sq_sum > 0 else 1.0,
            dual_distance = RuleOutput(plan.field, plan.dual).dual_distance,
            support = float(np.sum(plan.field[:-1] > 0.0)),
            resets = float(resets),
        )

    def end_outer_step(self) -> None:
        ids = self.ids
        self.rule.end_outer_step(ids, self.estimator.losses(ids))
        self._frozen_field = None

    def add_observations(self, member_id: str, count: int, mean_loss: float) -> None:
        """外环全池扫描的对局只更新估计，不进梯度。"""
        self.estimator.update(member_id, count, mean_loss)

    def snapshot(self) -> dict[str, list[Any]]:
        """日志用：当前成员、败率估计、训练场系数与有效局数。"""
        ids = self.ids
        out = self.output()
        return dict(
            pool_ids = ids,
            pool_losses_est = self.estimator.losses(ids).tolist(),
            pool_field = out.field[:-1].tolist(),
            pool_n_eff = self.estimator.n_eff(ids).tolist(),
        )
