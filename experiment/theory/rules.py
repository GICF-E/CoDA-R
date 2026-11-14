"""池规则在有限矩阵博弈上的轨迹（T10–T12）：共用的驱动、oracle 与对抗调度。

CoDA-R 与移动锚按论文的隐式配对求解；PFSP / 均匀 / Hedge 的池系数取步首 x_k 处的精确败率（它们没有配对形式）。
池的增删全部交给调度对象：oracle 周期性加入锚的（近似）最佳响应，对抗调度按各自的策略构造成员。
定理 3 与定理 9 对任意池序列成立，所以 CoDA-R 在所有调度下都逐步检查能量式。
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from experiment.core.pool_rules import HedgeRule, PFSPRule, PoolRule, UniformRule
from experiment.core.simplex import simplex_proj
from experiment.theory.games import MatrixGame, best_response, exploitability, make_game
from experiment.theory.solver import Solution, certificate, solve_step

RULES = ("coda_r", "moving_anchor", "pfsp", "uniform", "hedge", "none")
# 隐式配对求解的规则；其余规则的池系数在步首取定
PAIRED_RULES = frozenset({"coda_r", "moving_anchor", "none"})
SERIES_EVERY = 8
# 败率与 1/2 相差不到这么多的成员算作打平
TIE_TOL = 1e-3


def make_rule(name: str) -> PoolRule | None:
    """固定份额规则沿用 T10 的参数：自博弈份额 0.5、PFSP 指数 2、Hedge beta 1。"""
    return {"pfsp": PFSPRule(2.0, 0.5), "uniform": UniformRule(0.5), "hedge": HedgeRule(1.0, 0.5)}.get(name)


def epsilon_br(s: np.ndarray, x: np.ndarray, eps: float) -> np.ndarray:
    """对 x 的领先恰为 Expl(x) - eps 的对手（eps = 0 为精确 BR）：在最佳与最差纯策略之间插值。"""
    gains = x @ s
    best, worst = int(np.argmax(gains)), int(np.argmin(gains))
    gap = float(gains[best] - gains[worst])
    eye = np.eye(s.shape[0])
    if eps <= 0.0 or gap <= 0.0:
        return eye[:, best]
    t = min(eps / gap, 1.0)
    return (1.0 - t) * eye[:, best] + t * eye[:, worst]


# -------- 池与调度 --------
@dataclass
class MatrixPool:
    """池成员按加入顺序排列，每个成员是一个混合策略。"""
    m: int
    ids: list[str] = field(default_factory = list)
    members: list[np.ndarray] = field(default_factory = list)

    @property
    def columns(self) -> np.ndarray:
        return np.column_stack(self.members) if self.members else np.zeros((self.m, 0))

    def add(self, member_id: str, y: np.ndarray) -> None:
        self.ids.append(member_id)
        self.members.append(np.asarray(y, dtype = np.float64))

    def remove(self, member_id: str) -> None:
        i = self.ids.index(member_id)
        del self.ids[i], self.members[i]

    def clear(self) -> None:
        self.ids.clear()
        self.members.clear()


def initial_pool(game: MatrixGame) -> MatrixPool:
    pool = MatrixPool(game.m)
    for i in range(game.pool.shape[1]):
        pool.add(f"init_{i}", game.pool[:, i])
    return pool


class Schedule(Protocol):
    def update(self, k: int, x: np.ndarray, pool: MatrixPool) -> None:
        """第 k 步求解之前调整池；x 是本步的锚 x_k。"""
        ...


class StaticPool:
    def update(self, k: int, x: np.ndarray, pool: MatrixPool) -> None:
        return


@dataclass
class OracleSchedule:
    """
    第 first, first + every, ... 步（不晚于 last）加入一个 eps-BR；lag > 0 时针对 lag 步之前的锚（滞后的 oracle）。
    keep 限制池里只保留最近加入的 keep 个 oracle 成员，其它成员不受影响。
    """
    s: np.ndarray
    every: int
    first: int = 1
    last: int | None = None
    eps: float = 0.0
    lag: int = 0
    keep: int | None = None
    _anchors: list[np.ndarray] = field(default_factory = list)
    _added: list[str] = field(default_factory = list)

    def update(self, k: int, x: np.ndarray, pool: MatrixPool) -> None:
        self._anchors.append(x.copy())
        due = k >= self.first and (k - self.first) % self.every == 0 and (self.last is None or k <= self.last)
        if not due or k - self.lag < 1:
            return
        member_id = f"br_{k}"
        pool.add(member_id, epsilon_br(self.s, self._anchors[k - 1 - self.lag], self.eps))
        self._added.append(member_id)
        while self.keep is not None and len(self._added) > self.keep:
            pool.remove(self._added.pop(0))


@dataclass
class ComplementAdversary:
    """每步把池换成锚关于纳什的补 y = Pi(2 x* - x_k)：自博弈份额 1/2 的固定份额规则在补没有被截断时整步停在 x_k。"""
    nash: np.ndarray

    def update(self, k: int, x: np.ndarray, pool: MatrixPool) -> None:
        pool.clear()
        pool.add("complement", simplex_proj(2.0 * self.nash - x))


@dataclass
class FloodAdversary:
    """每步加入一个随机混合策略直到池里有 size 个成员；有用的 oracle 成员照常加入，但被大量无用成员稀释。"""
    rng: np.random.Generator
    size: int
    oracle: OracleSchedule

    def update(self, k: int, x: np.ndarray, pool: MatrixPool) -> None:
        if len(pool.ids) < self.size:
            pool.add(f"junk_{k}", self.rng.dirichlet(np.ones(pool.m)))
        self.oracle.update(k, x, pool)


ADVERSARIES = ("benign", "complement", "flood", "support", "stale")
ADVERSARY_EVERY = 64
FLOOD_SIZE = 64
STALE_LAG = 128


def adversary(name: str, game: MatrixGame) -> tuple[MatrixPool, Schedule]:
    """
    (初始池, 调度)。benign：T10 式的周期 oracle；complement：每步换成锚关于纳什的补；
    flood：随机成员灌满池，oracle 照常；support：纳什支撑上的全部纯策略（在纳什处打平），不再变化；
    stale：oracle 只给出 STALE_LAG 步之前的锚的 BR。
    """
    s = game.s
    if name == "benign":
        return initial_pool(game), OracleSchedule(s, ADVERSARY_EVERY, first = ADVERSARY_EVERY + 1)
    if name == "complement":
        return MatrixPool(game.m), ComplementAdversary(game.nash)
    if name == "flood":
        oracle = OracleSchedule(s, ADVERSARY_EVERY, first = ADVERSARY_EVERY + 1)
        return initial_pool(game), FloodAdversary(game.rng, FLOOD_SIZE, oracle)
    if name == "support":
        pool = MatrixPool(game.m)
        for j in np.flatnonzero(game.nash > 1e-9):
            pool.add(f"pure_{j}", np.eye(game.m)[:, j])
        return pool, StaticPool()
    if name == "stale":
        return initial_pool(game), OracleSchedule(s, ADVERSARY_EVERY, first = STALE_LAG + 1, lag = STALE_LAG)
    raise ValueError(f"unknown adversary {name!r}")


# -------- 驱动 --------
@dataclass
class RuleStep:
    k: int
    # 本步的锚 x_k（重置之后）与求解结果 x_{k+1}
    anchor: np.ndarray
    solution: Solution
    ids: list[str]
    # 本步的池（每列一个成员）、成员的训练场系数，以及新点 x_{k+1} 处的精确败率
    pool: np.ndarray
    coef: np.ndarray
    losses: np.ndarray


def trajectory(s: np.ndarray, x0: np.ndarray, rule_name: str, c: float, eta: float, steps: int, pool: MatrixPool,
               schedule: Schedule, reset_at: int | None = None) -> Iterator[RuleStep]:
    """逐步产出 RuleStep；reset_at 那一步在求解前把 latest 重置回 x0。'none' 不用池（纯近端自博弈）。"""
    rule = make_rule(rule_name)
    moving_anchor: dict[str, float] = {}
    mirror_anchor = 1.0
    x = np.array(x0, dtype = np.float64)
    m = s.shape[0]
    for k in range(1, steps + 1):
        schedule.update(k, x, pool)
        if k == reset_at:
            x = np.array(x0, dtype = np.float64)
        ids = list(pool.ids) if rule_name != "none" else []
        y = pool.columns if rule_name != "none" else np.zeros((m, 0))
        if rule_name in PAIRED_RULES:
            moving = rule_name == "moving_anchor"
            # 新成员的锚分量为 0；离开池的成员带走它的锚质量
            center = np.array([moving_anchor.get(i, 0.0) for i in ids] + [mirror_anchor]) if moving else None
            sol = solve_step(s, y, x, eta, c, anchor = center)
            coef = sol.w[:-1]
            if moving:
                moving_anchor = dict(zip(ids, (float(w) for w in sol.w[:-1])))
                mirror_anchor = float(sol.w[-1])
        else:
            assert rule is not None
            field_coef = rule.output(ids, 0.5 + (s @ y).T @ x).field
            sol = solve_step(s, y, x, eta, c, share = float(field_coef[-1]), frozen_weights = field_coef)
            coef = field_coef[:-1]
            rule.end_rollout(ids, 0.5 + (s @ y).T @ sol.x)
        yield RuleStep(k = k, anchor = x, solution = sol, ids = ids, pool = y, coef = coef, losses = 0.5 + (s @ y).T @ sol.x)
        x = sol.x


def _tied_coef(step: RuleStep) -> float:
    return float(np.sum(step.coef[np.abs(step.losses - 0.5) < TIE_TOL])) if step.coef.size else 0.0


def _idle_coef(step: RuleStep) -> float:
    """新点处没有在赢 latest 的成员（打平或被打败）持有的系数。"""
    return float(np.sum(step.coef[step.losses <= 0.5 + TIE_TOL])) if step.coef.size else 0.0


# -------- T10：sleeper 延迟 --------
def sleeper_run(p: dict[str, Any], seed: int) -> dict[str, Any]:
    """周期 oracle 加到 reset_at 为止，之后把 latest 重置回 x0，测"最能打 x0 的成员"重新拿到 0.05 以上系数所需的步数。"""
    game = make_game("random", p["m"], p["n"], seed)
    s = game.s
    x0 = game.rng.dirichlet(np.ones(game.m))
    if p["rule"] == "none":
        pool, schedule = MatrixPool(game.m), StaticPool()
    else:
        pool = initial_pool(game)
        schedule = OracleSchedule(s, p["insert_every"], first = p["insert_every"] + 1, last = p["reset_at"])
    expl, tied = [], []
    threat: str | None = None
    delay: int | None = None
    for step in trajectory(s, x0, p["rule"], p["c"], p["eta"], p["steps"], pool, schedule, reset_at = p["reset_at"]):
        if step.k == p["reset_at"] and step.ids:
            threat = step.ids[int(np.argmax(0.5 + (s @ step.pool).T @ x0))]
        expl.append(exploitability(s, step.solution.x))
        tied.append(_tied_coef(step))
        if threat is not None and delay is None and step.coef[step.ids.index(threat)] >= 0.05:
            delay = step.k - p["reset_at"]
    before_reset = slice(p["reset_at"] // 2, p["reset_at"] - 1)
    return dict(seed = seed, final_expl = expl[-1], expl_before_reset = expl[p["reset_at"] - 2],
                tied_weight_before_reset = float(np.mean(tied[before_reset])),
                sleeper_delay = float(delay) if delay is not None else float("nan"),
                sleeper_censored = threat is not None and delay is None, series = dict(full_expl = expl[::SERIES_EVERY]))


# -------- T11：oracle 的定量收益（定理 9）--------
# 臂 -> 池规则；none 不用池，coda_r_eps 的 oracle 误差取 payload 的 eps，其余臂用精确 BR
ORACLE_ARMS = {"none": "none", "coda_r": "coda_r", "coda_r_eps": "coda_r", "pfsp": "pfsp", "uniform": "uniform"}
# 池里只保留最近加入的这么多个 oracle 成员；定理 9 允许池里有任意其它成员
ORACLE_KEEP = 8
RATE_WINDOW = (1e-10, 1e-1)


def theorem9_c(s: np.ndarray, eta: float) -> float:
    """定理 9 允许的最大对偶强度：c <= 4 且 c ||S||^2 <= 2 eta。"""
    return float(min(4.0, 2.0 * eta / np.linalg.norm(s, 2) ** 2))


def log_rate(values: list[float], window: tuple[float, float] = RATE_WINDOW) -> float:
    """每步的对数收缩率 -d ln Expl / dk：对落在窗口内的点做最小二乘；点太少时用首尾两点。"""
    ys = np.asarray(values, dtype = np.float64)
    ks = np.arange(1, ys.size + 1, dtype = np.float64)
    inside = (ys > window[0]) & (ys < window[1])
    if inside.sum() >= 8:
        slope = np.polyfit(ks[inside], np.log(ys[inside]), 1)[0]
        return float(-slope)
    floor = np.maximum(ys, window[0])
    return float(np.log(floor[0] / floor[-1]) / max(ys.size - 1, 1))


def _first_below(values: list[float], target: float) -> float:
    return next((float(k) for k, v in enumerate(values, start = 1) if v <= target), float("nan"))


def oracle_rate_run(p: dict[str, Any], seed: int) -> dict[str, Any]:
    """每步把锚的 eps-BR 放进池（只保留最近 ORACLE_KEEP 个），c 取定理 9 允许的最大值；CoDA-R 臂逐步检查定理 9。"""
    game = make_game(p["family"], p["size"], 0, seed)
    s, nash, eta = game.s, game.nash, p["eta"]
    c = theorem9_c(s, eta)
    rule_name = ORACLE_ARMS[p["arm"]]
    eps = float(p["eps"]) if p["arm"] == "coda_r_eps" else 0.0
    x0 = game.rng.dirichlet(np.ones(game.m))
    schedule: Schedule = StaticPool() if rule_name == "none" else OracleSchedule(s, 1, eps = eps, keep = ORACLE_KEEP)
    expl: list[float] = []
    violations = 0
    for step in trajectory(s, x0, rule_name, c, eta, p["steps"], MatrixPool(game.m), schedule):
        after = step.solution.x
        expl.append(exploitability(s, after))
        if rule_name == "coda_r":
            decrease = float(np.sum((step.anchor - nash) ** 2) - np.sum((after - nash) ** 2))
            bound = c / (4.0 * eta) * max(exploitability(s, step.anchor) - eps, 0.0) ** 2
            # 求解器的 FW 间隙 delta 按定理 7 进入能量式：右端多出 2 delta / eta
            violations += int(decrease < bound - 2.0 * step.solution.delta / eta - 1e-9)
    return dict(seed = seed, c = c, final_expl = expl[-1], rate = log_rate(expl),
                steps_to_1e3 = _first_below(expl, 1e-3), steps_to_1e6 = _first_below(expl, 1e-6),
                violations = dict(theorem9_violation = violations) if rule_name == "coda_r" else {},
                series = dict(full_expl = expl[::SERIES_EVERY]))


# -------- T12：对抗池 --------
def adversarial_run(p: dict[str, Any], seed: int) -> dict[str, Any]:
    """对手按 adversary 构造池；CoDA-R 在每一步检查能量式（定理 1 对任意池成立）。"""
    game = make_game(p["family"], p["m"], 4, seed)
    s, nash, c, eta = game.s, game.nash, p["c"], p["eta"]
    x0 = game.rng.dirichlet(np.ones(game.m))
    pool, schedule = adversary(p["adversary"], game)
    expl, idle = [], []
    violations = 0
    largest = 0
    for step in trajectory(s, x0, p["rule"], c, eta, p["steps"], pool, schedule):
        expl.append(exploitability(s, step.solution.x))
        idle.append(_idle_coef(step))
        largest = max(largest, len(step.ids))
        if p["rule"] == "coda_r":
            rec = certificate(s, step.pool, step.anchor, step.solution, nash, eta, c, 1.0)
            violations += int(rec["fejer_violation"])
    tail = expl[len(expl) * 3 // 4:]
    return dict(seed = seed, final_expl = expl[-1], tail_mean_expl = float(np.mean(tail)), idle_coef = float(np.mean(idle)),
                largest_pool = largest, violations = dict(fejer_violation = violations) if p["rule"] == "coda_r" else {},
                series = dict(full_expl = expl[::SERIES_EVERY]))
