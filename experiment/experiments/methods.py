"""主实验的方法预设与调参网格。

每种需要调参的方法都有 9 个配置：熵终值 3 个 x 一个方法特有参数 3 个（NFSP 没有熵项，用两个自身参数）。
带池的方法（均匀 FSP、PFSP、CoDA-R）直接沿用 R-NaD-PPO 选出的 eta，原侧完全相同，只在池规则上不同；
熵终值同时作用于 learner 与它的 exploiter。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

from experiment.neural.config import RunConfig

ENT_FINALS = (0.01, 0.003, 0.001)
CODA_C = {"leduc_poker": (12.0, 24.0, 48.0), "goofspiel4": (3.0, 6.0, 15.0), "liars_dice": (3.0, 6.0, 15.0)}


@dataclass(frozen = True)
class Method:
    name: str
    title: str
    # single = 部署 latest 单网络；mixture = 部署多网络混合或平均网络
    deployment: str
    base: dict[str, Any] = field(default_factory = dict)
    # 与熵终值组合的方法特有参数；None 表示不调参（由别的方法的选择派生）
    param: str | None = None
    values: tuple[float, ...] = ()


_POOL = dict(proximal = "shaping", oracle = "ppo")
METHODS: dict[str, Method] = {m.name: m for m in (
    Method("selfplay", "PPO self-play", "single", dict(proximal = "none"), "lr", (1e-4, 3e-4, 1e-3)),
    Method("rnad", "R-NaD-PPO", "single", dict(proximal = "shaping"), "eta", (0.05, 0.1, 0.2)),
    Method("mmd", "MMD-PPO", "single", dict(proximal = "kl"), "eta", (0.05, 0.1, 0.2)),
    Method("ufsp", "Uniform FSP", "single", dict(rule = "uniform", **_POOL), "self_share", (0.5, 0.8, 0.9)),
    Method("pfsp", "PFSP", "single", dict(rule = "pfsp", **_POOL), "pfsp_exponent", (1.0, 2.0, 3.0)),
    Method("coda_r", "CoDA-R", "single", dict(rule = "coda_r", **_POOL), "coda_c"),
    Method("psro", "PSRO (PPO BR)", "mixture", dict(algorithm = "psro", oracle = "ppo"), "oracle_rollouts", (10, 20, 40)),
    Method("nfsp", "NFSP", "mixture", dict(algorithm = "nfsp"), "nfsp_anticipatory", (0.05, 0.1, 0.2)),
)}
NFSP_LRS = (1e-3, 3e-3, 1e-2)

# 两阶段调参：B 阶段的方法要用 A 阶段 R-NaD-PPO 选出的 eta
TUNE_STAGE_A = ("selfplay", "rnad", "mmd", "psro", "nfsp")
TUNE_STAGE_B = ("ufsp", "pfsp", "coda_r")
INHERITS_ETA = frozenset(TUNE_STAGE_B)

# 主实验额外的派生方法：沿用来源方法的选择，只改 oracle
DERIVED: dict[str, tuple[str, dict[str, Any], str, str]] = {
    "coda_r_no_oracle": ("coda_r", dict(oracle = "none"), "CoDA-R no oracle", "single"),
    "coda_r_exact_br": ("coda_r", dict(oracle = "exact_br"), "CoDA-R exact-BR oracle (upper bound)", "single"),
    "pfsp_exact_br": ("pfsp", dict(oracle = "exact_br"), "PFSP exact-BR oracle (upper bound)", "single"),
    "psro_exact_br": ("psro", dict(oracle = "exact_br"), "PSRO exact BR (upper bound)", "mixture"),
}
# 混合输出 q / Q 只对 CoDA-R 有定义
MIXTURE_METHODS = frozenset({"coda_r", "coda_r_no_oracle", "coda_r_exact_br"})


def grid(method: str, env: str) -> list[dict[str, Any]]:
    """一种方法在一个游戏上的 9 个调参配置。"""
    spec = METHODS[method]
    if method == "nfsp":
        return [dict(nfsp_anticipatory = a, nfsp_lr = lr) for a, lr in product(spec.values, NFSP_LRS)]
    values = CODA_C[env] if method == "coda_r" else spec.values
    assert spec.param is not None
    return [{"ent_final": ent, spec.param: value} for ent, value in product(ENT_FINALS, values)]


def params_label(params: dict[str, Any]) -> str:
    return ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(params.items()))


def build_config(method: str, env: str, seed: int, budget: int, params: dict[str, Any], label: str | None = None,
                 **extra: Any) -> RunConfig:
    """来源方法的基本设定 + 选出（或待试）的参数 + 实验额外覆盖。"""
    if method in DERIVED:
        source, overrides, title, _deployment = DERIVED[method]
        base = {**METHODS[source].base, **overrides}
    else:
        title = METHODS[method].title
        base = dict(METHODS[method].base)
    if method in MIXTURE_METHODS:
        base["evaluate_mixtures"] = True
    return RunConfig(env = env, label = label or title, seed = seed, budget = budget, **{**base, **params, **extra})


def title(method: str) -> str:
    return DERIVED[method][2] if method in DERIVED else METHODS[method].title


def deployment(method: str) -> str:
    return DERIVED[method][3] if method in DERIVED else METHODS[method].deployment


def source_method(method: str) -> str:
    return DERIVED[method][0] if method in DERIVED else method
