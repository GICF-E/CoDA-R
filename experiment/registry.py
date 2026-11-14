"""全部实验的注册表与分组；"all" 按这里的顺序串行运行，后面的实验会读取前面实验的输出。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from experiment.common.runner import RunOptions
from experiment.experiments import ablations, checks, main, mechanisms, scale, tuning
from experiment.theory import experiments as theory


@dataclass(frozen = True)
class Experiment:
    id: str
    title: str
    group: str
    run: Callable[[RunOptions], Path]
    # 正式运行的格子数；多阶段实验给出总数，检查类实验为 None
    planned: Callable[[], int] | None = None
    # 依赖：运行前必须已完成的实验（读取它们的输出）
    needs: tuple[str, ...] = ()


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("preflight", "环境、评估器与四种训练入口的检查", "checks", checks.run_preflight),
    Experiment("feasibility", "各游戏每 rollout 耗时、评估耗时与单格外推", "checks", checks.run_feasibility),
    *(Experiment(exp_id, title, "theory", partial(theory.run, exp_id), partial(theory.planned_cells, exp_id, False))
      for exp_id, title in theory.TITLES.items()),
    Experiment("references", "CFR+ 参考线", "references", checks.run_references, lambda: len(checks.CFR_ITERATIONS)),
    Experiment("tune-a", "调参 A：自博弈、R-NaD-PPO、MMD-PPO、PSRO、NFSP", "tune", partial(tuning.run_tuning, "a"),
               partial(tuning.planned_cells, "a")),
    Experiment("tune-b", "调参 B：均匀 FSP、PFSP、CoDA-R（沿用 tune-a 的 eta）", "tune", partial(tuning.run_tuning, "b"),
               partial(tuning.planned_cells, "b"), needs = ("tune-a",)),
    Experiment("ablations", "CoDA-R 消融 A1-A9", "ablations", ablations.run_ablations, ablations.planned_cells,
               needs = ("tune-b",)),
    Experiment("vuln-repair", "M1 漏洞注入、压制与重开（sleeper）", "mechanisms", mechanisms.run_vuln_repair,
               mechanisms.planned_vuln_cells, needs = ("tune-a", "tune-b")),
    Experiment("open-pool", "M2 开放池与无用成员", "mechanisms", mechanisms.run_open_pool, mechanisms.planned_open_cells,
               needs = ("tune-a", "tune-b")),
    Experiment("hidden-hole", "M3 自博弈到达概率受控的隐藏漏洞", "mechanisms", mechanisms.run_hidden_hole,
               mechanisms.planned_hole_cells, needs = ("tune-a", "tune-b")),
    Experiment("forgetting", "M4 从近似纳什出发的遗忘", "mechanisms", mechanisms.run_forgetting,
               mechanisms.planned_forgetting_cells, needs = ("tune-a", "tune-b")),
    Experiment("main", "同预算主对比", "main", main.run_main, main.planned_cells, needs = ("tune-a", "tune-b")),
    Experiment("scale-train", "3x3 abrupt Dark Hex 与 abrupt PTTT 训练", "scale", scale.run_scale_train,
               scale.planned_train_cells, needs = ("tune-a", "tune-b")),
    Experiment("scale-eval", "规模游戏的全树精确 exploitability（exp-a-spiel）", "scale", scale.run_scale_eval,
               scale.planned_eval_cells, needs = ("scale-train",)),
    Experiment("scale-crossplay", "规模游戏最终策略循环赛", "scale", scale.run_crossplay, needs = ("scale-train",)),
    Experiment("scale-history", "M4 规模臂：对自己历史版本的采样对局", "scale", scale.run_scale_history,
               scale.planned_history_cells, needs = ("scale-train",)),
)
BY_ID = {e.id: e for e in EXPERIMENTS}
GROUPS: dict[str, list[str]] = {}
for _exp in EXPERIMENTS:
    GROUPS.setdefault(_exp.group, []).append(_exp.id)
GROUPS["all"] = [e.id for e in EXPERIMENTS]


# 与实验 ID 同名的分组只能包含该实验本身，否则命令行无法区分
assert all(GROUPS[name] == [name] for name in set(GROUPS) & set(BY_ID))


def expand(name: str) -> list[str]:
    if name in GROUPS:
        return list(GROUPS[name])
    if name in BY_ID:
        return [name]
    raise ValueError(f"unknown experiment or group {name!r}; see --list")
