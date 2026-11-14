"""全部论文图的注册表：名称 -> 函数 (Outputs, 输出目录) -> 写出的文件名（不含扩展名）列表；缺数据的图返回空列表。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from analysis.data import Outputs
from analysis.figures import mechanisms, neural, scale, theory, tuning

FigureFn = Callable[[Outputs, Path], list[str]]

FIGURES: dict[str, FigureFn] = {
    "main_curves": neural.main_curves,
    "main_curves_upper": neural.main_curves_upper,
    "main_tail": neural.main_tail,
    "main_paired": neural.main_paired,
    "main_degradation": neural.main_degradation,
    "main_entropy": neural.main_entropy,
    "scale_curves": scale.scale_curves,
    "ablations_forest": mechanisms.ablations_forest,
    "diag_tail_curves": mechanisms.diag_tail_curves,
    "diag_tail_summary": mechanisms.diag_tail_summary,
    "vuln_repair": mechanisms.vuln_repair,
    "vuln_sleeper": mechanisms.vuln_sleeper,
    "open_pool": mechanisms.open_pool,
    "hidden_hole": mechanisms.hidden_hole,
    "forgetting": mechanisms.forgetting,
    "forgetting_drift": mechanisms.forgetting_drift,
    "scale_history": scale.scale_history,
    "theory_necessity": theory.theory_necessity,
    "theory_anchor_basin": theory.theory_anchor_basin,
    "theory_inexact": theory.theory_inexact,
    "theory_pool_rules": theory.theory_pool_rules,
    "theory_oracle_rate": theory.theory_oracle_rate,
    "theory_adversarial_pool": theory.theory_adversarial_pool,
    "tuning_a": tuning.tuning_a,
    "tuning_b": tuning.tuning_b,
}
