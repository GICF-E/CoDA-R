"""论文图的统一样式：每个方法固定的颜色、线型与名称，游戏名称，以及 matplotlib 参数。"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from experiment.common.plotting import get_pyplot


@dataclass(frozen = True)
class Style:
    label: str
    color: str
    linestyle: str = "-"
    marker: str = "o"
    linewidth: float = 1.2

    def line_kwargs(self) -> dict[str, Any]:
        return dict(color = self.color, linestyle = self.linestyle, linewidth = self.linewidth, label = self.label)


# 方法键与 experiment.experiments.methods 的方法名一致；coda_r_q 是 CoDA-R 同一次运行的时间平均混合 Q；名称与论文表格的行名一致
METHOD_STYLES: dict[str, Style] = {
    "coda_r": Style("CoDA-R", "#d62728", "-", "o", 2.0),
    "coda_r_no_oracle": Style("CoDA-R without oracle", "#ff9896", "--", "o"),
    "coda_r_exact_br": Style("CoDA-R, exact-BR oracle", "#8b0000", "--", "o"),
    "coda_r_q": Style(r"CoDA-R time average $\bar{q}$", "#ff7f0e", "-.", "D", 1.6),
    "pfsp": Style("PFSP", "#1f77b4", "-", "s"),
    "pfsp_exact_br": Style("PFSP, exact-BR oracle", "#1f77b4", "--", "s"),
    "ufsp": Style("Uniform FSP", "#2ca02c", "-", "^"),
    "rnad": Style("R-NaD-PPO", "#4d4d4d", "-", "v"),
    "mmd": Style("MMD-PPO", "#9467bd", "-", "P"),
    "selfplay": Style("PPO self-play", "#8c564b", ":", "X"),
    "psro": Style("PSRO (PPO BR)", "#17becf", "-.", "*"),
    "psro_exact_br": Style("PSRO, exact BR", "#17becf", ":", "*"),
    "nfsp": Style("NFSP", "#bcbd22", "-.", "h"),
    "hedge": Style("Hedge", "#e377c2", "-", "<"),
    "fixed": Style("Fixed mixing", "#7f7f7f", "--", ">"),
    "moving_anchor": Style("Moving dual anchor", "#ff7f0e", "--", "D"),
    "none": Style("R-NaD-PPO (no pool)", "#4d4d4d", "-", "v"),
}
SINGLE_METHODS = ("coda_r", "coda_r_no_oracle", "pfsp", "ufsp", "rnad", "mmd", "selfplay")
MIXTURE_METHODS = ("coda_r_q", "psro", "nfsp")
UPPER_BOUND_METHODS = ("coda_r_exact_br", "pfsp_exact_br", "psro_exact_br")

ENV_TITLES = {
    "leduc_poker": "Leduc poker",
    "goofspiel4": "Goofspiel(4)",
    "liars_dice": "Liar's Dice",
    "abrupt_dark_hex": "Abrupt Dark Hex 3x3",
    "abrupt_phantom_ttt": "Abrupt Phantom TTT",
}
MAIN_ENVS = ("goofspiel4", "leduc_poker", "liars_dice")
ABLATION_ENVS = ("goofspiel4", "leduc_poker")
SCALE_ENVS = ("abrupt_dark_hex", "abrupt_phantom_ttt")

# 20 色定性色板，给没有固定样式的变体（消融、诊断）按名称顺序取色
_PALETTE = ("#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd",
            "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
            "#17becf", "#9edae5")
# 对数纵轴的下限：精确 BR 的 PSRO 会到 1e-40，不截断会把其它曲线压成一条线
LOG_FLOOR = 1e-3


def method_style(method: str) -> Style:
    if method not in METHOD_STYLES:
        raise KeyError(f"no style for method {method!r}")
    return METHOD_STYLES[method]


def variant_styles(labels: Iterable[str], highlight: str | None = None) -> dict[str, Style]:
    """按名称排序取色；highlight（通常是基准）固定为黑色粗线。"""
    out: dict[str, Style] = {}
    others = sorted(label for label in set(labels) if label != highlight)
    for i, label in enumerate(others):
        out[label] = Style(label, _PALETTE[i % len(_PALETTE)], ("-", "--", "-.", ":")[(i // len(_PALETTE)) % 4])
    if highlight is not None:
        out[highlight] = Style(highlight, "#000000", "-", "o", 2.0)
    return out


def env_title(env: str) -> str:
    return ENV_TITLES.get(env, env)


def apply_rc() -> Any | None:
    """论文用的 matplotlib 参数；返回 pyplot，不可用时返回 None。"""
    plt = get_pyplot()
    if plt is None:
        return None
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200,
    })
    return plt
