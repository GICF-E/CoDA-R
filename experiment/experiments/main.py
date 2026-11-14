"""主实验：全部方法在三个可精确评估的游戏上同预算比较，10 个评估 seed，配置取自 tune-a / tune-b。"""
from __future__ import annotations

from pathlib import Path

from experiment.common.runner import CellSpec, RunOptions, prepare_output, run_cells
from experiment.experiments.cells import run_train_cell, train_spec
from experiment.experiments.methods import build_config, deployment, source_method, title
from experiment.experiments.protocol import MAIN_BUDGET, MAIN_GAMES, MAIN_SEEDS, SMOKE_SEEDS
from experiment.experiments.report import Pair, load_results, write_grid_report
from experiment.experiments.tuning import load_selected

MAIN_METHODS = (
    "selfplay", "rnad", "mmd", "ufsp", "pfsp", "coda_r", "coda_r_no_oracle",
    "psro", "nfsp",
    "coda_r_exact_br", "pfsp_exact_br", "psro_exact_br",
)
PAIRS = [
    # 对偶规则：同一个池、同一个 oracle、同一个原侧，只差池权重规则
    Pair(title("coda_r"), title("pfsp")),
    Pair(title("coda_r"), title("ufsp")),
    # 对手池与 oracle：相对正则自博弈
    Pair(title("coda_r"), title("rnad")),
    Pair(title("coda_r"), title("mmd")),
    Pair(title("coda_r"), title("coda_r_no_oracle")),
    # oracle 强度上限下的对偶规则
    Pair(title("coda_r_exact_br"), title("pfsp_exact_br")),
    # 混合对混合：CoDA-R 的时间平均 Q 与当前混合 q 对 PSRO
    Pair(title("coda_r"), title("psro"), "mix_average_expl", "exploitability"),
    Pair(title("coda_r"), title("psro"), "mix_current_expl", "exploitability"),
]
NOTES = [
    f"预算 B = {MAIN_BUDGET} 局算法性对局；熵在前 0.8B 内线性退到熵终值，末段 [0.8B, B] 保持不变。",
    "主指标 tail_mean，次要指标 tail_worst 与 at_budget；全程曲线与 AUC 只作描述。",
    "所有方法的配置来自 tune-a / tune-b（调参 seed 900-902，不进入本表）。",
    "精确 BR 不消耗环境对局，带 exact-BR 的三行是 oracle 强度上限对照，不是同成本比较。",
    "单网络与混合部署分表报告；CoDA-R 的 q / Q 来自 CoDA-R 同一次运行。",
]


def specs(out_root: Path, smoke: bool) -> list[CellSpec]:
    selected = load_selected(out_root, smoke)
    seeds = SMOKE_SEEDS if smoke else MAIN_SEEDS
    out = []
    for env in MAIN_GAMES:
        for method in MAIN_METHODS:
            params = selected[env][source_method(method)]
            for seed in seeds:
                cfg = build_config(method, env, seed, MAIN_BUDGET, params)
                out.append(train_spec(cfg, smoke, method = method, deployment = deployment(method)))
    return out


def run_main(options: RunOptions) -> Path:
    out = prepare_output(options)
    cells = specs(out.parent, options.smoke)
    records = run_cells(cells, run_train_cell, options)
    write_grid_report(out, "main：同预算主对比", NOTES, load_results(out, records), len(cells), PAIRS)
    return out


def planned_cells() -> int:
    return len(MAIN_GAMES) * len(MAIN_METHODS) * len(MAIN_SEEDS)
