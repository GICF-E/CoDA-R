"""神经实验的统一协议：游戏、预算与 seed 划分。"""
from __future__ import annotations

MAIN_GAMES = ("leduc_poker", "goofspiel4", "liars_dice")
ABLATION_GAMES = ("leduc_poker", "goofspiel4")
SCALE_GAMES = ("abrupt_dark_hex", "abrupt_phantom_ttt")
# 规模游戏不调参，收益同为 +-1 的 Goofspiel(4) 选出的配置原样沿用
SCALE_SOURCE_GAME = "goofspiel4"

MAIN_BUDGET = 4_000_000
TUNE_BUDGET = MAIN_BUDGET // 4
ABLATION_BUDGET = 1_000_000
OPEN_POOL_BUDGET = 1_000_000

TUNE_SEEDS = (900, 901, 902)
MAIN_SEEDS = tuple(range(100, 110))
SMALL_SEEDS = tuple(range(100, 106))
SCALE_SEEDS = tuple(range(100, 105))
SMOKE_SEEDS = (100,)
