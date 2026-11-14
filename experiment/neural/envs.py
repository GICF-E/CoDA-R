"""OpenSpiel 环境加载、信息状态、合法动作与终局得分口径。"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyspiel

GAME_SPECS: dict[str, str] = {
    "kuhn_poker": "kuhn_poker",
    "leduc_poker": "leduc_poker",
    "goofspiel4": "goofspiel(num_cards=4,points_order=descending)",
    "liars_dice": "liars_dice(players=2,numdice=1,dice_sides=6)",
    "abrupt_dark_hex": "dark_hex(num_rows=3,num_cols=3,gameversion=adh)",
    "abrupt_phantom_ttt": "phantom_ttt(gameversion=abrupt)",
}

# OpenSpiel 的全树表格展开不可行的环境：训练中不评估也不算精确败率，只存 checkpoint，训练后由 exp-a-spiel 精确评估
FULL_TREE_INTRACTABLE = frozenset({"abrupt_dark_hex", "abrupt_phantom_ttt"})

Scorer = Callable[[list[float], int], float]


def load_game(env_name: str) -> Any:
    game = pyspiel.load_game(GAME_SPECS.get(env_name, env_name))
    # exploitability 需要轮流行动；Goofspiel 这类同时行动博弈先转成等价的轮流形式
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        return pyspiel.convert_to_turn_based(game)
    return game


def _provides_info_tensor(game: Any) -> bool:
    return bool(game.get_type().provides_information_state_tensor)


def obs_and_action_size(game: Any) -> tuple[int, int]:
    if _provides_info_tensor(game):
        obs = int(game.information_state_tensor_size())
    else:
        obs = int(game.observation_tensor_size())
    return obs, int(game.num_distinct_actions())


def information_tensor(state: Any, player: int) -> list[float]:
    if _provides_info_tensor(state.get_game()):
        return list(state.information_state_tensor(player))
    return list(state.observation_tensor(player))


def legal_actions(state: Any, player: int) -> list[int]:
    if state.is_simultaneous_node():
        return [int(a) for a in state.legal_actions(player)]
    return [int(a) for a in state.legal_actions()]


def legal_mask(state: Any, player: int, n_actions: int) -> list[bool]:
    mask = [False] * n_actions
    for a in legal_actions(state, player):
        mask[a] = True
    return mask


def score_span(game: Any) -> float:
    return float(game.max_utility() - game.min_utility())


def make_scorer(game: Any) -> Scorer:
    """终局得分 = 收益仿射归一到 [0, 1]；均衡集合与原博弈相同，平局线为 1/2。"""
    lo = float(game.min_utility())
    span = score_span(game)

    def score(returns: list[float], player: int) -> float:
        return min(1.0, max(0.0, (float(returns[player]) - lo) / span))

    return score
