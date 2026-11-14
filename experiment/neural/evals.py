"""全树评估：exploitability、成员精确败率、精确 BR 与元博弈 LP。整局混合按到达概率聚合。

OpenSpiel 的 TabularPolicy、exploitability 与 PolicyAggregator 每次调用都会重新枚举整棵博弈树。
这里按游戏缓存一次枚举结果，并按完全相同的遍历顺序与浮点运算复刻这三处，结果与 OpenSpiel 逐位相同。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyspiel
from open_spiel.python import policy as sp_policy
from open_spiel.python.algorithms import best_response as br_mod
from open_spiel.python.algorithms import expected_game_score as egs_mod
from open_spiel.python.algorithms import get_all_states as states_mod
from open_spiel.python.algorithms import policy_aggregator
from scipy.optimize import linprog

from experiment.neural.envs import legal_actions
from experiment.neural.policies import MixtureActor, TabularActor

# 每个 rollout 都能负担精确败率的环境
EXACT_LOSS_ENVS = frozenset({"kuhn_poker", "leduc_poker", "goofspiel4"})
# PolicyAggregator 归一化时加在每个动作上的质量，避免零到达概率的信息集除零
AGGREGATE_EPSILON = 1e-40


# -------- 博弈树缓存 --------
@dataclass(frozen = True)
class _GameTree:
    # 全部决策状态（不含终局与机会节点），键为 history_str
    all_states: dict[str, Any]
    info_strings: dict[str, str]
    # 均匀策略模板；展开时复制它再填概率，共享 states 与合法动作掩码
    template: Any


_TREES: dict[str, _GameTree] = {}


def _tree(game: Any) -> _GameTree:
    key = str(game)
    tree = _TREES.get(key)
    if tree is None:
        all_states = states_mod.get_all_states(game, depth_limit = -1, include_terminals = False,
                                               include_chance_states = False, include_mean_field_states = False)
        info_strings = {s: all_states[s].information_state_string() for s in all_states}
        tree = _GameTree(all_states, info_strings, sp_policy.TabularPolicy(game, states = all_states))
        _TREES[key] = tree
    return tree


def infoset_states(game: Any) -> dict[str, tuple[Any, int]]:
    """每个信息集取一个代表状态：{信息集字符串: (状态, 行动的玩家)}；只依赖信息集的策略在代表状态上的输出就是它在整个信息集上的输出。"""
    tree = _tree(game)
    out: dict[str, tuple[Any, int]] = {}
    for history, state in tree.all_states.items():
        out.setdefault(tree.info_strings[history], (state, int(state.current_player())))
    return out


def tabular_from_callable(game: Any, call: Callable[[Any], dict[int, float]]) -> Any:
    """同 sp_policy.tabular_policy_from_callable(game, call)，只是不重新枚举博弈树。"""
    table = _tree(game).template.__copy__()
    n_actions = int(game.num_distinct_actions())
    for index, state in enumerate(table.states):
        probs = dict(call(state))
        table.action_probability_array[index, :] = [probs.get(action, 0.0) for action in range(n_actions)]
    return table


def to_tabular(game: Any, actor: Any) -> Any:
    """把对手展开成 OpenSpiel 策略对象；整局混合用到达概率聚合，不能逐信息集平均。"""
    if isinstance(actor, MixtureActor):
        return aggregate(game, [to_tabular(game, a) for a in actor.actors], actor.weights)
    n_actions = int(game.num_distinct_actions())

    def call(state: Any, player_id: Any = None) -> dict[int, float]:
        player = int(state.current_player()) if player_id is None else int(player_id)
        if hasattr(actor, "probs"):
            return actor.probs(state, player, n_actions)
        return actor.action_probabilities(state, player_id = player)

    return tabular_from_callable(game, call)


# -------- 按到达概率聚合 --------
def _infoset_probs(table: Any, state: Any, player: int, key: str) -> Any:
    """table.action_probabilities(state, player)；普通 TabularPolicy 直接按键取行，省去逐策略重算信息集字符串。"""
    if type(table) is sp_policy.TabularPolicy:
        return table.action_probability_array[table.state_lookup[key]]
    return table.action_probabilities(state, player_id = player)


def _aggregate_player(game: Any, pid: int, tables: list[Any], weights: list[float]) -> dict[str, dict[int, Any]]:
    """
    PolicyAggregator._sub_aggregate 的复刻：同样的遍历顺序与乘加顺序。
    原实现给每条边深拷贝 [w, w]，两个座位的到达向量是同一个列表，只有 pid 那一份被读写，所以这里只保留一份。
    """
    policy: dict[str, dict[int, Any]] = {}
    direct = [type(table) is sp_policy.TabularPolicy for table in tables]

    def recurse(state: Any, reach: list[Any]) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            outcomes, _probs = zip(*state.chance_outcomes())
            for outcome in outcomes:
                recurse(state.child(outcome), reach)
            return
        turn = int(state.current_player())
        moves = state.legal_actions(turn)
        if turn != pid:
            for uid in moves:
                recurse(state.child(uid), reach)
            return
        key = state.information_state_string(turn)
        rows = [_infoset_probs(table, state, turn, key) for table in tables]
        node = policy.setdefault(key, {})
        for uid in moves:
            new_reach = list(reach)
            for i, row in enumerate(rows):
                new_reach[i] *= row[uid] if direct[i] else row.get(uid, 0)
                if uid in node:
                    node[uid] += new_reach[i]
                else:
                    node[uid] = new_reach[i]
            recurse(state.child(uid), new_reach)

    recurse(game.new_initial_state(), list(weights))
    for key in policy:
        actions, probabilities = zip(*policy[key].items())
        shifted = [p + AGGREGATE_EPSILON for p in probabilities]
        denom = sum(shifted)
        for i in range(len(actions)):
            policy[key][actions[i]] = shifted[i] / denom
    return policy


def aggregate(game: Any, tables: list[Any], weights: np.ndarray) -> Any:
    """同 PolicyAggregator(game).aggregate(players, [tables] * n, [w] * n)，返回同类型的 PolicyFunction。"""
    game_type = game.get_type()
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL or not game_type.provides_information_state_string:
        raise ValueError("aggregate expects a turn-based game with information state strings; load it through envs.load_game")
    players = list(range(game.num_players()))
    w = (np.asarray(weights, dtype = np.float64) / float(np.sum(weights))).tolist()
    return policy_aggregator.PolicyFunction(players, [_aggregate_player(game, pid, tables, w) for pid in players], game)


def exploitability_of_table(game: Any, table: Any) -> float:
    """OpenSpiel 原生收益口径（Leduc 为筹码）；纳什处为 0。与 exploitability.exploitability 相同，只是复用缓存的状态枚举。"""
    tree = _tree(game)
    root = game.new_initial_state()
    nash_conv = sum(br_mod.CPPBestResponsePolicy(game, player, table, tree.all_states, tree.info_strings).value(root)
                    for player in range(game.num_players())) - game.utility_sum()
    return float(nash_conv / game.num_players())


def full_exploitability(game: Any, actor: Any) -> float:
    return exploitability_of_table(game, to_tabular(game, actor))


def policy_values(game: Any, policies: list[Any]) -> np.ndarray:
    """各座位按 policies[i] 行动时的期望收益（全树遍历，不消耗环境对局）。"""
    return np.asarray(egs_mod.policy_value(game.new_initial_state(), policies), dtype = np.float64)


def exact_loss(game: Any, latest_table: Any, member_table: Any) -> float:
    """latest 对成员的精确败率：两个座位各算一次再平均，对应训练中随机分配的座位。"""
    lo = float(game.min_utility())
    span = float(game.max_utility()) - lo
    value_0 = float(policy_values(game, [latest_table, member_table])[0])
    value_1 = float(policy_values(game, [member_table, latest_table])[1])
    score = (0.5 * (value_0 + value_1) - lo) / span
    return 1.0 - min(1.0, max(0.0, score))


class TableCache:
    """冻结成员的表格只展开一次；缓存绑定到对象身份，重名的新成员不会拿到旧表。"""

    def __init__(self, game: Any):
        self.game = game
        self._tables: dict[str, tuple[Any, Any]] = {}

    def get(self, member_id: str, actor: Any) -> Any:
        cached = self._tables.get(member_id)
        if cached is None or cached[0] is not actor:
            cached = (actor, to_tabular(self.game, actor))
            self._tables[member_id] = cached
        return cached[1]

    def retain(self, member_ids: set[str]) -> None:
        self._tables = {k: v for k, v in self._tables.items() if k in member_ids}


def exact_member_losses(game: Any, latest: Any, members: list[tuple[str, Any]], cache: TableCache) -> dict[str, float]:
    latest_table = to_tabular(game, latest)
    return {member_id: exact_loss(game, latest_table, cache.get(member_id, actor)) for member_id, actor in members}


class _BestResponseWrap:
    def __init__(self, responses: dict[int, Any]):
        self.responses = responses

    def probs(self, state: Any, player: int, n_actions: int) -> dict[int, float]:
        dist = self.responses[player].action_probabilities(state, player_id = player)
        return {int(a): float(p) for a, p in dist.items()}


def exact_br_actor(game: Any, opponent: Any) -> TabularActor:
    """对手的全树精确 BR（两个座位都要会下），返回只通过对局交互的表格策略。"""
    return exact_br_from_table(game, to_tabular(game, opponent))


def exact_br_from_table(game: Any, table: Any) -> TabularActor:
    """同 exact_br_actor，但对手已经展开成 OpenSpiel 策略对象（例如由缓存的成员表格聚合而成）。"""
    if not hasattr(table, "action_probability_array"):
        # 聚合产物不是 TabularPolicy，再展开一次
        table = tabular_from_callable(game, table.action_probabilities)
    responses = {p: br_mod.BestResponsePolicy(game, p, table) for p in range(game.num_players())}
    br_table = to_tabular(game, _BestResponseWrap(responses))
    out: dict[str, dict[int, float]] = {}
    for idx, state in enumerate(br_table.states):
        player = int(state.current_player())
        row = br_table.action_probability_array[idx]
        out[state.information_state_string(player)] = {i: float(p) for i, p in enumerate(row) if p > 0.0}
    return TabularActor(out)


def tabular_actor_from_policy(game: Any, policy: Any) -> TabularActor:
    """把 OpenSpiel 的策略对象（如 CFR+ 平均策略）包装成可对局的表格对手。"""
    table = tabular_from_callable(game, policy.action_probabilities)
    out: dict[str, dict[int, float]] = {}
    for idx, state in enumerate(table.states):
        player = int(state.current_player())
        legal = legal_actions(state, player)
        row = table.action_probability_array[idx]
        out[state.information_state_string(player)] = {a: float(row[a]) for a in legal}
    return TabularActor(out)


def meta_nash_weights(payoff: np.ndarray) -> np.ndarray:
    """对称零和元博弈 payoff[i, j] = 成员 i 对 j 的得分；返回安全策略 argmin_mu max_i (L mu)_i，L = 1 - payoff。"""
    loss = 1.0 - np.asarray(payoff, dtype = np.float64)
    n = loss.shape[0]
    if n == 1:
        return np.ones(1, dtype = np.float64)
    c = np.zeros(n + 1)
    c[-1] = -1.0
    res = linprog(
        c,
        A_ub = np.hstack([-loss, np.ones((n, 1))]),
        b_ub = np.zeros(n),
        A_eq = np.append(np.ones(n), 0.0).reshape(1, -1),
        b_eq = np.array([1.0]),
        bounds = [(0.0, None)] * n + [(None, None)],
        method = "highs",
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"meta-game LP failed: {res.message}")
    w = np.maximum(np.asarray(res.x[:n], dtype = np.float64), 0.0)
    return w / float(w.sum())
