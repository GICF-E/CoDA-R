"""缓存与复刻的评估函数必须与 OpenSpiel 原实现逐位相同（用 ==，不设容差）。"""
import numpy as np
import pytest
import torch
from open_spiel.python import policy as sp_policy
from open_spiel.python.algorithms import best_response as br_mod
from open_spiel.python.algorithms import exploitability as expl_mod
from open_spiel.python.algorithms import policy_aggregator

from experiment.neural.envs import load_game, obs_and_action_size
from experiment.neural.evals import aggregate, exact_br_actor, exploitability_of_table, tabular_from_callable, to_tabular
from experiment.neural.mlp import SmallMLP
from experiment.neural.policies import MixtureActor, frozen_actor

GAMES = ("kuhn_poker", "goofspiel4", "leduc_poker")


def _actors(game, n: int) -> list:
    obs_dim, n_actions = obs_and_action_size(game)
    actors = []
    for seed in range(n):
        torch.manual_seed(seed)
        net = SmallMLP(obs_dim, n_actions, 32)
        # 放大权重让分布更尖锐，接近训练后的网络
        with torch.no_grad():
            for p in net.parameters():
                p.mul_(3.0)
        actors.append(frozen_actor(net, "cpu"))
    return actors


def _same_dicts(a: list[dict], b: list[dict]) -> bool:
    return all(list(x.items()) == list(y.items()) for x, y in zip(a, b)) and len(a) == len(b)


@pytest.mark.parametrize("env", GAMES)
def test_tabular_and_exploitability_match_openspiel(env: str):
    game = load_game(env)
    n_actions = int(game.num_distinct_actions())
    for actor in _actors(game, 2):
        def call(state, actor = actor):
            return actor.probs(state, int(state.current_player()), n_actions)
        ours = tabular_from_callable(game, call)
        reference = sp_policy.tabular_policy_from_callable(game, call)
        assert np.array_equal(ours.action_probability_array, reference.action_probability_array)
        assert ours.state_lookup == reference.state_lookup
        assert exploitability_of_table(game, ours) == float(expl_mod.exploitability(game, reference))


@pytest.mark.parametrize("env", GAMES)
def test_aggregate_matches_policy_aggregator(env: str):
    game = load_game(env)
    tables = [to_tabular(game, a) for a in _actors(game, 3)]
    weights = np.array([0.2, 0.5, 0.3])
    players = list(range(game.num_players()))
    w = (weights / weights.sum()).tolist()
    ours = aggregate(game, tables, weights)
    reference = policy_aggregator.PolicyAggregator(game).aggregate(players, [tables for _ in players], [w for _ in players])
    assert _same_dicts(ours.policy, reference.policy)
    # 混合输出里的时间平均是对聚合结果再聚合
    nested_ours = aggregate(game, [ours, tables[0]], np.array([1.5, 1.0]))
    nested_w = (np.array([1.5, 1.0]) / 2.5).tolist()
    nested_reference = policy_aggregator.PolicyAggregator(game).aggregate(
        players, [[reference, tables[0]] for _ in players], [nested_w for _ in players])
    assert _same_dicts(nested_ours.policy, nested_reference.policy)
    assert exploitability_of_table(game, nested_ours) == float(expl_mod.exploitability(game, nested_reference))


def _reference_exact_br(game, mixture: MixtureActor) -> dict[str, dict[int, float]]:
    """改动之前的 exact_br_actor：全部走 OpenSpiel 原函数。"""
    n_actions = int(game.num_distinct_actions())
    players = list(range(game.num_players()))
    tables = [sp_policy.tabular_policy_from_callable(game, lambda s, a = a: a.probs(s, int(s.current_player()), n_actions))
              for a in mixture.actors]
    w = (np.asarray(mixture.weights) / float(np.sum(mixture.weights))).tolist()
    mixed = policy_aggregator.PolicyAggregator(game).aggregate(players, [tables for _ in players], [w for _ in players])
    table = sp_policy.tabular_policy_from_callable(game, mixed.action_probabilities)
    responses = {p: br_mod.BestResponsePolicy(game, p, table) for p in players}

    def call(state):
        player = int(state.current_player())
        return {int(a): float(p) for a, p in responses[player].action_probabilities(state, player_id = player).items()}

    br_table = sp_policy.tabular_policy_from_callable(game, call)
    out = {}
    for idx, state in enumerate(br_table.states):
        row = br_table.action_probability_array[idx]
        out[state.information_state_string(int(state.current_player()))] = {i: float(p) for i, p in enumerate(row) if p > 0.0}
    return out


@pytest.mark.parametrize("env", ("kuhn_poker", "goofspiel4"))
def test_exact_br_of_mixture_is_unchanged(env: str):
    game = load_game(env)
    mixture = MixtureActor(_actors(game, 3), np.array([0.1, 0.6, 0.3]))
    ours = exact_br_actor(game, mixture)
    assert list(ours.table.items()) == list(_reference_exact_br(game, mixture).items())
