import torch
from open_spiel.python.algorithms import cfr

from experiment.neural.envs import load_game, obs_and_action_size
from experiment.neural.evals import tabular_actor_from_policy
from experiment.neural.holes import HoleProbe, damaged, distill, infoset_profiles, select_holes
from experiment.neural.mlp import SmallMLP
from experiment.neural.policies import frozen_actor


def _kuhn_table() -> tuple[object, dict[str, dict[int, float]]]:
    game = load_game("kuhn_poker")
    solver = cfr.CFRPlusSolver(game)
    for _ in range(200):
        solver.evaluate_and_update_policy()
    return game, tabular_actor_from_policy(game, solver.average_policy()).table


def test_first_decisions_have_full_opponent_reach():
    game, table = _kuhn_table()
    profiles = infoset_profiles(game, table)
    first = [p for p in profiles.values() if p.player == 0 and p.own == 1.0]
    assert first and all(abs(p.opponent_reach - 1.0) < 1e-12 for p in first)


def test_holes_are_costly_and_survive_distillation():
    game, table = _kuhn_table()
    holes = select_holes(infoset_profiles(game, table), table, ((0.0, 1.01),), 2, own_min = 0.1, regret_min = 0.05)[0]
    assert holes and all(h.regret >= 0.05 and table[h.infoset][h.action] <= 0.05 for h in holes)
    torch.manual_seed(0)
    net = SmallMLP(*obs_and_action_size(game), 32)
    assert distill(net, game, damaged(table, holes), steps = 400, lr = 1e-2)["max_tv"] < 0.1
    assert HoleProbe(game, [h.as_dict() for h in holes])(0, frozen_actor(net, "cpu"))["hole_mass"] > 0.9
