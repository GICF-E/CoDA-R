from open_spiel.python.algorithms import cfr

from experiment.neural.envs import load_game
from experiment.neural.evals import exact_loss, tabular_actor_from_policy, to_tabular
from experiment.neural.policies import TabularActor, UniformActor
from experiment.neural.retention import RetentionProbe, reach_band


def _kuhn_table():
    game = load_game("kuhn_poker")
    solver = cfr.CFRPlusSolver(game)
    for _ in range(200):
        solver.evaluate_and_update_policy()
    return game, tabular_actor_from_policy(game, solver.average_policy()).table


def test_reach_band_is_half_open():
    bands = ((0.1, 1.01), (0.0, 0.1))
    assert reach_band(0.1, bands) == 0
    assert reach_band(0.05, bands) == 1
    assert reach_band(1.01, bands) is None


def test_retention_probe_records_drift_and_grows_the_archive():
    game, table = _kuhn_table()
    bands = ((0.0, 1.01),)
    base = TabularActor(table)
    # 底座用 CFR+ 表，探针用均匀策略：应测到正的 drift；档案在考试之后才增长
    probe = RetentionProbe(game, table, bands, own_min = 0.0, every = 1, archive_every = 2, base = base)
    assert probe.band_sizes()[0] > 0
    first = probe(0, UniformActor())
    assert "probe_exploitability" in first and len(first["archive_losses"]) == 1
    assert first["drift_tv"][0] > 0.0
    second = probe(1, UniformActor())
    assert len(second["archive_losses"]) == 1
    third = probe(2, UniformActor())
    assert len(third["archive_losses"]) == 2
    assert third["archive_losses"][0] == exact_loss(game, to_tabular(game, UniformActor()), to_tabular(game, base))
