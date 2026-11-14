import numpy as np
import pytest

from experiment.common.stats import history_forgetting, iqm, paired, tail_metrics, wilson
from experiment.neural.config import LATE_FIELDS, RunConfig, config_from_dict
from experiment.neural.schedule import EvalSchedule, entropy_coef, learning_rate, oracle_cost, oracle_period


def _cfg(**kw):
    return RunConfig(env = "leduc_poker", label = "t", seed = 0, budget = 4_000_000, **kw)


def test_entropy_anneals_until_tail_then_holds():
    cfg = _cfg(ent_final = 0.003)
    assert entropy_coef(cfg, 0) == pytest.approx(0.03)
    assert entropy_coef(cfg, 1_600_000) == pytest.approx(0.0165)
    assert entropy_coef(cfg, 3_200_000) == pytest.approx(0.003)
    assert entropy_coef(cfg, 4_000_000) == pytest.approx(0.003)


def test_learning_rate_is_constant_by_default_and_anneals_over_the_budget():
    assert learning_rate(_cfg(), 3_000_000) == pytest.approx(3e-4)
    cfg = _cfg(lr_final_frac = 0.0)
    assert learning_rate(cfg, 0) == pytest.approx(3e-4)
    assert learning_rate(cfg, 2_000_000) == pytest.approx(1.5e-4)
    assert learning_rate(cfg, 4_000_000) == pytest.approx(0.0)


def test_late_fields_keep_existing_config_hashes():
    payload = _cfg().as_dict()
    assert not set(LATE_FIELDS) & set(payload)
    assert config_from_dict(payload) == _cfg()
    changed = _cfg(lr_final_frac = 0.0, oracle_ent_final = 0.01).as_dict()
    assert changed["lr_final_frac"] == 0.0 and changed["oracle_ent_final"] == 0.01


def test_oracle_period_matches_target_share_and_is_shared_by_exact_br():
    cfg = _cfg(rule = "coda_r", oracle = "ppo")
    # Leduc：20 x 3 个 rollout x 512 局 + 256 局入池评估
    assert oracle_cost(cfg) == 20 * 3 * 512 + 256
    period = oracle_period(cfg)
    share = oracle_cost(cfg) / (period * 32 * 512)
    assert abs(share - 0.12) < 0.01
    assert oracle_period(_cfg(rule = "coda_r", oracle = "exact_br")) == period
    assert oracle_period(_cfg(rule = "coda_r", oracle = "none")) == 0


def test_eval_schedule_has_log_points_and_dense_tail():
    schedule = EvalSchedule(_cfg())
    assert schedule.points[0] >= 512 and schedule.points[-1] == pytest.approx(3_200_000)
    assert schedule.due(3_999_000, 4_000_000, 5)
    assert schedule.due(3_300_000, 3_300_512, 7) and not schedule.due(3_300_000, 3_300_512, 6)


def test_tail_metrics_use_forward_hold():
    evals = [(0, 1.0), (500, 0.5), (800, 0.4), (900, 0.2), (1000, 0.1)]
    m = tail_metrics(evals, budget = 1000, tail_start = 800)
    assert m["at_budget"] == pytest.approx(0.1)
    assert m["tail_mean"] == pytest.approx((0.4 * 100 + 0.2 * 100) / 200)
    assert m["tail_worst"] == pytest.approx(0.4)
    assert np.isnan(tail_metrics(evals[:-1], 1000, 800)["at_budget"])


def test_small_statistics():
    assert iqm([1, 2, 3, 4]) == pytest.approx(2.5)
    point, lo, hi = wilson(4, 40)
    assert lo < point < hi and point == pytest.approx(0.1)
    result = paired([1.0, 2.0, 3.0], [2.0, 2.5, 2.0])
    assert result["wins"] == 2 and result["losses"] == 1


def test_history_forgetting_counts_reopened_versions():
    # 版本 0：先被打败（0.4），后来反超（0.6）；版本 1：一直打平，不算遗忘
    stats = history_forgetting([[0.4], [0.45, 0.5], [0.6, 0.5]], margin = 0.05)
    assert stats["forgotten"] == pytest.approx(0.5)
    assert stats["final_worst_past"] == pytest.approx(0.1)
    assert stats["worst_past"] == pytest.approx((0.4 + 0.5 + 0.6) / 3 - 0.5)
    empty = history_forgetting([], margin = 0.05)
    assert all(np.isnan(empty[k]) for k in ("worst_past", "final_worst_past", "forgotten"))


def test_config_rejects_oracle_without_pool():
    with pytest.raises(ValueError):
        _cfg(oracle = "ppo")
