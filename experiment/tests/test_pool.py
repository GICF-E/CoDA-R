import numpy as np
import pytest

from experiment.core.estimator import EXACT_N_EFF, LossEstimator
from experiment.core.pool import MIRROR_ID, OpponentPool, SamplingConfig
from experiment.core.pool_rules import CoDARRule, PFSPRule


def _pool(sampling: SamplingConfig | None = None, capacity: int = 8, freeze: bool = False) -> OpponentPool:
    return OpponentPool(CoDARRule(6.0), LossEstimator(), sampling or SamplingConfig(), capacity = capacity, freeze_field = freeze)


def test_importance_ratios_recover_the_training_field():
    pool = _pool()
    pool.add("a", None, "oracle", initial_loss = 0.6)
    pool.add("b", None, "oracle", initial_loss = 0.4)
    plan = pool.begin_rollout(512)
    expected = np.array([plan.probs[i] * plan.importance_ratio(i) for i in range(plan.probs.size)])
    np.testing.assert_allclose(expected, plan.field)
    assert plan.field[-1] == 1.0 and np.isclose(plan.probs.sum(), 1.0)
    # 自博弈臂的下限保证 iw <= 1 / mirror_floor
    assert plan.importance_ratio(plan.probs.size - 1) <= 1.0 / 0.1 + 1e-12


def test_dual_sampling_gives_the_pool_more_games_than_field_sampling():
    shares = {}
    for base in ("dual", "field"):
        pool = _pool(SamplingConfig(base = base, detection = "none", eps = 0.0))
        pool.add("a", None, "oracle", initial_loss = 0.65)
        plan = pool.begin_rollout(512)
        shares[base] = 1.0 - plan.probs[-1]
    a = 1.0 - shares["dual"]
    assert shares["dual"] > shares["field"]
    assert np.isclose(shares["field"], (1 - a) / (2 - a))


def test_probe_floors_and_zero_field_members():
    pool = _pool()
    for i in range(4):
        pool.add(f"m{i}", None, "snapshot", initial_loss = 0.2)
    plan = pool.begin_rollout(512)
    assert np.all(plan.field[:-1] == 0.0)
    member_probs = plan.probs[:-1]
    assert np.all(member_probs >= 8 / 512 - 1e-12) and member_probs.sum() <= 0.25 + 0.05 + 1e-9


def test_sampling_records_and_estimates_update():
    pool = _pool(SamplingConfig(detection = "none"))
    pool.add("a", None, "oracle", initial_loss = 0.5)
    rng = np.random.default_rng(0)
    pool.begin_rollout(256)
    for _ in range(256):
        _member, arm, _iw = pool.sample(rng)
        pool.record(arm, 1.0)
    stats = pool.end_rollout()
    assert pool.estimator.losses(["a"])[0] > 0.5
    assert 0.0 < stats["ess_fraction"] <= 1.0


def test_eviction_prefers_zero_field_members_and_respects_protection():
    pool = _pool(capacity = 4)
    pool.add("strong", None, "oracle", initial_loss = 0.9)
    pool.add("weak", None, "snapshot", initial_loss = 0.1)
    pool.add("new", None, "snapshot", initial_loss = 0.1)
    pool.add("newer", None, "snapshot", initial_loss = 0.1)
    # 最近入池的两个受保护；未受保护的成员里先淘汰训练场系数为 0 的
    assert pool.add("newest", None, "oracle", initial_loss = 0.5) == ["weak"]
    with pytest.raises(ValueError):
        pool.add(MIRROR_ID, None, "oracle")


def test_freeze_field_keeps_outer_step_coefficients():
    pool = _pool(freeze = True)
    pool.add("a", None, "oracle", initial_loss = 0.7)
    first = pool.begin_rollout(64).field.copy()
    pool.end_rollout(exact_losses = {"a": 0.5})
    assert np.allclose(pool.begin_rollout(64).field, first)
    pool.end_rollout()
    pool.end_outer_step()
    assert pool.begin_rollout(64).field[0] == 0.0


def test_estimator_resets_on_a_jump_and_exact_overrides():
    est = LossEstimator(n0 = 64, reset_k = 2.5)
    est.add("a", 0.3)
    assert est.update("a", 256, 0.9)
    assert est.losses(["a"])[0] > 0.7
    est.set_exact("a", 0.42)
    assert est.losses(["a"])[0] == 0.42 and est.n_eff(["a"])[0] == EXACT_N_EFF


def test_pfsp_pool_has_no_zero_field_members():
    pool = OpponentPool(PFSPRule(2.0, 0.5), LossEstimator(), SamplingConfig(), capacity = 4)
    pool.add("a", None, "oracle", initial_loss = 0.3)
    plan = pool.begin_rollout(64)
    assert plan.field[0] > 0.0 and np.isclose(plan.field.sum(), 1.0)
