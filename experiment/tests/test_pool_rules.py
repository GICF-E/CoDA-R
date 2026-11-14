import numpy as np
import pytest

from experiment.core.pool_rules import CoDARRule, FixedRule, HedgeRule, MovingAnchorRule, PFSPRule, UniformRule
from experiment.core.simplex import simplex_proj


def test_simplex_projection_is_on_simplex_and_exact_zero_below_water_level():
    w = simplex_proj(np.array([6.0, 3.06, 4.0]))
    assert np.isclose(w.sum(), 1.0)
    np.testing.assert_allclose(w, [1.0, 0.0, 0.0])


@pytest.mark.parametrize("c, lead", [(1.0, 0.5), (6.0, 0.1), (6.0, 0.4), (48.0, 0.03)])
def test_coda_r_single_member_closed_form(c, lead):
    out = CoDARRule(c).output(["a"], np.array([0.5 + lead]))
    assert np.isclose(out.dual[0], np.clip(c * lead / 2, 0.0, 1.0))
    # 自博弈份额恒为 1，池系数等于对偶权重
    assert out.field[-1] == 1.0 and np.isclose(out.field[0], out.dual[0])


def test_coda_r_tied_or_losing_members_get_exactly_zero():
    out = CoDARRule(6.0).output(["a", "b", "c"], np.array([0.5, 0.3, 0.6]))
    assert out.dual[0] == 0.0 and out.dual[1] == 0.0 and out.dual[2] > 0.0
    assert CoDARRule(6.0).output(["a"], np.array([0.5])).pool_mass == 0.0


def test_moving_anchor_keeps_weight_on_a_tied_member():
    rule = MovingAnchorRule(6.0)
    rule.anchor = {"a": 1.0}
    rule.mirror_anchor = 0.0
    out = rule.output(["a"], np.array([0.5]))
    np.testing.assert_allclose(out.dual, [1.0, 0.0])
    # 固定锚下同一个打平成员没有权重
    assert CoDARRule(6.0).output(["a"], np.array([0.5])).dual[0] == 0.0


def test_pfsp_and_uniform_fields_sum_to_one_with_fixed_self_share():
    for rule in (PFSPRule(2.0, 0.5), UniformRule(0.8)):
        out = rule.output(["a", "b"], np.array([0.9, 0.1]))
        assert np.isclose(out.field.sum(), 1.0)
        np.testing.assert_allclose(out.field, out.dual)
    pfsp = PFSPRule(2.0, 0.5).output(["a", "b"], np.array([0.9, 0.1]))
    assert np.isclose(pfsp.field[0] / pfsp.field[1], 81.0)


def test_hedge_accumulates_history():
    rule = HedgeRule(beta = 1.0, self_share = 0.5)
    for _ in range(10):
        rule.end_rollout(["a", "b"], np.array([0.3, 0.6]))
    out = rule.output(["a", "b"], np.array([0.9, 0.6]))
    # 当前 a 更强，但累计分数仍让 b 占大头：这就是 sleeper 延迟
    assert out.field[1] > out.field[0]


def test_fixed_rule_rejects_shares_without_self_play():
    with pytest.raises(ValueError):
        FixedRule({"a": 1.0})
    out = FixedRule({"a": 0.35}).output(["a", "b"], np.array([0.5, 0.5]))
    np.testing.assert_allclose(out.field, [0.35, 0.0, 0.65])


def test_empty_pool_is_pure_self_play():
    for rule in (CoDARRule(6.0), PFSPRule(2.0, 0.5), UniformRule(0.5), HedgeRule(1.0, 0.5)):
        out = rule.output([], np.zeros(0))
        np.testing.assert_allclose(out.field, [1.0])
