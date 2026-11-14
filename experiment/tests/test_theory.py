import numpy as np

from experiment.theory.experiments import exact_run
from experiment.theory.games import FAMILIES, exploitability, make_game, rps
from experiment.theory.rules import adversarial_run, epsilon_br, oracle_rate_run, sleeper_run
from experiment.theory.solver import solve_step


def test_game_families_are_skew_with_bounded_entries():
    for family in FAMILIES:
        game = make_game(family, 3 if family == "blotto" else 5, 2, seed = 0)
        np.testing.assert_allclose(game.s, -game.s.T)
        assert np.max(np.abs(game.s)) <= 0.5 + 1e-12
        assert exploitability(game.s, game.nash) < 1e-8


def test_share_one_witness_is_a_fixed_point_only_under_the_moving_anchor():
    s = rps()
    pool = np.array([[0.0], [0.5], [0.5]])
    x = np.array([2 / 3, 1 / 6, 1 / 6])
    moving = solve_step(s, pool, x, eta = 1.0, c = 6.0, anchor = np.array([1.0, 0.0]))
    np.testing.assert_allclose(moving.x, x, atol = 1e-8)
    assert exploitability(s, moving.x) > 0.24
    fixed = solve_step(s, pool, x, eta = 1.0, c = 6.0)
    assert np.linalg.norm(fixed.x - x) > 1e-3


def _payload(**kw):
    base = dict(exp = "theory-convergence", family = "random", size = 6, n = 4, c = 6.0, eta = 0.1, e = 0.0, delta = 0.0,
                steps = 64, mode = "default", anchor = "fixed")
    return {**base, **kw}


def test_correct_algorithm_has_no_violations_and_converges():
    run = exact_run(_payload(), seed = 0)
    assert all(v == 0 for v in run["violations"].values())
    assert run["final_expl"] < run["series"]["full_expl"][0]


def test_inexact_checks_hold_with_injected_errors():
    run = exact_run(_payload(exp = "theory-inexact", e = 0.05, delta = 0.01), seed = 1)
    assert all(v == 0 for v in run["violations"].values())


def test_explicit_mirror_is_caught_by_the_energy_check():
    run = exact_run(_payload(n = 0, mode = "explicit"), seed = 0)
    assert run["violations"]["fejer_violation"] > 0


def test_pool_rules_run_reports_sleeper_metrics():
    payload = dict(exp = "theory-pool-rules", rule = "hedge", m = 6, n = 4, c = 6.0, eta = 0.1, steps = 32, reset_at = 16,
                   insert_every = 8)
    run = sleeper_run(payload, seed = 0)
    assert np.isfinite(run["final_expl"]) and "sleeper_delay" in run


def test_epsilon_br_leads_by_exactly_expl_minus_eps():
    game = make_game("random", 6, 0, seed = 2)
    x = game.rng.dirichlet(np.ones(game.m))
    for eps in (0.0, 0.01):
        y = epsilon_br(game.s, x, eps)
        assert abs(float(y.sum()) - 1.0) < 1e-12
        assert abs(float(x @ game.s @ y) - (exploitability(game.s, x) - eps)) < 1e-12


def test_theorem9_holds_with_a_best_response_oracle():
    run = oracle_rate_run(dict(family = "random", size = 6, eta = 4.0, arm = "coda_r", eps = 0.0, steps = 32), seed = 0)
    assert run["violations"] == dict(theorem9_violation = 0)
    assert run["final_expl"] < run["series"]["full_expl"][0]


def test_complement_adversary_stalls_fixed_share_rules_but_not_coda_r():
    base = dict(family = "random", m = 6, adversary = "complement", c = 6.0, eta = 0.1, steps = 24)
    coda = adversarial_run({**base, "rule": "coda_r"}, seed = 0)
    uniform = adversarial_run({**base, "rule": "uniform"}, seed = 0)
    assert coda["violations"] == dict(fejer_violation = 0)
    assert coda["final_expl"] < uniform["final_expl"]
