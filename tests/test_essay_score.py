import itertools

import numpy as np
import pytest

from lvpm.essays import score as S

# the worked example: two hypotheses about when RL exploits reward bugs, four environments
P = np.array([[0.9, 0.8, 0.2, 0.1],    # h1: reachability matters
              [0.8, 0.2, 0.7, 0.1]])   # h2: payoff matters
W = np.array([0.6, 0.4])
Y = np.array([1, 1, 0, 0])


def test_worked_example_numbers():
    assert S.s_tau(W, P, Y, 1.0) == pytest.approx(np.log(0.6 * 0.5184 + 0.4 * 0.0432))
    assert S.s_tau(W, P, Y, 1.0) == pytest.approx(-1.114, abs=1e-3)
    assert S.s_tau(W, P, Y, 0.9) == pytest.approx(-1.033, abs=1e-3)
    assert S.posterior(W, P, Y, 1.0)[0] == pytest.approx(0.947, abs=1e-3)
    assert S.posterior(W, P, Y, 0.9)[0] == pytest.approx(0.934, abs=1e-3)


def joint(w, P):
    """Implied distribution over all outcome patterns."""
    pats = np.array(list(itertools.product([0, 1], repeat=P.shape[1])))
    return pats, np.array([np.exp(S.s_tau(w, P, y, 1.0)) for y in pats])


def test_cheat_sheet_ties_at_tau_1_and_loses_below():
    pats, m = joint(W, P)
    wc, Pc = S.cheat_sheet(pats, m, eps=1e-12)
    for y in pats:
        assert S.s_tau(wc, Pc, y, 1.0) == pytest.approx(S.s_tau(W, P, y, 1.0), abs=1e-5)
        # p^tau >= p: the compact report weakly beats the cheat sheet on every outcome
        assert S.s_tau(W, P, y, 0.9) >= S.s_tau(wc, Pc, y, 0.9) - 1e-5
    # and strictly in expectation
    e_compact = sum(mi * S.s_tau(W, P, y, 0.9) for y, mi in zip(pats, m))
    e_cheat = sum(mi * S.s_tau(wc, Pc, y, 0.9) for y, mi in zip(pats, m))
    assert e_compact > e_cheat + 0.01


def test_held_out_score_is_conditional_log_likelihood():
    test = np.array([False, False, True, True])
    post = S.posterior(W, P, np.where(test, np.nan, Y))
    cond = np.log(post @ np.prod(np.where(Y[test] == 1, P[:, test], 1 - P[:, test]), axis=1))
    assert S.held_out_score(W, P, Y, test) == pytest.approx(cond)
    # one test task: the conditional is the predictive probability
    one = np.array([False, True, False, False])
    pred = S.predictive(W, P, np.where(one, np.nan, Y))
    assert S.held_out_score(W, P, Y, one) == pytest.approx(np.log(pred[1]))


def test_information_of_a_report():
    info = S.information(W, P, n_samples=20000)
    assert 0 < info["I_joint_bits"] <= info["H_weights_bits"] + 1e-9
    # a single hypothesis carries no information
    assert S.information([1.0], P[:1])["I_joint_bits"] == pytest.approx(0.0, abs=1e-9)
    # two identical hypotheses carry none either
    assert S.information([0.5, 0.5], np.vstack([P[0], P[0]]))["I_joint_bits"] == pytest.approx(0.0, abs=1e-9)


def test_importance_weighted_adaptive_selection_is_unbiased():
    rng = np.random.default_rng(1)
    n = 6
    base = np.full(n, 0.5)
    adaptive = S.adaptive_inclusion(base, rng.random(n), strength=2.0)
    f = lambda sel: S.s_tau(W, np.tile(P, 2)[:, :n], np.where(sel, np.tile(Y, 2)[:n], np.nan), 0.9)  # noqa: E731
    exact_base = 0.0
    exact_adapt = 0.0
    for sel in itertools.product([0, 1], repeat=n):
        sel = np.array(sel, dtype=bool)
        qb = np.prod(np.where(sel, base, 1 - base))
        qa = np.prod(np.where(sel, adaptive, 1 - adaptive))
        exact_base += qb * f(sel)
        exact_adapt += qa * S.importance_weight(sel, base, adaptive) * f(sel)
    assert exact_adapt == pytest.approx(exact_base)
