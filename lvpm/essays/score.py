"""Scoring natural-language latent reports with the tempered log score.

A report is K hypotheses h_k (essays) with weights w_k. A frozen reader turns
each hypothesis into probabilities for binary tasks: P[k, i] = p_R(y_i = 1 | h_k),
one call per (hypothesis, task), so the tasks are conditionally independent
given the hypothesis by construction - the X -> Y_i structure of
notes/latent-meaning.md, with hypotheses as latent states, weights as the prior
and the reader's predictions as the conditionals. On resolved outcomes y,

    S_tau(w, P; y) = log sum_k w_k prod_i p_k(y_i)^tau,

whose expectation is maximized by the latent-bottleneck optimum
I(H;Y) - tau sum_i I(H;Y_i). With one hypothesis tau only rescales the
log score; the information cost lives in the mixture: tempering inflates soft
predictions (p^tau > p for p < 1) and gives nothing to deterministic ones, so a
"cheat sheet" that enumerates outcome patterns loses to a report that explains
the same joint with a few soft hypotheses.

Everything here is plain numpy on the probability matrix; the reader lives in
reader.py.
"""

import numpy as np
from scipy.special import logsumexp

EPS = 1e-12


def _norm(w):
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def loglik(P, y):
    """[K, n] matrix of log p_k(y_i) (NaN outcomes are unresolved and count 0)."""
    P = np.clip(np.asarray(P, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    L = np.where(y == 1, np.log(P), np.log1p(-P))
    return np.where(np.isnan(y), 0.0, L)


def s_tau(w, P, y, tau=1.0):
    """The tempered log score of report (w, P) on outcomes y, in nats."""
    w = _norm(w)
    return float(logsumexp(np.log(np.maximum(w, EPS)) + tau * loglik(P, y).sum(axis=1)))


def posterior(w, P, y, tau=1.0):
    """Tempered posterior over hypotheses after outcomes y (the optimal encoder)."""
    w = _norm(w)
    a = np.log(np.maximum(w, EPS)) + tau * loglik(P, y).sum(axis=1)
    return np.exp(a - logsumexp(a))


def marginals(w, P):
    """The report's prediction for each task on its own: sum_k w_k p_k(y_i = 1)."""
    return _norm(w) @ np.asarray(P, dtype=float)


def predictive(w, P, y_seen, tau=1.0):
    """P(y_j = 1 | outcomes seen) for every task: the report updated on y_seen.

    Hypotheses are reweighted by their (tempered) fit to the resolved tasks and
    then predict the unresolved ones - the use a latent belief has.
    """
    return posterior(w, P, y_seen, tau) @ np.asarray(P, dtype=float)


def held_out_score(w, P, y, test, tau=1.0):
    """log P(y_test | y_train) under the report: S on all tasks minus S on train.

    This is the part of the joint score that a report earns by explaining
    the dependence between tasks it saw resolved and tasks it did not.
    """
    test = np.asarray(test, dtype=bool)
    y = np.asarray(y, dtype=float)
    y_train = np.where(test, np.nan, y)
    return s_tau(w, P, y, tau) - s_tau(w, P, y_train, tau)


def information(w, P, n_samples=4000, rng=None):
    """Bits the report's latent carries about the outcomes, under its own model.

    I(H;Y) = H(w) - E_y H(posterior(y)), with y drawn from the report's own
    mixture (Monte Carlo). Also returns the per-task part sum_i I(H;Y_i), so
    the latent bottleneck I(H;Y) - tau sum_i I(H;Y_i) of the report can be read
    off.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    w, P = _norm(w), np.asarray(P, dtype=float)
    K, n = P.shape

    def H(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    ks = rng.choice(K, size=n_samples, p=w)
    Y = (rng.random((n_samples, n)) < P[ks]).astype(float)
    post_H = np.mean([H(posterior(w, P, y)) for y in Y])
    i_joint = H(w) - post_H
    m = marginals(w, P)

    def hb(p):
        p = np.clip(p, EPS, 1 - EPS)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    i_each = float((hb(m) - w @ hb(P)).sum())
    return {"I_joint_bits": i_joint, "I_sum_bits": i_each, "H_weights_bits": H(w)}


def cheat_sheet(y_patterns, weights, eps=1e-6):
    """A report enumerating outcome patterns: one deterministic hypothesis per row."""
    Y = np.asarray(y_patterns, dtype=float)
    return _norm(weights), np.clip(Y, eps, 1 - eps)


# ---------------------------------------------------------------- adaptive tasks
def disagreement(reports):
    """Per-task spread of the competing reports' marginal predictions (std)."""
    M = np.array([marginals(w, P) for w, P in reports])
    return M.std(axis=0)


def adaptive_inclusion(base, score, strength=4.0, floor=0.05):
    """Inclusion probabilities tilted towards high-``score`` tasks, each at least ``floor``.

    ``base`` is the baseline inclusion probability of each task (the fixed
    design whose incentives we want to keep).
    """
    base = np.asarray(base, dtype=float)
    s = np.asarray(score, dtype=float)
    s = (s - s.min()) / (np.ptp(s) + EPS)
    tilt = base * np.exp(strength * s)
    tilt = tilt * base.sum() / tilt.sum()  # same expected number of tasks
    return np.clip(tilt, floor, 1 - floor)


def importance_weight(selected, base, adaptive):
    """q_base(S) / q_adaptive(S) for an independently-sampled subset S.

    Multiplying any score of the selected tasks by this weight makes its
    expectation over adaptive subsets equal to its expectation over baseline
    subsets, for every fixed set of reports - so task selection steered by the
    reports' disagreement leaves every writer's incentives unchanged.
    """
    s = np.asarray(selected, dtype=bool)
    b, a = np.asarray(base, dtype=float), np.asarray(adaptive, dtype=float)
    return float(np.exp(np.where(s, np.log(b) - np.log(a), np.log1p(-b) - np.log1p(-a)).sum()))
