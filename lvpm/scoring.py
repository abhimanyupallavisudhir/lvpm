"""Scores for joint (latent-structure) forecasts over a cluster of binary indicators.

log P(x) = sum_i log P(x_i)  +  [log P(x) - sum_i log P(x_i)]
           \\_ marginal skill _/    \\______ structure skill ______/

Ordinary forecasting benchmarks only score the first term. A latent variable net is
only useful beyond them through the second, so we score it separately by holding the
marginals fixed and swapping the copula ("copula swap")."""
from __future__ import annotations

import numpy as np

from . import model as M


def joint_loglik(x, p, R=None, L=None) -> float:
    """log P(x): exact quadrature for a factor model (L given), Genz for a general R."""
    if L is not None:
        return M.factor_loglik(x, p, L)
    if np.allclose(R, np.eye(len(p))):
        return M.independent_loglik(x, p)
    return M.gaussian_copula_loglik(x, p, R)


def structure_score(x, p, R=None, L=None) -> float:
    """Gain (nats) of the dependence structure over independence, with the same marginals."""
    return joint_loglik(x, p, R, L) - M.independent_loglik(x, p)


def covariance_pairs(x, p, R):
    """(predicted Cov(X_i, X_j), realised (x_i - p_i)(x_j - p_j)) for every resolved pair.

    A calibrated joint has E[realised | predicted] = predicted (slope 1 through the origin)."""
    out = []
    obs = np.where(~np.isnan(np.asarray(x, float)))[0]
    for a in range(len(obs)):
        for b in range(a + 1, len(obs)):
            i, j = obs[a], obs[b]
            c = M.p11_from_rho(p[i], p[j], R[i, j]) - p[i] * p[j]
            out.append((c, (x[i] - p[i]) * (x[j] - p[j])))
    return np.array(out).reshape(-1, 2)


def calibration_slope(pairs_by_cluster):
    """Through-origin slope of realised on predicted covariance, cluster-robust SE."""
    P = np.concatenate([q for q in pairs_by_cluster if len(q)])
    sxx = (P[:, 0] ** 2).sum()
    beta = (P[:, 0] * P[:, 1]).sum() / sxx
    meat = sum(((q[:, 0] * (q[:, 1] - beta * q[:, 0])).sum()) ** 2 for q in pairs_by_cluster if len(q))
    return beta, np.sqrt(meat) / sxx


def bootstrap_mean(v, n: int = 5000, seed: int = 0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, (n, len(v))).mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
