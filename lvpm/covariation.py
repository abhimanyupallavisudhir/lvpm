"""Infer the market-implied dependence structure from how prediction-market prices co-move.

Key identity. If p^i_t = E_t[X_i] is a martingale (an efficient market's price), then the
increments are orthogonal over time, so for any s < T (T = resolution, p^i_T = X_i):

    Cov_s(X_i, X_j) = E_s[ sum_{s<t<=T} dp^i_t dp^j_t ].

The realised quadratic covariation of the two belief paths is therefore an unbiased
(single-path) estimate of the market-implied covariance of the outcomes - no model needed.
To forecast at time t0 we use the *correlation* of increments in a window before t0 and
assume it persists, giving Cov_{t0}(X_i, X_j) = r_ij sqrt(p_i (1-p_i) p_j (1-p_j))."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from . import model as M

DAY = 86400


def daily_grid(histories: list[list[tuple[int, float]]], t_start: int, t_end: int) -> np.ndarray:
    """T x n matrix of end-of-day prices (forward-filled; NaN before a market's first trade)."""
    days = np.arange(t_start, t_end + 1, DAY)
    P = np.full((len(days), len(histories)), np.nan)
    for j, h in enumerate(histories):
        if not h:
            continue
        t, v = np.array(h, float).T
        idx = np.searchsorted(t, days, side="right") - 1
        ok = idx >= 0
        P[ok, j] = v[idx[ok]]
    return P


def increment_corr(P: np.ndarray, probit: bool = False, min_obs: int = 10, shrink: float = 10.0):
    """Pairwise-complete correlation of daily increments, shrunk towards 0 by n/(n + shrink)."""
    Q = ndtri(np.clip(P, 0.01, 0.99)) if probit else P
    D = np.diff(Q, axis=0)
    n = D.shape[1]
    r = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ok = ~np.isnan(D[:, i]) & ~np.isnan(D[:, j])
            a, b = D[ok, i], D[ok, j]
            if ok.sum() < min_obs or a.std() == 0 or b.std() == 0:
                continue
            nz = ((a != 0) | (b != 0)).sum()  # days on which anything happened
            r[i, j] = r[j, i] = np.corrcoef(a, b)[0, 1] * nz / (nz + shrink)
    np.fill_diagonal(r, 1.0)
    return r


def latent_corr_from_outcome_corr(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Map an outcome (phi-) correlation matrix to the threshold model's latent correlation."""
    n = len(p)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = r[i, j] * np.sqrt(p[i] * (1 - p[i]) * p[j] * (1 - p[j]))
            R[i, j] = R[j, i] = M.rho_from_p11(p[i], p[j], p[i] * p[j] + c)
    return M.nearest_correlation(R)


def realised_covariation(P: np.ndarray) -> np.ndarray:
    D = np.nan_to_num(np.diff(P, axis=0))
    return D.T @ D
