"""Threshold (probit) latent-variable models over binary indicators.

Every model here is a Gaussian copula with exact marginals, as in tailcalled's
"store P(X_i) and a loading" parameterisation:

    Z ~ N(0, R),  R_ii = 1,    X_i = 1[Z_i > -tau_i],   tau_i = Phi^{-1}(p_i),

so P(X_i = 1) = p_i exactly and all dependence lives in the latent correlation R.
A k-factor IRT model is the special case R = L L^T + diag(1 - |L_i|^2), i.e.

    Z_i = L_i . Y + sqrt(1 - |L_i|^2) eps_i,   Y ~ N(0, I_k).

Gauge: L -> L O for any orthogonal O leaves the distribution of X unchanged
(k = 1: only the sign flip Y -> -Y).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from scipy.special import log_ndtr, ndtri

_GH = {}


def _gauss_hermite(m: int):
    """Nodes/weights for E[f(Y)], Y ~ N(0, 1)."""
    if m not in _GH:
        x, w = np.polynomial.hermite_e.hermegauss(m)
        _GH[m] = (x, w / w.sum())
    return _GH[m]


def _signs(x):
    x = np.asarray(x, float)
    obs = ~np.isnan(x)
    return obs, np.where(x[obs] == 1, 1.0, -1.0)


def _clip(p, eps=1e-4):
    return np.clip(np.asarray(p, float), eps, 1 - eps)


def factor_loglik(x, p, L, m: int = 48) -> float:
    """log P(X = x) under the k-factor probit model (k = 1 or 2), by Gauss-Hermite quadrature.

    x: 0/1 vector, NaN = unresolved/annulled (simply marginalised out)."""
    L = np.atleast_2d(np.asarray(L, float).T).T if np.ndim(L) == 1 else np.asarray(L, float)
    obs, s = _signs(x)
    tau = ndtri(_clip(p))[obs]
    L = L[obs]
    k = L.shape[1]
    u = np.sqrt(np.clip(1 - (L ** 2).sum(1), 1e-6, None))
    y, w = _gauss_hermite(m if k == 1 else max(16, m // 2))
    if k == 1:
        Y, W = y[:, None], w
    elif k == 2:
        Y = np.stack(np.meshgrid(y, y, indexing="ij"), -1).reshape(-1, 2)
        W = np.outer(w, w).ravel()
    else:
        raise ValueError("use gaussian_copula_loglik for k > 2")
    # P(X_i = x_i | Y) = Phi(s_i (tau_i + L_i.Y) / u_i)
    lp = log_ndtr(s * (tau[None, :] + Y @ L.T) / u[None, :]).sum(1)
    return float(np.log(np.sum(W * np.exp(lp - lp.max()))) + lp.max())


def gaussian_copula_loglik(x, p, R, abseps: float = 1e-7, maxpts: int = 200_000, seed: int = 0) -> float:
    """log P(X = x) for a general latent correlation matrix R (Genz orthant probability).

    P(X = x) = P(s_i (-Z_i) < s_i tau_i for all i) = Phi_n(s * tau; S R S), with -Z ~ N(0, R)."""
    obs, s = _signs(x)
    tau = ndtri(_clip(p))[obs]
    R = np.asarray(R, float)[np.ix_(obs, obs)]
    n = len(s)
    if n == 0:
        return 0.0
    if np.allclose(R, np.eye(n)):
        return float(log_ndtr(s * tau).sum())
    S = np.outer(s, s)
    np.random.seed(seed)  # Genz QMC uses the global RNG in older scipy
    val = stats.multivariate_normal.cdf(s * tau, mean=np.zeros(n), cov=R * S, allow_singular=True,
                                        abseps=abseps, releps=1e-5, maxpts=maxpts)
    return float(np.log(max(val, 1e-300)))


def independent_loglik(x, p) -> float:
    obs, s = _signs(x)
    return float(log_ndtr(s * ndtri(_clip(p))[obs]).sum())


def factor_corr(L) -> np.ndarray:
    L = np.asarray(L, float).reshape(len(L), -1)
    R = L @ L.T
    np.fill_diagonal(R, 1.0)
    return R


def sample(p, R, size: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    Z = rng.multivariate_normal(np.zeros(len(p)), R, size=size, method="eigh")
    return (Z > -ndtri(_clip(p))).astype(float)


# ---------- conversions between association measures and the latent correlation ----------

def p11_from_rho(pi, pj, rho) -> float:
    """P(X_i = 1, X_j = 1) for the bivariate threshold model."""
    if abs(rho) < 1e-12:
        return pi * pj
    ti, tj = ndtri(_clip(pi)), ndtri(_clip(pj))
    return float(stats.multivariate_normal(cov=[[1, rho], [rho, 1]]).cdf([ti, tj]))


def rho_from_p11(pi, pj, p11) -> float:
    """Tetrachoric correlation reproducing a given joint P(1,1) (clamped to Frechet bounds)."""
    lo, hi = max(0.0, pi + pj - 1), min(pi, pj)
    p11 = float(np.clip(p11, lo + 1e-9, hi - 1e-9)) if hi - lo > 2e-9 else pi * pj
    f = lambda r: p11_from_rho(pi, pj, r) - p11
    a, b = -0.999, 0.999
    if f(a) > 0:
        return a
    if f(b) < 0:
        return b
    return float(optimize.brentq(f, a, b, xtol=1e-6))


def rho_from_conditionals(pi, pj, q1, q0) -> float:
    """Latent correlation from elicited P(X_j=1 | X_i=1) = q1 and P(X_j=1 | X_i=0) = q0.

    Uses only the odds ratio of (q1, q0) - which does not depend on the marginals - and
    finds the rho whose bivariate threshold model with marginals (pi, pj) has that odds ratio.
    This tolerates elicited conditionals that are incoherent with the elicited marginals."""
    q1, q0 = _clip(q1, 1e-3), _clip(q0, 1e-3)
    log_or = np.log(q1 / (1 - q1)) - np.log(q0 / (1 - q0))
    pi, pj = float(_clip(pi, 1e-3)), float(_clip(pj, 1e-3))

    def lor(r):
        p11 = p11_from_rho(pi, pj, r)
        p10, p01 = pi - p11, pj - p11
        p00 = 1 - pi - pj + p11
        return np.log(max(p11, 1e-12) * max(p00, 1e-12) / max(p10 * p01, 1e-24))

    a, b = -0.99, 0.99
    if lor(a) > log_or:
        return a
    if lor(b) < log_or:
        return b
    return float(optimize.brentq(lambda r: lor(r) - log_or, a, b, xtol=1e-5))


def loading_from_conditionals(p, q_hi, q_lo, y: float = 1.0) -> float:
    """1-factor loading from P(X=1 | Y=+y) and P(X=1 | Y=-y) with Y ~ N(0,1).

    Model: P(X=1 | Y) = Phi((tau + lam Y) / sqrt(1 - lam^2)) (sign: positive lam = more likely when Y high).
    Least-squares fit of lam in probit space to both elicited points, holding tau = Phi^{-1}(p)."""
    tau = ndtri(_clip(p))
    targets = ndtri(_clip([q_hi, q_lo], 1e-3))

    def loss(lam):
        u = np.sqrt(1 - lam ** 2)
        return ((tau + lam * y) / u - targets[0]) ** 2 + ((tau - lam * y) / u - targets[1]) ** 2

    return float(optimize.minimize_scalar(loss, bounds=(-0.98, 0.98), method="bounded").x)


def nearest_correlation(R, iters: int = 100) -> np.ndarray:
    """Higham (2002) alternating projections onto {PSD} and {unit diagonal}."""
    Y = np.asarray(R, float).copy()
    dS = np.zeros_like(Y)
    for _ in range(iters):
        Rk = Y - dS
        w, V = np.linalg.eigh((Rk + Rk.T) / 2)
        X = (V * np.clip(w, 1e-6, None)) @ V.T
        dS = X - Rk
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if np.linalg.eigvalsh(Y).min() > -1e-9:
            break
    return Y


def fit_factor(R, k: int = 1, iters: int = 500) -> np.ndarray:
    """Minimum-residual factor analysis of a correlation matrix (iterated principal axes)."""
    R = np.asarray(R, float)
    h = np.clip(1 - 1 / np.diag(np.linalg.pinv(R)), 0.05, 0.95)  # SMC starting communalities
    for _ in range(iters):
        Rr = R.copy()
        np.fill_diagonal(Rr, h)
        w, V = np.linalg.eigh(Rr)
        w, V = w[::-1][:k], V[:, ::-1][:, :k]
        L = V * np.sqrt(np.clip(w, 0, None))
        h_new = np.clip((L ** 2).sum(1), 0, 0.98)
        if np.max(np.abs(h_new - h)) < 1e-6:
            break
        h = h_new
    return L / np.maximum(1, np.sqrt((L ** 2).sum(1, keepdims=True)) / 0.98)


def align(L, target) -> np.ndarray:
    """Fix the gauge: orthogonal Procrustes rotation of L onto target (sign flip when k = 1)."""
    L, target = np.asarray(L, float).reshape(len(L), -1), np.asarray(target, float).reshape(len(target), -1)
    U, _, Vt = np.linalg.svd(L.T @ target)
    return L @ U @ Vt
