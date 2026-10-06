"""Continuous latent variable models: IRT, factor analysis, and mixtures of the two.

A latent Z in R^k (prior N(0, I), or a free distribution on a grid) and
indicators Y_1..Y_n, conditionally independent given Z, each one of

    Bernoulli(link(a.z + b))          link = logit or probit      (IRT)
    Normal(a.z + b, psi)                                          (factor analysis)
    Categorical(softmax(A z + B))                                 (nominal response)

Integrals over Z use fixed nodes: tensor Gauss-Hermite for k <= 3, scrambled
Sobol quasi-Monte Carlo above that (cost linear, not exponential, in k).
Gaussian indicators with a Gaussian prior also have an exact path for any k:
``fit_factor_analysis`` / ``factor_analysis_score``.
Data are rows of Y (NaN = unresolved indicator) with weights: an empirical
sample, or an exact table of response patterns.

The tempered log score (see notes/latent-meaning.md) is

    S_tau(theta; y) = log  integral  p(z) prod_i p(y_i | z)^tau  dz,

the ordinary log score at tau = 1. ``fit`` maximises its average by tempered
EM: E-step q(z|y) ∝ p(z) prod_i p(y_i|z)^tau, M-step = weighted maximum
likelihood for each indicator (and for the prior, if free).

Gauge: with the Gaussian prior, z -> R z (R orthogonal) together with
a -> R a leaves every prediction unchanged (``rotated``); for k = 1, R = -1.
"""

import copy
import itertools

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize
from scipy.special import log_expit, log_ndtr, logsumexp, ndtri
from scipy.stats import qmc

LN2 = np.log(2)
_LBFGS = dict(method="L-BFGS-B", options=dict(gtol=1e-13, ftol=1e-15, maxiter=500))


def gauss_hermite(k, m):
    """Tensor Gauss-Hermite grid for N(0, I_k): nodes (G, k), log weights (G,)."""
    x, w = hermegauss(m)
    w = w / w.sum()
    Z = np.stack([g.ravel() for g in np.meshgrid(*[x] * k, indexing="ij")], -1)
    logw = sum(np.log(g.ravel()) for g in np.meshgrid(*[w] * k, indexing="ij"))
    return Z, logw


def sobol_gaussian(k, n=4096, seed=0):
    """Scrambled Sobol nodes mapped to N(0, I_k), equal weights: (G, k), (G,)."""
    u = qmc.Sobol(d=k, scramble=True, seed=seed).random(n)
    return ndtri(np.clip(u, 1e-12, 1 - 1e-12)), np.full(n, -np.log(n))


def _design(Z):
    return np.hstack([Z, np.ones((len(Z), 1))])


class Bernoulli:
    """P(Y=1 | z) = F(a.z + b), F = logistic or standard normal CDF."""

    def __init__(self, a, b=0.0, link="logit"):
        self.theta = np.append(np.atleast_1d(np.asarray(a, float)), b)
        self.link = link
        self.categories = 2

    def _logF(self, eta):
        return log_expit(eta) if self.link == "logit" else log_ndtr(eta)

    def _dlogF(self, eta):
        if self.link == "logit":
            return np.exp(log_expit(-eta))
        return np.exp(-0.5 * eta ** 2 - 0.5 * np.log(2 * np.pi) - log_ndtr(eta))

    def logprob_table(self, Z):
        """(G, 2): log P(Y=c | z_g)."""
        eta = _design(Z) @ self.theta
        return np.stack([self._logF(-eta), self._logF(eta)], -1)

    def features(self, y):
        """log p(y|z_g) = features(y) @ table(Z).T  (NaN rows give 0)."""
        return np.stack([y == 0, y == 1], -1).astype(float)

    def table(self, Z):
        return self.logprob_table(Z)

    def m_step(self, y, W):
        s1, s0 = (y == 1) @ W, (y == 0) @ W  # NaN compares False: unresolved rows drop out
        X = _design(self.Z)
        if self.link == "logit":  # concave: damped Newton to convergence
            th = self.theta
            obj = lambda t: (s1 * log_expit(X @ t) + s0 * log_expit(-(X @ t))).sum()
            for _ in range(100):
                pr = np.exp(log_expit(X @ th))
                g = X.T @ (s1 * (1 - pr) - s0 * pr)
                H = (X.T * ((s1 + s0) * pr * (1 - pr) + 1e-12)) @ X
                step, f0 = np.linalg.solve(H, g), obj(th)
                while obj(th + step) < f0 - 1e-15 and np.abs(step).max() > 1e-14:
                    step = step / 2
                th = th + step
                if np.abs(step).max() < 1e-12:
                    break
            self.theta = th
            return

        def f(th):
            eta = X @ th
            val = -(s1 * self._logF(eta) + s0 * self._logF(-eta)).sum()
            g = s1 * self._dlogF(eta) - s0 * self._dlogF(-eta)
            return val, -X.T @ g

        self.theta = minimize(f, self.theta, jac=True, **_LBFGS).x


class Normal:
    """Y | z ~ N(a.z + b, psi)."""

    def __init__(self, a, b=0.0, psi=1.0):
        self.theta = np.append(np.atleast_1d(np.asarray(a, float)), b)
        self.psi = float(psi)
        self.categories = None

    def features(self, y):
        ok = ~np.isnan(y)
        y0 = np.where(ok, y, 0.0)
        return np.stack([ok, y0, y0 ** 2], -1).astype(float)

    def table(self, Z):
        mean = _design(Z) @ self.theta
        return np.stack([-0.5 * np.log(2 * np.pi * self.psi) - mean ** 2 / (2 * self.psi),
                         mean / self.psi, np.full(len(Z), -0.5 / self.psi)], -1)

    def m_step(self, y, W):
        ok = ~np.isnan(y)
        y0 = np.where(ok, y, 0.0)
        X = _design(self.Z)
        wg, wy, wyy = ok @ W, y0 @ W, (y0 ** 2) @ W  # sufficient statistics per node
        self.theta = np.linalg.solve(X.T @ (wg[:, None] * X), X.T @ wy)
        mean = X @ self.theta
        sse = (wyy - 2 * mean * wy + mean ** 2 * wg).sum()
        self.psi = max(float(sse / wg.sum()), 1e-8)


class Categorical:
    """P(Y=c | z) ∝ exp(A_c.z + B_c), category 0 the reference (A_0 = B_0 = 0)."""

    def __init__(self, A, B):
        A, B = np.atleast_2d(np.asarray(A, float)), np.asarray(B, float)
        self.categories = len(B) + 1
        self.theta = np.hstack([A, B[:, None]])  # (C-1, k+1)

    def logprob_table(self, Z):
        eta = np.hstack([np.zeros((len(Z), 1)), _design(Z) @ self.theta.T])
        return eta - logsumexp(eta, axis=1, keepdims=True)

    def features(self, y):
        return np.stack([y == c for c in range(self.categories)], -1).astype(float)

    def table(self, Z):
        return self.logprob_table(Z)

    def m_step(self, y, W):
        S = np.stack([(y == c) @ W for c in range(self.categories)], -1)
        X, shape = _design(self.Z), self.theta.shape

        def f(th):
            th = th.reshape(shape)
            eta = np.hstack([np.zeros((len(X), 1)), X @ th.T])
            logp = eta - logsumexp(eta, axis=1, keepdims=True)
            r = S - S.sum(1, keepdims=True) * np.exp(logp)
            return -(S * logp).sum(), -(r[:, 1:].T @ X).ravel()

        self.theta = minimize(f, self.theta.ravel(), jac=True, **_LBFGS).x.reshape(shape)


class LatentFactorModel:
    def __init__(self, indicators, k=1, prior="gaussian", grid=None, m=None):
        """indicators: list of Bernoulli / Normal / Categorical.
        prior: 'gaussian' (N(0, I), fixed: this fixes the gauge up to rotation)
               or 'free' (any distribution on the grid nodes, learned).
        grid: (Z, logw) nodes and prior log weights. Default: Gauss-Hermite
              with m points per dimension for k <= 3, else m Sobol nodes."""
        self.k, self.indicators = k, indicators
        self.free_prior = prior == "free"
        if grid is None:
            grid = gauss_hermite(k, m or {1: 61, 2: 25, 3: 13}[k]) if k <= 3 \
                else sobol_gaussian(k, m or 4096)
        self.Z, self.logw = np.asarray(grid[0], float).reshape(-1, k), np.asarray(grid[1], float)
        self.logw = self.logw - logsumexp(self.logw)
        for ind in indicators:
            ind.Z = self.Z

    def copy(self):
        return copy.deepcopy(self)

    # ---------------------------------------------------------------- scoring
    def loglik(self, Y):
        """(N, G): sum_i log p(y_i | z_g), as one matrix product."""
        Y = np.atleast_2d(np.asarray(Y, float))
        F = np.hstack([ind.features(Y[:, i]) for i, ind in enumerate(self.indicators)])
        T = np.hstack([ind.table(self.Z) for ind in self.indicators])
        return F @ T.T

    def scores(self, Y, tau=1.0):
        """Tempered log score of each row of Y, in bits."""
        return logsumexp(self.logw + tau * self.loglik(Y), axis=1) / LN2

    def score(self, Y, w=None, tau=1.0):
        """Weighted average tempered log score, in bits."""
        s = self.scores(Y, tau)
        return float(s.mean() if w is None else np.dot(w, s))

    def posterior(self, Y, tau=1.0):
        a = self.logw + tau * self.loglik(Y)
        return np.exp(a - logsumexp(a, axis=1, keepdims=True))

    # ---------------------------------------------------------------- fitting
    def fit(self, Y, w=None, tau=1.0, iters=2000, tol=1e-10, fit_indicators=True):
        """Tempered EM. Returns the trace of the average tempered score."""
        Y = np.atleast_2d(np.asarray(Y, float))
        w = np.full(len(Y), 1 / len(Y)) if w is None else np.asarray(w, float)
        trace = []
        for _ in range(iters + 1):
            W = self.logw + tau * self.loglik(Y)  # becomes the E-step weights in place
            lse = logsumexp(W, axis=1, keepdims=True)
            trace.append(float(w @ lse[:, 0]) / LN2)
            if len(trace) > 1 and trace[-1] - trace[-2] < tol or len(trace) > iters:
                break
            W -= lse
            np.exp(W, out=W)
            W *= w[:, None]
            if fit_indicators:
                for i, ind in enumerate(self.indicators):
                    ind.m_step(Y[:, i], W)
            if self.free_prior:
                self.logw = np.log(np.maximum(W.sum(0), 1e-300))
                self.logw -= logsumexp(self.logw)
        return trace

    # ----------------------------------------------------- discrete indicators
    def patterns(self):
        """All response patterns (P, n) and their model probabilities (P,);
        discrete indicators only."""
        cats = [ind.categories for ind in self.indicators]
        assert all(cats), "patterns() needs discrete indicators"
        Y = np.array(list(itertools.product(*[range(c) for c in cats])), float)
        return Y, np.exp(self.scores(Y) * LN2)

    def info(self):
        """I(Z;Y) in bits (discrete indicators): H(Y) - E_z sum_i H(Y_i | z)."""
        _, p = self.patterns()
        h_y = -(p[p > 0] * np.log2(p[p > 0])).sum()
        w = np.exp(self.logw)
        h_y_z = 0.0
        for ind in self.indicators:
            t = ind.logprob_table(self.Z)
            h_y_z += -(w[:, None] * np.exp(t) * t).sum() / LN2
        return float(h_y - h_y_z)

    # ---------------------------------------------------------------- gauge
    def rotated(self, R):
        """The same model in rotated coordinates z' = R z (Gaussian prior)."""
        R = np.atleast_2d(R)
        out = self.copy()
        for ind in out.indicators:
            if isinstance(ind, Categorical):
                ind.theta[:, :-1] = ind.theta[:, :-1] @ R.T
            else:
                ind.theta[:-1] = R @ ind.theta[:-1]
        return out

    def sample(self, N, rng):
        g = rng.choice(len(self.Z), size=N, p=np.exp(self.logw))
        Z = self.Z[g]
        X = _design(Z)
        cols = []
        for ind in self.indicators:
            if isinstance(ind, Normal):
                cols.append(X @ ind.theta + rng.normal(0, np.sqrt(ind.psi), N))
            else:
                t = np.exp(ind.logprob_table(Z))
                cols.append((rng.random(N)[:, None] > np.cumsum(t, 1)).sum(1).astype(float))
        return np.stack(cols, -1)


def factor_analysis_score(A, b, psi, Y, tau=1.0):
    """Closed-form tempered log score (bits) of a Gaussian factor model,
    p(z) = N(0, I), Y_i | z ~ N(A_i.z + b_i, psi_i): per row of Y.

    S_tau = sum_i [(1-tau)/2 log(2 pi psi_i) - 1/2 log tau] + log N(y; b, A A^T + Psi/tau)."""
    A, psi = np.atleast_2d(A), np.asarray(psi, float)
    C = A @ A.T + np.diag(psi / tau)
    d = np.atleast_2d(Y) - b
    _, logdet = np.linalg.slogdet(C)
    quad = np.einsum("ni,ij,nj->n", d, np.linalg.inv(C), d)
    lognorm = -0.5 * (len(psi) * np.log(2 * np.pi) + logdet + quad)
    const = (0.5 * (1 - tau) * np.log(2 * np.pi * psi) - 0.5 * np.log(tau)).sum()
    return (const + lognorm) / LN2


def factor_analysis_expected_score(A, b, psi, mean, cov, tau=1.0):
    """Expected closed-form tempered score (bits) when Y ~ N(mean, cov)."""
    A, psi = np.atleast_2d(A), np.asarray(psi, float)
    C = A @ A.T + np.diag(psi / tau)
    d = mean - b
    _, logdet = np.linalg.slogdet(C)
    Ci = np.linalg.inv(C)
    e = -0.5 * (len(psi) * np.log(2 * np.pi) + logdet + np.trace(Ci @ cov) + d @ Ci @ d)
    const = (0.5 * (1 - tau) * np.log(2 * np.pi * psi) - 0.5 * np.log(tau)).sum()
    return float((const + e) / LN2)


def factor_analysis_info(A, psi):
    """I(Z;Y) in bits for a Gaussian factor model: 1/2 log det(AA^T + Psi) / det Psi."""
    A, psi = np.atleast_2d(A), np.asarray(psi, float)
    _, logdet = np.linalg.slogdet(A @ A.T + np.diag(psi))
    return float(0.5 * (logdet - np.log(psi).sum()) / LN2)


def fit_factor_analysis(Y, k, tau=1.0, iters=5000, tol=1e-12, rng=None, init=None):
    """Tempered EM for Gaussian factor analysis, exact for any k.

    E-step: q(z|y) = N(m, V), V = (I + tau A^T Psi^-1 A)^-1, m = V tau A^T Psi^-1 (y - b).
    M-step: regress y on (z, 1) under q (closed form), Psi = residual variances.
    Returns (A, b, psi, trace of the average tempered score in bits)."""
    Y = np.asarray(Y, float)
    N, n = Y.shape
    rng = np.random.default_rng() if rng is None else rng
    if init is None:
        A, b, psi = rng.normal(0, 0.5, (n, k)), Y.mean(0), Y.var(0)
    else:
        A, b, psi = (np.array(x, float) for x in init)
    trace = []
    for _ in range(iters + 1):
        trace.append(float(factor_analysis_score(A, b, psi, Y, tau).mean()))
        if len(trace) > 1 and trace[-1] - trace[-2] < tol or len(trace) > iters:
            break
        AtPi = A.T / psi
        V = np.linalg.inv(np.eye(k) + tau * AtPi @ A)
        M = tau * (Y - b) @ AtPi.T @ V  # (N, k) posterior means
        Ez = np.hstack([M, np.ones((N, 1))])
        Ezz = Ez.T @ Ez
        Ezz[:k, :k] += N * V
        W = np.linalg.solve(Ezz, Ez.T @ Y).T  # (n, k+1) = [A b]
        A, b = W[:, :k], W[:, k]
        psi = np.maximum(((Y ** 2).sum(0) - (W * (Y.T @ Ez)).sum(1)) / N, 1e-8)
    return A, b, psi, trace
