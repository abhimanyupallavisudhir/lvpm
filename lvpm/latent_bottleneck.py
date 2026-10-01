"""Information-theoretic scoring of latent variables for observables Y1..Yn.

A joint over observables is an ndarray ``py`` of shape (d1, ..., dn).
A latent X is given either as
  * a coupling ``qxy`` = q(x | y), shape (k, d1, ..., dn), or
  * a model (pi, conds) with pi shape (k,) and conds[i] shape (k, di),
    meaning p(x, y) = pi[x] * prod_i conds[i][x, y_i]  (the views are
    conditionally independent given X).

Everything is in bits. The central objective is the latent bottleneck

    L_tau(X) = I(X; Y) - tau * sum_i I(X; Y_i),        tau in [0, 1),

which equals (I(X;Y) + beta * TC(Y|X)) / (1 + beta) up to a constant, with
beta = tau / (1 - tau) and TC(Y|X) = sum_i H(Y_i|X) - H(Y|X) the dependence
among the views left unexplained by X. Its tau -> 1 limit is Wyner's common
information; restricted to X - Y1 - Y2 it is the information bottleneck.
It is minimised by tempered EM:  q(x|y) ∝ p(x) prod_i p(y_i|x)^tau.
"""

import numpy as np

EPS = 1e-300


def entropy(p):
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def model_joint(pi, conds):
    """p(x, y) for a conditionally independent latent model, shape (k, d1..dn)."""
    k, n = len(pi), len(conds)
    out = np.asarray(pi, dtype=float).reshape((k,) + (1,) * n)
    for i, c in enumerate(conds):
        shape = [k] + [1] * n
        shape[i + 1] = c.shape[1]
        out = out * c.reshape(shape)
    return out


def posterior(pi, conds):
    """q(x | y) under the model, shape (k, d1..dn)."""
    pxy = model_joint(pi, conds)
    return pxy / np.maximum(pxy.sum(0, keepdims=True), EPS)


def model_from_coupling(pxy):
    """p(x) and p(y_i | x) of a joint p(x, y)."""
    n = pxy.ndim - 1
    px = pxy.sum(axis=tuple(range(1, n + 1)))
    conds = []
    for i in range(n):
        axes = tuple(j + 1 for j in range(n) if j != i)
        pxi = pxy.sum(axis=axes)
        conds.append(pxi / np.maximum(px[:, None], EPS))
    return px, conds


def stats(pxy):
    """Information quantities of a joint p(x, y1..yn)."""
    n = pxy.ndim - 1
    px = pxy.sum(axis=tuple(range(1, n + 1)))
    py = pxy.sum(0)
    h_x, h_y, h_xy = entropy(px), entropy(py), entropy(pxy)
    i_xy = h_x + h_y - h_xy
    i_views, h_views_given_x = [], []
    for i in range(n):
        axes = tuple(j + 1 for j in range(n) if j != i)
        pxi = pxy.sum(axis=axes)
        h_i = entropy(pxi.sum(0))
        h_i_given_x = entropy(pxi) - h_x
        i_views.append(h_i - h_i_given_x)
        h_views_given_x.append(h_i_given_x)
    tc_given_x = sum(h_views_given_x) - (h_xy - h_x)
    tc = sum(entropy(pxy.sum(axis=tuple(j + 1 for j in range(n) if j != i)).sum(0))
             for i in range(n)) - h_y
    return dict(rate=i_xy, residual=max(tc_given_x, 0.0), tc=tc,
                i_views=i_views, h_x=h_x)


def tempered_em(py, k, tau, q0=None, rng=None, iters=3000, tol=1e-11):
    """Minimise L_tau over couplings q(x|y) with |X| = k. Returns q(x|y)."""
    rng = np.random.default_rng() if rng is None else rng
    n = py.ndim
    if q0 is None:
        q = rng.dirichlet(np.ones(k), size=py.shape).transpose((n,) + tuple(range(n)))
    else:
        q = q0
    prev = np.inf
    for t in range(iters):
        px, conds = model_from_coupling(q * py)
        logq = np.log(np.maximum(px, EPS)).reshape((k,) + (1,) * n)
        for i, c in enumerate(conds):
            shape = [k] + [1] * n
            shape[i + 1] = c.shape[1]
            logq = logq + tau * np.log(np.maximum(c, EPS)).reshape(shape)
        logq -= logq.max(0, keepdims=True)
        q = np.exp(logq)
        q /= q.sum(0, keepdims=True)
        if t % 25:
            continue
        s = stats(q * py)
        obj = s["rate"] - tau * sum(s["i_views"])
        if prev - obj < tol:
            break
        prev = obj
    return q


def bottleneck_curve(py, k, taus, restarts=20, seed=0):
    """Solutions of the latent bottleneck over a grid of tau (best of restarts,
    each tau also warm-started from the previous solution)."""
    rng = np.random.default_rng(seed)
    out, warm = [], None
    for tau in taus:
        cands = [tempered_em(py, k, tau, rng=rng) for _ in range(restarts)]
        if warm is not None:
            cands.append(tempered_em(py, k, tau, q0=warm, rng=rng))
        objs = []
        for q in cands:
            s = stats(q * py)
            objs.append(s["rate"] - tau * sum(s["i_views"]))
        warm = cands[int(np.argmin(objs))]
        s = stats(warm * py)
        out.append(dict(tau=tau, rate=s["rate"], residual=s["residual"], q=warm))
    return out


def em(py, k, rng, iters=5000, tol=1e-13):
    """Maximum-likelihood latent class model (tempered EM at tau = 1)."""
    q = tempered_em(py, k, 1.0, rng=rng, iters=iters, tol=tol)
    return model_from_coupling(q * py)


def log_loss(py, pi, conds):
    """Expected log score (bits) of the model's predictions for Y, under py."""
    m = model_joint(pi, conds).sum(0)
    mask = py > 0
    return float(-(py[mask] * np.log2(np.maximum(m[mask], EPS))).sum())
