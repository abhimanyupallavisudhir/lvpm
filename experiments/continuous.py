"""Checks for the continuous-latent parts of notes/latent-meaning.md.

Run:  PYTHONPATH=. .venv/bin/python experiments/continuous.py   (a few minutes)
"""

import numpy as np
from scipy.optimize import linprog, minimize

from lvpm.continuous import (Bernoulli, Categorical, LatentFactorModel, Normal,
                             factor_analysis_expected_score, factor_analysis_info,
                             factor_analysis_score)

A3, B3 = np.array([1.5, 1.0, 2.0]), np.array([0.0, -0.5, 0.5])


def irt(a, b, link="logit", **kw):
    return LatentFactorModel([Bernoulli(a[i], b[i], link) for i in range(len(b))], **kw)


def correctness():
    rng = np.random.default_rng(0)
    A, b, psi = np.array([[0.9], [0.6], [0.3]]), np.array([0.1, -0.2, 0.3]), np.array([0.3, 0.5, 0.8])
    m = LatentFactorModel([Normal(A[i], b[i], psi[i]) for i in range(3)])
    Y = m.sample(500, rng)
    err = max(np.abs(m.scores(Y, t) - factor_analysis_score(A, b, psi, Y, t)).max() for t in (1.0, 0.6))
    print(f"[1] quadrature vs closed-form tempered score (factor analysis, tau in {{1, .6}}): {err:.0e} bits")


def recovery():
    print("[2] Recovery at tau = 1 from random starts (exact pattern tables / large samples)")
    a, b = np.array([1.5, 1.0, 2.0, 0.8, 1.2, 1.7]), np.array([0, -.5, .5, 1, -1, .2])
    for link in ("logit", "probit"):
        P, p = irt(a, b, link).patterns()
        errs = []
        for s in range(5):
            r = np.random.default_rng(s)
            m = irt(r.normal(0, 1, 6), r.normal(0, 1, 6), link)
            m.fit(P, p, iters=20000, tol=1e-15)
            th = np.array([ind.theta for ind in m.indicators])
            th[:, 0] *= np.sign(th[0, 0])  # the k = 1 gauge: z -> -z
            errs.append(np.abs(th - np.stack([a, b], 1)).max())
        print(f"    1-factor {link} IRT, 6 items, 5 starts: max |param - truth| (up to sign) {max(errs):.0e}")

    # two factors: loadings are identified only up to rotation; compare A A^T and b
    rng = np.random.default_rng(1)
    A2, b2 = rng.normal(0, 1.2, (8, 2)), rng.normal(0, 0.7, 8)
    P, p = LatentFactorModel([Bernoulli(A2[i], b2[i]) for i in range(8)], k=2).patterns()
    errs = []
    for s in range(3):
        r = np.random.default_rng(10 + s)
        m = LatentFactorModel([Bernoulli(r.normal(0, 1, 2), 0.0) for _ in range(8)], k=2)
        m.fit(P, p, iters=20000, tol=1e-15)
        A, b = (np.array([ind.theta[:2] for ind in m.indicators]),
                np.array([ind.theta[2] for ind in m.indicators]))
        errs.append(max(np.abs(A @ A.T - A2 @ A2.T).max(), np.abs(b - b2).max()))
    print(f"    2-factor IRT, 8 items, 3 starts: max error in rotation-invariant A A^T and b {max(errs):.0e}")

    # mixed indicators from a sample
    truth = LatentFactorModel([Bernoulli(1.2, 0.3), Bernoulli(-0.8, -0.5, "probit"),
                               Normal(0.9, 1.0, 0.5), Categorical([[1.0], [2.0]], [0.5, -0.5])])
    Y = truth.sample(150000, np.random.default_rng(3))
    m = LatentFactorModel([Bernoulli(0.5, 0.0), Bernoulli(0.5, 0.0, "probit"),
                           Normal(0.5, 0.0, 1.0), Categorical([[0.1], [0.2]], [0.0, 0.0])])
    m.fit(Y, iters=3000, tol=1e-12)
    flip = np.sign(m.indicators[0].theta[0])
    err = 0.0
    for t, f in zip(truth.indicators, m.indicators):
        ft = f.theta.copy()
        if ft.ndim == 1:
            ft[:-1] *= flip
        else:
            ft[:, :-1] *= flip
        err = max(err, np.abs(ft - t.theta).max())
    print(f"    mixed (logit, probit, normal, 3-category), 150k samples: max |param - truth| {err:.3f} "
          f"(normal variance {m.indicators[2].psi:.3f} vs 0.5)")


def gauge():
    rng = np.random.default_rng(4)
    A2, b2 = rng.normal(0, 1.2, (6, 2)), rng.normal(0, 0.7, 6)
    m = LatentFactorModel([Bernoulli(A2[i], b2[i]) for i in range(6)], k=2, m=41)
    P, p = m.patterns()
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    _, p_rot = m.rotated(R).patterns()
    _, p_flip = irt(A3, B3).rotated(-np.eye(1)).patterns()
    _, p3 = irt(A3, B3).patterns()
    print("[3] Gauge vs genuine non-identification")
    print(f"    gauge: rotate a 2-factor IRT model by 0.7 rad -> max change in any pattern probability "
          f"{np.abs(p_rot - p).max():.0e};  sign flip of a 1-factor model -> {np.abs(p_flip - p3).max():.0e}")

    rho = 0.5
    S = np.array([[1, rho], [rho, 1]])
    y = np.array([1.0, 1.0])
    rows = []
    for a1 in (0.6, np.sqrt(rho), 0.9):
        a = np.array([a1, rho / a1])
        psi = 1 - a ** 2
        sc = factor_analysis_expected_score(a[:, None], np.zeros(2), psi, np.zeros(2), S)
        post_mean = a @ np.linalg.solve(np.outer(a, a) + np.diag(psi), y)
        rows.append((a, sc, factor_analysis_info(a[:, None], psi), post_mean))
    print(f"    genuine: 2 Gaussian indicators (rho = {rho}), loadings with a1*a2 = rho all fit exactly:")
    for a, sc, inf, pm in rows:
        print(f"      a = ({a[0]:.3f}, {a[1]:.3f}):  E[log score] {sc:.6f}  I(Z;Y) {inf:.3f}  "
              f"E[Z | y = (1,1)] {pm:.3f}")


def gaussian_wyner():
    rho = 0.5
    S = np.array([[1, rho], [rho, 1]])
    wyner = 0.5 * np.log2((1 + rho) / (1 - rho))

    def coupling_rate(a, psi, tau):  # I(Z;Y) with Y ~ N(0, S) and the tempered posterior
        pi = a / psi
        V = 1 / (1 + tau * a @ pi)
        m = V * tau * pi
        return 0.5 * np.log2((V + m @ S @ m) / V)

    print(f"[4] Gaussian indicators (rho = {rho}): tempered-score optimum vs Wyner's "
          f"1/2 log((1+rho)/(1-rho)) = {wyner:.4f} bits")
    for tau in (0.9, 0.99, 0.999, 0.9999):
        best = None
        for s in range(10):
            r = np.random.default_rng(s)
            x0 = np.r_[r.uniform(0.1, 1, 2), np.log(r.uniform(0.2, 1, 2))]
            f = lambda x: -factor_analysis_expected_score(x[:2, None], np.zeros(2), np.exp(x[2:]),
                                                         np.zeros(2), S, tau)
            res = minimize(f, x0, method="BFGS", options=dict(gtol=1e-12))
            best = res if best is None or res.fun < best.fun else best
        a, psi = np.abs(best.x[:2]), np.exp(best.x[2:])
        print(f"    tau={tau}: loadings ({a[0]:.4f}, {a[1]:.4f})  I(Z;Y) {coupling_rate(a, psi, tau):.4f}")


def free_prior():
    """Free-form latent distribution with 3 binary IRT indicators."""
    P, p = irt(A3, B3).patterns()
    zg = np.linspace(-4, 4, 161)
    base = -zg ** 2 / 2

    def profile_hi(m):  # gauge-invariant: P(the latent state makes Y1 likely)
        return np.exp(m.logw)[np.exp(m.indicators[0].logprob_table(m.Z)[:, 1]) > 0.8].sum()

    print("[5] Free-form latent distribution, 3 binary indicators, 3 random starts each")
    print("    columns: E[S_tau] | I(Z;Y) | P(P(Y1=1|Z) > 0.8) | forecast error TV(m, P(Y))")
    for links in ("fixed", "free"):
        for tau in (1.0, 0.95, 0.8, 0.5):
            out = []
            for s in range(3):
                r = np.random.default_rng(s)
                m = irt(A3, B3, prior="free", grid=(zg[:, None], base + r.normal(0, 2, len(zg))))
                tr = m.fit(P, p, tau, iters=20000, tol=1e-14, fit_indicators=links == "free")
                _, pm = m.patterns()
                out.append((tr[-1], m.info(), profile_hi(m), 0.5 * np.abs(pm - p).sum()))
            o = np.array(out)
            print(f"    links {links:<5} tau={tau:<4}  " + "  ".join(
                f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(o.min(0), o.max(0))))

    # exact range of a latent belief over all maximisers, links fixed
    m = irt(A3, B3, prior="free", grid=(zg[:, None], base))
    lik = np.exp(m.loglik(P))  # (patterns, grid)
    event = (zg > 1).astype(float)
    print("    links fixed: exact range of P(Z > 1) over ALL latent distributions maximising E[S_tau]:")
    for tau in (1.0, 0.95, 0.8):
        best = m.copy()
        best.fit(P, p, tau, iters=50000, tol=1e-15, fit_indicators=False)
        M = lik ** tau
        v = M @ np.exp(best.logw)
        A_eq, b_eq = np.vstack([M, np.ones(len(zg))]), np.r_[v, 1.0]
        lo, hi = (linprog(sg * event, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
                  for sg in (1, -1))
        assert lo.status == 0 and hi.status == 0
        print(f"      tau={tau:<4}  [{lo.fun:.3f}, {-hi.fun:.3f}]")


if __name__ == "__main__":
    correctness()
    recovery()
    gauge()
    gaussian_wyner()
    free_prior()
