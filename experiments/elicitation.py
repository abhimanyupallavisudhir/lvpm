"""Checks for the elicitation / algorithm / LVPM sections of notes/latent-meaning.md.

Run:  PYTHONPATH=. .venv/bin/python experiments/elicitation.py
"""

import itertools

import numpy as np
from scipy.optimize import linprog

from lvpm.dag import LatentDAG
from lvpm.latent_bottleneck import bottleneck_curve, entropy, model_joint, stats


def bern(v):
    v = np.asarray(v, dtype=float)
    return np.stack([1 - v, v], -1)


def star(k, n):
    ys = [f"Y{i + 1}" for i in range(n)]
    return LatentDAG({"X": k, **{y: 2 for y in ys}}, {y: ["X"] for y in ys}, ys)


def identity():
    """max_theta E S_tau(theta) = -min_X L_tau(X) - tau * sum_i H(Y_i)."""
    py = model_joint(np.array([.4, .6]), [bern([.2, .85]), bern([.3, .9]), bern([.1, .7])]).sum(0)
    h_views = sum(entropy(py.sum(axis=tuple(j for j in range(3) if j != i))) for i in range(3))
    rng = np.random.default_rng(0)
    gap = 0.0
    for k, tau in itertools.product((2, 3), (0.5, 0.8, 0.95)):
        d = star(k, 3)
        es = d.score(d.fit(py, tau, rng, restarts=20), py, tau)
        q = bottleneck_curve(py, k, [tau], restarts=20, seed=1)[0]["q"]
        s = stats(q * py)
        gap = max(gap, abs(es + s["rate"] - tau * sum(s["i_views"]) + tau * h_views))
    print("[1] max E[S_tau] vs -min L_tau - tau*sum H(Y_i), |X| in {2,3}, tau in {.5,.8,.95}: "
          f"max discrepancy {gap:.1e} bits")

    # General DAG: at the optimum, E[S_tau] = -[(1-tau) I(H;Y) + D_H + tau D_Y + tau H(Y)].
    d = LatentDAG({"X": 2, "Z1": 2, "Z2": 2, **{f"Y{i}": 2 for i in range(1, 5)}},
                  {"Z1": ["X"], "Z2": ["X"], "Y1": ["Z1"], "Y2": ["Z1"], "Y3": ["Z2"], "Y4": ["Z2"]},
                  ["Y1", "Y2", "Y3", "Y4"])
    py = d.predict(d.random(rng))
    gap = 0.0
    for tau in (0.6, 0.9):
        c = d.fit(py, tau, rng, restarts=10, iters=20000, tol=1e-14)
        inf = d.info(d.coupling(c, py, tau))
        rhs = (1 - tau) * inf["rate"] + inf["d_hidden"] + tau * inf["d_observed"] + tau * entropy(py)
        gap = max(gap, abs(d.score(c, py, tau) + rhs))
    print(f"    same identity for a two-level latent DAG, tau in {{.6,.9}}: max discrepancy {gap:.1e} bits")


def incentives():
    """Which latent model does each scoring rule pay most for? Three binary views,
    true latent has 2 states; a 3-state latent explains them with fewer bits."""
    pi = np.array([.122, .878])
    cs = [bern([.104, .985]), bern([.264, .102]), bern([.675, .48])]
    py = model_joint(pi, cs).sum(0)
    rng = np.random.default_rng(1)
    d2, d3 = star(2, 3), star(3, 3)
    true = {"X": pi, "Y1": cs[0], "Y2": cs[1], "Y3": cs[2]}
    taus = [1.0, 0.999, 0.99, 0.9]
    reports = {"true 2-state latent": (d2, true)}
    for t in taus[1:]:
        reports[f"optimum at tau={t}"] = (d3, d3.fit(py, t, rng, restarts=30))
    print("[2] Expected tempered score of each report (bits, relative to the best report per column)")
    print("    " + " " * 26 + "".join(f"S_{t:<9}" for t in taus))
    table = {name: [d.score(c, py, t) for t in taus] for name, (d, c) in reports.items()}
    best = [max(v[j] for v in table.values()) for j in range(len(taus))]
    for name, v in table.items():
        print(f"    {name:<26}" + "".join(f"{x - b:+.5f}   " for x, b in zip(v, best)))
    for name, (d, c) in reports.items():
        j = d.joint(c)
        inf = d.info(j)
        fit = d.score(c, py, 1.0) + entropy(py)
        print(f"    {name:<26} I(X;Y) {inf['rate']:.4f}  log-score regret on P(Y) {max(-fit, 0):.1e}")


def hierarchy():
    """X -> Z1, Z2, Z3;  Zj -> two binary indicators each. Recover all tables."""
    card = {"X": 2, "Z1": 2, "Z2": 2, "Z3": 2, **{f"Y{i}": 2 for i in range(1, 7)}}
    parents = {"Z1": ["X"], "Z2": ["X"], "Z3": ["X"],
               "Y1": ["Z1"], "Y2": ["Z1"], "Y3": ["Z2"], "Y4": ["Z2"], "Y5": ["Z3"], "Y6": ["Z3"]}
    obs = [f"Y{i}" for i in range(1, 7)]
    d = LatentDAG(card, parents, obs)
    truth = {"X": np.array([.35, .65]),
             "Z1": bern([.15, .8]), "Z2": bern([.25, .9]), "Z3": bern([.7, .1]),
             "Y1": bern([.1, .85]), "Y2": bern([.2, .75]), "Y3": bern([.15, .9]),
             "Y4": bern([.3, .8]), "Y5": bern([.05, .7]), "Y6": bern([.25, .95])}
    py = d.predict(truth)
    rng = np.random.default_rng(2)

    def distance(cpt):
        """max |param - truth| minimised over relabellings of each hidden node."""
        best = np.inf
        for flips in itertools.product((0, 1), repeat=4):
            f = dict(zip(["X", "Z1", "Z2", "Z3"], flips))
            err = 0.0
            for v in d.nodes:
                t = cpt[v]
                for ax, u in enumerate(d.parents[v] + [v]):
                    if f.get(u):
                        t = np.flip(t, ax)
                err = max(err, np.abs(t - truth[v]).max())
            best = min(best, err)
        return best

    fits = []
    for _ in range(20):
        c = d.fit(py, 1.0, rng, restarts=1, iters=20000, tol=1e-14)
        fits.append((d.score(c, py, 1.0), distance(c)))
    top = max(s for s, _ in fits)
    good = [e for s, e in fits if s > top - 1e-7]
    print("[3] Hierarchical DAG (X -> Z1,Z2,Z3 -> two indicators each), tempered EM at tau=1, 20 starts")
    print(f"    {len(good)}/20 reach the best score; their max |param - truth| (up to relabelling): "
          f"{max(good):.1e}")
    for tau in (0.95, 0.8):
        c = d.fit(py, tau, rng, restarts=10)
        inf = d.info(d.joint(c))
        print(f"    tau={tau}: I(H;Y) {inf['rate']:.4f} (truth {d.info(d.joint(truth))['rate']:.4f}),  "
              f"log-score regret on P(Y) {-(d.score(c, py, 1.0) + entropy(py)):.4f}")


def unpriced():
    """IRT market (Gaussian latent, logistic links): how far can a belief about
    the latent move while the indicators' joint, hence every payout, stays fixed?"""
    y = np.linspace(-4, 4, 401)
    prior = np.exp(-y ** 2 / 2)
    prior /= prior.sum()
    rng = np.random.default_rng(3)
    print("[4] IRT market, Gaussian P(Y): range of P(Y > 1) (Gaussian value "
          f"{prior[y > 1].sum():.3f}) over all latent distributions")
    print("    giving the identical indicator joint, i.e. identical payouts to every trader:")
    a_all, b_all = rng.uniform(0.8, 2.5, 10), rng.uniform(-1.5, 1.5, 10)
    for n in (3, 6, 10):
        links = 1 / (1 + np.exp(-(a_all[:n, None] * y[None] + b_all[:n, None])))
        L = np.ones((2 ** n, len(y)))
        for row, xs in enumerate(itertools.product((0, 1), repeat=n)):
            for i, x in enumerate(xs):
                L[row] *= links[i] if x else 1 - links[i]
        target = L @ prior
        event = (y > 1).astype(float)
        lo, hi = (linprog(sgn * event, A_eq=L, b_eq=target, bounds=(0, None), method="highs")
                  for sgn in (1, -1))
        assert lo.status == 0 and hi.status == 0, (lo.message, hi.message)
        print(f"    {n:>2} indicators: [{lo.fun:.3f}, {-hi.fun:.3f}]")


if __name__ == "__main__":
    identity()
    incentives()
    hierarchy()
    unpriced()
