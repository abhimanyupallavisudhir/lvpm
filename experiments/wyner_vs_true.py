"""Is the Wyner-minimal latent (min I(X;Y) over exact CI models, |X| <= r+2)
the true latent? Exact equality-constrained solve (SLSQP).

Part 1: random models, 2 vs 3 views.  Part 2: two 3-binary-view models where a
3-state latent beats the true 2-state one, with and without a 4th view.

Run:  PYTHONPATH=. .venv/bin/python experiments/wyner_vs_true.py
"""
import numpy as np
from scipy.optimize import minimize

from lvpm.latent_bottleneck import model_joint, stats


def unpack(z, k, dims):
    def sm(v):
        e = np.exp(v - v.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    pi, i = sm(z[:k]), k
    conds = []
    for d in dims:
        conds.append(sm(z[i:i + k * d].reshape(k, d)))
        i += k * d
    return pi, conds


def wyner_exact(py, k, rng, starts=40, init=None):
    dims = py.shape
    rate = lambda z: stats(model_joint(*unpack(z, k, dims)))["rate"]
    cons = {"type": "eq",
            "fun": lambda z: (model_joint(*unpack(z, k, dims)).sum(0) - py).ravel()[1:]}
    n = k + k * sum(dims)
    best = None
    inits = [rng.normal(0, 2, n) for _ in range(starts)] + ([init] if init is not None else [])
    for z0 in inits:
        r = minimize(rate, z0, method="SLSQP", constraints=[cons],
                     options=dict(maxiter=2000, ftol=1e-13))
        viol = np.abs(cons["fun"](r.x)).max()
        if viol < 1e-9 and (best is None or r.fun < best[0]):
            best = (r.fun, r.x)
    return best


def split_init(pi, conds, k):
    """Logits of the true model with its components duplicated up to k states."""
    r = len(pi)
    idx = [x % r for x in range(k)]
    counts = np.bincount(idx, minlength=r)
    pis = np.array([pi[i] / counts[i] for i in idx])
    z = [np.log(pis)] + [np.log(np.maximum(c[idx], 1e-12)).ravel() for c in conds]
    return np.concatenate(z)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rows = []
    for trial in range(10):
        r, d = [(2, 2), (2, 3), (3, 3)][trial % 3]
        pi = rng.dirichlet(np.ones(r) * 2)
        conds = [rng.dirichlet(np.ones(d), size=r) for _ in range(3)]
        for nv in (2, 3):
            cs = conds[:nv]
            py = model_joint(pi, cs).sum(0)
            true = stats(model_joint(pi, cs))["rate"]
            w = min(wyner_exact(py, k, rng, starts=15, init=split_init(pi, cs, k))[0]
                    for k in (r, r + 1, r + 2))
            rows.append((trial, r, d, nv, true, w))
            print(f"trial {trial}  |X|={r} |Y_i|={d}  {nv} views:  true I(X;Y) {true:.5f}"
                  f"   Wyner-minimal {w:.5f}   gap {true - w:+.5f}", flush=True)
    for nv in (2, 3):
        gaps = [t - w for (_, _, _, v, t, w) in rows if v == nv]
        print(f"{nv} views: gap true - Wyner  min {min(gaps):+.5f}  max {max(gaps):+.5f}")

    def bern(v):
        v = np.asarray(v)
        return np.stack([1 - v, v], 1)

    print("Part 2: Wyner-minimal I(X;Y) at |X| = 2, 3, 4 (true |X| = 2)")
    cases = {"A": (np.array([.122, .878]), [bern([.104, .985]), bern([.264, .102]), bern([.675, .48])]),
             "B": (np.array([.425, .575]), [bern([.944, .689]), bern([.93, .939]), bern([.183, .111])])}
    for name, (pi, cs) in cases.items():
        for extra in ([], [bern([.3, .75])]):
            c = cs + extra
            py = model_joint(pi, c).sum(0)
            true = stats(model_joint(pi, c))["rate"]
            w = [wyner_exact(py, k, rng, starts=8, init=split_init(pi, c, k))[0] for k in (2, 3, 4)]
            print(f"model {name}, {len(c)} binary views:  true {true:.5f}   "
                  + "   ".join(f"|X|={k}: {v:.5f}" for k, v in zip((2, 3, 4), w)), flush=True)
