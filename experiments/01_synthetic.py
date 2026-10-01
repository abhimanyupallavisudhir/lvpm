"""Synthetic checks of the two claims the proposal rests on.

A. Price co-movement identifies the joint.  In a world where a 1-factor threshold model
   generates outcomes and traders receive Gaussian news about the factor and about each
   question's idiosyncratic part, the efficient price path p_t = P(X_i | news_t) is a
   martingale.  We check E[sum_t dp_i dp_j] = Cov(X_i, X_j), and how well pre-t0
   co-movement recovers the time-t0 dependence structure (scored on resolved outcomes).

B. How much signal is there?  Distribution of the per-cluster structure score
   log P(x | p, R_hat) - log P(x | p, I) for forecasters with different structure
   knowledge, and the number of resolved clusters needed to tell them apart.

Writes reports/figures/synthetic_*.png and reports/synthetic.json."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvpm import covariation as C  # noqa: E402
from lvpm import model as M  # noqa: E402
from lvpm.plot import INK2, SERIES, plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
res = {}


# ---------------------------------------------------------------- A: belief paths
def simulate_world(lam, tau, T, t0, a, b, rng):
    """Daily news: precision a about the factor Y, b about each idiosyncratic eps_i."""
    n = len(lam)
    u = np.sqrt(1 - lam ** 2)
    Y, eps = rng.standard_normal(), rng.standard_normal(n)
    sY = np.cumsum(Y + rng.standard_normal(T) / np.sqrt(a)) * a  # precision-weighted sums
    sE = np.cumsum(eps + rng.standard_normal((T, n)) / np.sqrt(b), 0) * b
    t = np.arange(1, T + 1)[:, None]
    mY, vY = sY[:, None] / (1 + a * t), 1 / (1 + a * t)
    mE, vE = sE / (1 + b * t), 1 / (1 + b * t)
    mean = lam * mY + u * mE
    var = lam ** 2 * vY + u ** 2 * vE
    P = ndtr((tau + mean) / np.sqrt(var))
    P = np.vstack([ndtr(tau), P])  # day 0 = prior
    X = (lam * Y + u * eps > -tau).astype(float)
    # time-t0 conditional structure: latent corr of Z given news
    V = var[t0 - 1]
    Rt0 = np.outer(lam, lam) * vY[t0 - 1, 0] / np.sqrt(np.outer(V, V))
    np.fill_diagonal(Rt0, 1)
    return P, X, Rt0


n, T, t0, W = 8, 60, 30, 1500
lam = rng.uniform(0.3, 0.85, n) * np.where(rng.random(n) < 0.75, 1, -1)
tau = rng.normal(0, 0.7, n)
iu = np.triu_indices(n, 1)
A = {}
for regime, (a, b) in {"balanced news": (0.04, 0.04), "factor-heavy news": (0.12, 0.02)}.items():
    qv, prod, implied, sc = [], [], [], {k: [] for k in ["true", "prob-scale", "probit-scale", "independent"]}
    for w in range(W):
        P, X, Rt0 = simulate_world(lam, tau, T, t0, a, b, rng)
        p0 = P[t0]
        path = np.vstack([P[t0:], X])  # include the resolution jump
        qv.append(C.realised_covariation(path)[iu])
        prod.append(np.outer(X - p0, X - p0)[iu])
        implied.append([M.p11_from_rho(p0[i], p0[j], Rt0[i, j]) - p0[i] * p0[j] for i, j in zip(*iu)])
        # forecasts of the t0 structure from the pre-t0 window only
        lam_t0 = M.fit_factor(Rt0, 1)[:, 0]
        r_prob = C.increment_corr(P[: t0 + 1], probit=False, shrink=0)
        r_pbt = C.increment_corr(P[: t0 + 1], probit=True, shrink=0)
        L_prob = M.fit_factor(C.latent_corr_from_outcome_corr(r_prob, p0), 1)[:, 0]
        L_pbt = M.fit_factor(M.nearest_correlation(r_pbt), 1)[:, 0]
        base = M.independent_loglik(X, p0)
        sc["true"].append(M.factor_loglik(X, p0, lam_t0) - base)
        sc["prob-scale"].append(M.factor_loglik(X, p0, L_prob) - base)
        sc["probit-scale"].append(M.factor_loglik(X, p0, L_pbt) - base)
        sc["independent"].append(0.0)
    qv, prod, implied = map(lambda z: np.mean(z, 0), (qv, prod, implied))
    A[regime] = dict(qv=qv, prod=prod, implied=implied,
                     scores={k: (float(np.mean(v)), float(np.std(v) / np.sqrt(W))) for k, v in sc.items()})

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, (regime, d) in zip(axes, A.items()):
    lim = max(np.abs(d["implied"]).max(), np.abs(d["qv"]).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls=":")
    ax.scatter(d["implied"], d["qv"], s=22, color=SERIES[0], label=r"$E[\sum_t \Delta p_i\Delta p_j]$ (price paths)",
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(d["implied"], d["prod"], s=22, color=SERIES[1], marker="s", label=r"$E[(X_i-p_i)(X_j-p_j)]$ (outcomes)",
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_title(regime)
    ax.set_xlabel(r"model $\mathrm{Cov}_{t_0}(X_i, X_j)$")
axes[0].set_ylabel(f"average over {W} simulated worlds")
axes[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("Martingale identity: price co-movement recovers outcome covariance", fontsize=11, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT / "figures" / "synthetic_identity.png")
res["A"] = {r: {"scores": d["scores"],
                "slope_qv": float(np.polyfit(d["implied"], d["qv"], 1)[0]),
                "slope_outcomes": float(np.polyfit(d["implied"], d["prod"], 1)[0])} for r, d in A.items()}

# ---------------------------------------------------------------- B: power analysis
def forecasters(lam, rng):
    k = len(lam)
    return {
        "oracle": lam,
        "right signs, half strength": 0.5 * lam,
        "noisy (sd 0.25)": np.clip(lam + rng.normal(0, 0.25, k), -0.95, 0.95),
        "uniform +0.5 (knows 'related')": np.full(k, 0.5) * np.sign(lam.mean()),
        "half the signs wrong": lam * np.where(rng.random(k) < 0.5, -1, 1),
    }


B = {}
for nq in (4, 8, 16):
    S = {}
    for c in range(3000):
        lamc = rng.uniform(0.3, 0.85, nq) * np.where(rng.random(nq) < 0.75, 1, -1)
        p = ndtr(rng.normal(0, 0.8, nq))
        x = M.sample(p, M.factor_corr(lamc), 1, rng)[0]
        base = M.independent_loglik(x, p)
        for name, L in forecasters(lamc, rng).items():
            S.setdefault(name, []).append(M.factor_loglik(x, p, L) - base)
    B[nq] = {k: np.array(v) for k, v in S.items()}

res["B"] = {}
for nq, S in B.items():
    o = S["oracle"]
    res["B"][nq] = {}
    for k, v in S.items():
        d = o - v if k != "oracle" else o
        m, sd = d.mean(), d.std()
        res["B"][nq][k] = {"mean": float(v.mean()), "sd": float(v.std()),
                           # clusters for a 2-sigma (one-sided ~97.7%) detection at 80% power
                           "clusters_needed": float(((2 + 0.84) * sd / m) ** 2) if m > 0 else None}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
names = list(B[8].keys())
ax = axes[0]
for i, k in enumerate(names):
    v = B[8][k]
    q1, q3 = np.percentile(v, [25, 75])
    ax.plot([q1, q3], [i, i], color=SERIES[i], lw=2.5, solid_capstyle="round")
    ax.plot(v.mean(), i, "o", ms=8, color=SERIES[i], mec="white", mew=1)
ax.axvline(0, color=INK2, lw=1, ls=":")
ax.set_yticks(range(len(names)), names)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("structure score per cluster (nats): mean, IQR")
ax.set_title("8 questions per cluster")
ax = axes[1]
for i, k in enumerate(names):
    ys = [res["B"][nq][k]["clusters_needed"] for nq in B]
    ax.plot(list(B), ys, marker="o", ms=6, color=SERIES[i], label=k if i else "oracle vs independence")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(list(B), [str(z) for z in B])
ax.set_xlabel("questions per cluster")
ax.set_ylabel("resolved clusters needed")
ax.set_title("Clusters to distinguish from oracle")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figures" / "synthetic_power.png")

(OUT / "synthetic.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
