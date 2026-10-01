"""Score every dependence-structure source on the resolved pilot clusters.

Marginals are fixed to the Polymarket price at t0 for every method ("copula swap"), so
all differences are differences in *structure*:

    S = log P(x | p_mkt, R_method) - log P(x | p_mkt, I)        (nats, per cluster)

Methods
  co-movement   correlation of probit-price increments over the 30 days before t0
  LLM latent    protocol A: LLM-named latent + P(YES | Y = +-1)  -> 1-factor loadings
  LLM label     protocol C: a fresh LLM given only A's label     -> 1-factor loadings
  LLM pairwise  protocol B: P(j | i), P(j | not i) for all pairs -> tetrachoric R
  hindsight     (diagnostic, not a forecast) co-movement from t0 to just before
                resolution - what the market's own belief paths imply

Writes reports/pilot.json and reports/figures/pilot_*.png."""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvpm import covariation as C  # noqa: E402
from lvpm import model as M  # noqa: E402
from lvpm import scoring as S  # noqa: E402
from lvpm.plot import INK2, MUTED, SERIES, plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "data" / "pilot_clusters.json").read_text())
RESP = ROOT / "data" / "elicitation" / "responses"
T0, DAY = D["meta"]["t0"], C.DAY
METHODS = ["co-movement", "LLM latent", "LLM label", "LLM pairwise", "hindsight"]
FORECASTS = METHODS[:4]


def ts(s):
    return int(datetime.fromisoformat(s.replace(" ", "T").replace("+00", "+00:00")).timestamp())


def loadings(p, r):
    return np.array([M.loading_from_conditionals(pi, h, l) for pi, h, l in zip(p, r["p_hi"], r["p_lo"])])


def pairwise_R(p, r):
    n = len(p)
    R = np.eye(n)
    for i, j, q1, q0 in r["pairs"]:
        R[i, j] = R[j, i] = M.rho_from_conditionals(p[i], p[j], q1, q0)
    return M.nearest_correlation(R)


def comovement_R(qs, start, end, cut_resolution=False):
    G = C.daily_grid([q["history"] for q in qs], start, end)
    if cut_resolution:  # drop each market's last day so the resolution jump is excluded
        days = np.arange(start, end + 1, DAY)
        for j, q in enumerate(qs):
            G[days >= ts(q["closed_time"]) - DAY, j] = np.nan
    return M.nearest_correlation(C.increment_corr(G, probit=True, shrink=10))


rows, pairs, extra = {}, {m: [] for m in METHODS}, {}
for c, qs in D["clusters"].items():
    p = np.array([q["p_t0"] for q in qs])
    x = np.array([q["outcome"] for q in qs], float)
    A = json.loads((RESP / f"A1_{c}.json").read_text())
    B = json.loads((RESP / f"B_{c}.json").read_text())
    Cl = json.loads((RESP / f"C_{c}.json").read_text())
    lam_A, lam_C = loadings(p, A), loadings(p, Cl)
    Rs = {
        "co-movement": comovement_R(qs, T0 - D["meta"]["pre_days"] * DAY, T0),
        "LLM latent": M.factor_corr(lam_A),
        "LLM label": M.factor_corr(lam_C),
        "LLM pairwise": pairwise_R(p, B),
        "hindsight": comovement_R(qs, T0, max(ts(q["closed_time"]) for q in qs), cut_resolution=True),
    }
    score = {m: S.structure_score(x, p, L=lam_A) if m == "LLM latent" else
             S.structure_score(x, p, L=lam_C) if m == "LLM label" else
             S.structure_score(x, p, R=R) for m, R in Rs.items()}
    rows[c] = dict(n=len(qs), yes=int(x.sum()), label=A["label"], scores=score,
                   mean_abs_rho={m: float(np.abs(R[np.triu_indices(len(p), 1)]).mean()) for m, R in Rs.items()},
                   independent_logloss=float(-M.independent_loglik(x, p)))
    for m, R in Rs.items():
        pairs[m].append(S.covariance_pairs(x, p, R))
    iu = np.triu_indices(len(p), 1)
    extra[c] = dict(
        latent_vs_label_loading_corr=float(np.corrcoef(lam_A, lam_C)[0, 1]),
        latent_vs_label_sign_agreement=float(np.mean(np.sign(lam_A[np.abs(lam_A) > .05]) ==
                                                     np.sign(lam_C[np.abs(lam_A) > .05]))),
        latent_vs_pairwise_rho_corr=float(np.corrcoef(Rs["LLM latent"][iu], Rs["LLM pairwise"][iu])[0, 1]),
        comovement_vs_hindsight_rho_corr=float(np.corrcoef(Rs["co-movement"][iu], Rs["hindsight"][iu])[0, 1]),
        loadings_A=lam_A.round(3).tolist(), loadings_C=lam_C.round(3).tolist())

summary = {}
for m in METHODS:
    v = [rows[c]["scores"][m] for c in rows]
    tot = float(np.sum(v))
    mean, lo, hi = S.bootstrap_mean(v)
    beta, se = S.calibration_slope(pairs[m])
    summary[m] = dict(total=tot, per_cluster_mean=mean, boot95=[lo, hi], clusters_positive=int(np.sum(np.array(v) > 0)),
                      cov_slope=float(beta), cov_slope_se=float(se))

out = dict(t0=D["meta"]["t0_iso"], clusters=rows, summary=summary, agreement=extra)
(ROOT / "reports" / "pilot.json").write_text(json.dumps(out, indent=2))

# ---------------- figure 1: structure score per cluster and in total
cl = sorted(rows, key=lambda c: rows[c]["scores"]["LLM latent"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
ax = axes[0]
off = np.linspace(-0.28, 0.28, len(METHODS))
for k, m in enumerate(METHODS):
    ys = np.arange(len(cl)) + off[k]
    ax.scatter([rows[c]["scores"][m] for c in cl], ys, s=34, color=SERIES[k] if m != "hindsight" else MUTED,
               marker="o" if m != "hindsight" else "D", edgecolor="white", linewidth=0.8, zorder=3, label=m)
ax.axvline(0, color=INK2, lw=1, ls=":")
ax.set_yticks(range(len(cl)), [c.replace("_", " ") for c in cl])
ax.set_xlabel("structure score  log P(x | R) − log P(x | independent)   [nats]")
ax.set_title("Per cluster (marginals fixed at market price, 2026-07-01)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", visible=False)
ax = axes[1]
for k, m in enumerate(METHODS):
    s = summary[m]
    col = SERIES[k] if m != "hindsight" else MUTED
    ax.errorbar(s["per_cluster_mean"], k, xerr=[[s["per_cluster_mean"] - s["boot95"][0]], [s["boot95"][1] - s["per_cluster_mean"]]],
                fmt="o" if m != "hindsight" else "D", color=col, ms=7, capsize=0, lw=2)
ax.axvline(0, color=INK2, lw=1, ls=":")
ax.set_yticks(range(len(METHODS)), METHODS)
ax.invert_yaxis()
ax.set_xlabel("mean per cluster, bootstrap 95% CI")
ax.set_title(f"All {len(rows)} clusters")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(ROOT / "reports" / "figures" / "pilot_scores.png")

# ---------------- figure 2: covariance calibration slope (1 = calibrated, 0 = no information)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
for k, m in enumerate(FORECASTS):
    b, se = summary[m]["cov_slope"], summary[m]["cov_slope_se"]
    ax.errorbar(b, k, xerr=1.96 * se, fmt="o", color=SERIES[k], ms=7, lw=2, capsize=0)
ax.axvline(1, color=INK2, lw=1, ls=":")
ax.axvline(0, color=MUTED, lw=1, ls=":")
ax.set_yticks(range(len(FORECASTS)), FORECASTS)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel(r"slope of realised $(x_i-p_i)(x_j-p_j)$ on predicted Cov  (95% CI)")
ax.set_title("Pairwise covariance calibration (1 = calibrated)")
fig.tight_layout()
fig.savefig(ROOT / "reports" / "figures" / "pilot_calibration.png")

print(json.dumps(summary, indent=1))
for c in rows:
    print(c, rows[c]["yes"], "/", rows[c]["n"], {m: round(v, 2) for m, v in rows[c]["scores"].items()})
print(json.dumps({c: {k: v for k, v in e.items() if not k.startswith("loadings")} for c, e in extra.items()}, indent=1))
