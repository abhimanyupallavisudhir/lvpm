"""Analysis of the essay arena under best-of-n (experiments/arena_bon.py): which reward selects better reports?

For every topic, reward and evaluation: the exact best-of-k value (order statistics over the sampled reports) minus
the mean over all reports - the gain from selecting on that reward. Aggregated over topics with bootstrap CIs over
topics, separately for planted worlds (truth known) and testbed families. Also: the per-topic Spearman correlation
of each reward with each evaluation, and, for planted worlds, the share of the gap from the average report to the
true theory that selection closes.

``oracle`` is the report's score on the held-out test questions themselves, so its curve on ``eval_test`` is an
in-sample upper bound; on the other evaluations it is a fair comparison.

    PYTHONPATH=. python experiments/arena_report.py --dir results/arena-bon
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lvpm.essays.arena import best_of_k

REWARDS = ["latent", "judge", "debate", "oracle"]
EVALS = {"eval_test": "held-out questions", "eval_shift": "shifted questions", "eval_reader_b": "held-out reader",
         "eval_rating": "expert rubric (1-10)"}
COLOURS = {"latent": "#2a78d6", "judge": "#eb6834", "debate": "#1baf7a", "oracle": "#eda100"}
NAMES = {"latent": "latent S_τ", "judge": "LLM judge", "debate": "debate", "oracle": "oracle (test S)"}


def gains(df, kmax):
    rows = []
    for topic, g in df.groupby("topic"):
        for e in EVALS:
            v = g[e].to_numpy(dtype=float)
            ok = ~np.isnan(v)
            if ok.sum() < 3:
                continue
            for r in REWARDS:
                rew = g[r].to_numpy(dtype=float)[ok]
                if np.isnan(rew).all():
                    continue
                rew = np.nan_to_num(rew, nan=np.nanmin(rew) - 1)
                for k in range(1, min(kmax, ok.sum()) + 1):
                    rows.append({"topic": topic, "eval": e, "reward": r, "k": k,
                                 "gain": best_of_k(rew, v[ok], k) - v[ok].mean()})
    return pd.DataFrame(rows)


def boot(x, n=4000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x))).mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def correlations(df):
    rows = []
    for topic, g in df.groupby("topic"):
        for e in EVALS:
            for r in REWARDS:
                a, b = g[r].to_numpy(dtype=float), g[e].to_numpy(dtype=float)
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() >= 4 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                    rows.append({"topic": topic, "eval": e, "reward": r, "rho": spearmanr(a[ok], b[ok])[0]})
    return pd.DataFrame(rows)


def plot(G, out, kmax):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("planted", "Planted worlds (truth known)"), ("testbed", "Testbed families")]
    evs = list(EVALS)
    fig, axes = plt.subplots(2, len(evs), figsize=(4.0 * len(evs), 6.6), sharex=True)
    for i, (grp, gname) in enumerate(groups):
        for j, e in enumerate(evs):
            ax = axes[i, j]
            sub = G[(G.group == grp) & (G["eval"] == e)]
            ax.text(0.02, 0.97, f"{sub.topic.nunique()} topics", transform=ax.transAxes, fontsize=8,
                    color="#52514e", va="top")
            ax.axhline(0, color="#b5b4ae", lw=1, ls=(0, (3, 3)))
            ends = []
            for r in REWARDS:
                s = sub[sub.reward == r].groupby("k").gain.agg(list)
                if s.empty:
                    continue
                ks = s.index.to_numpy()
                stats = np.array([boot(v) for v in s])
                ls = (0, (4, 2)) if r == "oracle" else "-"
                ax.plot(ks, stats[:, 0], color=COLOURS[r], lw=2, ls=ls, label=NAMES[r])
                if r != "oracle":
                    ax.fill_between(ks, stats[:, 1], stats[:, 2], color=COLOURS[r], alpha=0.10, lw=0)
                ends.append([stats[-1, 0], ks[-1], NAMES[r].split(" ")[0]])
            # direct labels at the line ends, pushed apart so they never overlap
            lo, hi = ax.get_ylim()
            gap = 0.075 * (hi - lo)
            ends.sort()
            for a in range(1, len(ends)):
                ends[a][0] = max(ends[a][0], ends[a - 1][0] + gap)
            for y, x, name in ends:
                ax.annotate(name, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=8, color="#52514e",
                            va="center")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color("#b5b4ae")
            ax.spines["bottom"].set_color("#b5b4ae")
            ax.tick_params(colors="#52514e", labelsize=8)
            ax.grid(axis="y", color="#ecebe7", lw=0.8)
            ax.set_xlim(1, kmax + 2.5)
            if i == 0:
                ax.set_title(EVALS[e], fontsize=10, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(f"{gname}\ngain over random report", fontsize=9, color="#52514e")
            if i == 1:
                ax.set_xlabel("best of k reports", fontsize=9, color="#52514e")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "bon_curves.png", dpi=150, facecolor="#fcfcfb")


def main(d: Path):
    df = pd.read_json(d / "reports.jsonl", lines=True)
    refs = pd.read_csv(d / "references.csv")
    df["group"] = np.where(df.topic.str.startswith("planted/"), "planted", "testbed")
    kmax = int(df.groupby("topic").size().max())
    G = gains(df, kmax)
    G["group"] = np.where(G.topic.str.startswith("planted/"), "planted", "testbed")
    G["kmax"] = G.groupby("topic").k.transform("max")
    G.to_csv(d / "gains.csv", index=False)
    C = correlations(df)
    C["group"] = np.where(C.topic.str.startswith("planted/"), "planted", "testbed")
    C.to_csv(d / "correlations.csv", index=False)

    lines = ["# Essay arena: best-of-n", "",
             f"{df.topic.nunique()} topics, {len(df)} reports (up to {kmax} per topic). Gains are best-of-{kmax} "
             "minus the mean report, averaged over topics, with 95% bootstrap CIs over topics. `oracle` on "
             "held-out questions is in-sample (an upper bound).", ""]
    for grp in ("planted", "testbed"):
        sub = G[(G.group == grp) & (G.k == G.kmax)]
        if sub.empty:
            continue
        lines += [f"## {grp}: gain of best-of-{kmax} over a random report", "",
                  "| reward | " + " | ".join(EVALS.values()) + " |", "|---" * (len(EVALS) + 1) + "|"]
        for r in REWARDS:
            cells = []
            for e in EVALS:
                x = sub[(sub.reward == r) & (sub["eval"] == e)].gain
                cells.append("" if x.empty else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*boot(x)))
            lines.append(f"| {NAMES[r]} | " + " | ".join(cells) + " |")
        cs = C[C.group == grp]
        lines += ["", f"Mean per-topic Spearman correlation of reward with evaluation ({grp}):", "",
                  "| reward | " + " | ".join(EVALS.values()) + " |", "|---" * (len(EVALS) + 1) + "|"]
        for r in REWARDS:
            cells = []
            for e in EVALS:
                x = cs[(cs.reward == r) & (cs["eval"] == e)].rho
                cells.append("" if x.empty else "{:+.2f} [{:+.2f}, {:+.2f}]".format(*boot(x)))
            lines.append(f"| {NAMES[r]} | " + " | ".join(cells) + " |")
        lines.append("")
    pl = df[df.group == "planted"]
    if not pl.empty and "true_test" in refs:
        lines += ["## Planted worlds: share of the gap from the mean report to the true theory closed by "
                  f"best-of-{kmax}", ""]
        rows = []
        for topic, g in pl.groupby("topic"):
            ref = refs[refs.topic == topic].iloc[0]
            gap = ref.true_test - g.eval_test.mean()
            row = {"topic": topic, "mean report": g.eval_test.mean(), "best report": g.eval_test.max(),
                   "true theory": ref.true_test, "bayes": ref.bayes_test, "no theory": ref.empty_test}
            for r in REWARDS:
                row[r] = (best_of_k(g[r].to_numpy(), g.eval_test.to_numpy(), len(g)) - g.eval_test.mean()) / gap
            rows.append(row)
        T = pd.DataFrame(rows).set_index("topic")
        lines += [T.round(3).to_markdown(), "", "Mean share closed: " + ", ".join(
            f"{NAMES[r]} {T[r].mean():+.2f}" for r in REWARDS), ""]
    (d / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    plot(G, d, kmax)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/arena-bon")
    main(Path(ap.parse_args().dir))
