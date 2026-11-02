"""Figures for notes/latent-essays-results.md (E0-E3). Run the four *_report.py scripts first.

    PYTHONPATH=. python experiments/essays_figures.py
"""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "notes" / "figures"
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
rng = np.random.default_rng(0)


def ci(x, n=4000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.where(np.asarray(y) == 1, np.log(p), np.log(1 - p))


def exp_ll(p, t):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return t * np.log(p) + (1 - t) * np.log(1 - p)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)


def hbar_ci(ax, labels, rows, color, zero=True):
    """Dot-and-interval rows, top to bottom."""
    y = np.arange(len(rows))[::-1]
    for yi, (m, lo, hi) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round")
        ax.plot(m, yi, "o", color=color, ms=8, mec=SURFACE, mew=2, zorder=3)
    ax.set_yticks(y, labels)
    if zero:
        ax.axvline(0, color=INK2, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(axis="y", length=0)


# ------------------------------------------------------------------ E0
def e0():
    d = pd.read_parquet(RES / "essays_validity" / "cells.parquet")
    empty = d[d.essay == "empty"].set_index("item").p
    W = d[d.essay != "empty"].pivot_table(index=["essay", "item"], columns="version", values="p").reset_index()
    W["empty"] = W.item.map(empty)
    reads = ["reading1", "reading2", "reading3"]
    rows = {
        "The essay itself\n(vs. no essay)": (W.original - W.empty).abs(),
        "Paraphrase\n(same claims, new words)": pd.concat([(W[f"paraphrase{i}"] - W.original).abs()
                                                           for i in (1, 2, 3)]),
        "Another reading of\nthe same text (noise)": pd.concat([(W[a] - W[b]).abs() for a, b in
                                                               [(reads[0], reads[1]), (reads[0], reads[2]),
                                                                (reads[1], reads[2])]]),
        "Author framed as\nrespected expert": (W.respected - W.original).abs(),
        "Author framed as\nunreliable": (W.unreliable - W.original).abs(),
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names = list(rows)
    vals = [rows[k].mean() for k in names]
    y = np.arange(len(names))[::-1]
    colors = [BLUE] + [MUTED] * (len(names) - 1)
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.2f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("mean change in the reader's probability, per (essay, question)")
    ax.set_title("E0. Content moves the reader; wording and author don't")
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "essays_e0_what_moves_the_reader.png")


# ------------------------------------------------------------------ E1
def e1_tables():
    R = pd.read_csv(RES / "essays_worlds" / "reports_scored.csv", dtype={"world": str})
    cells = pd.read_parquet(RES / "essays_worlds" / "cells.parquet")
    return R, cells


def e1_tracking(R, cells):
    H = R[R.condition.isin(["honest_haiku", "honest_gemini"])].copy()
    for col in ("oracle", "S_exp", "judge"):
        H[col + "_c"] = H[col] - H.groupby("setting")[col].transform("mean")
    profiles = ["misled", "uninformed", "partial", "weak", "strong"]
    pal = dict(zip(profiles, ["#e34948", "#b9b8b2", "#eda100", "#1baf7a", "#2a78d6"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, ylab, title in [
            (axes[0], "S_exp_c", "S2: reader's expected log score\n(nats/question, vs. setting mean)",
             "S2 rises with the author's real information"),
            (axes[1], "judge_c", "judge rating (1-10, vs. setting mean)", "The judge's rating does not")]:
        jitter = rng.normal(0, 0.06, len(H)) if col == "judge_c" else 0
        for p in profiles:
            g = H[H.profile == p]
            ax.scatter(g.oracle_c, g[col] + (jitter[H.profile.values == p] if col == "judge_c" else 0), s=26,
                       color=pal[p], alpha=0.85, edgecolor=SURFACE, linewidth=0.6, label=p)
        r = np.mean([spearmanr(g.oracle, g[col.replace("_c", "")]).correlation for _, g in H.groupby("setting")])
        ax.text(0.03, 0.95, f"Spearman {r:.2f}", transform=ax.transAxes, va="top", fontsize=10, color=INK)
        ax.set_xlabel("author's true information (nats/question, vs. setting mean)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(color=GRID, lw=0.6)
    axes[0].legend(title="author profile", frameon=False, fontsize=8, title_fontsize=8, loc="lower right")
    fig.suptitle("E1. Synthetic worlds: 80 honest essays, two settings", x=0.01, ha="left", fontsize=11,
                 color=INK2)
    save(fig, "essays_e1_tracking.png")

    # summary bars: Spearman with the oracle, incl. holistic
    hol = cells[cells.condition == "holistic"].copy()
    hol["e"] = exp_ll(hol.p, hol.p_true_model)
    hol = hol.groupby(["setting", "world", "profile"]).e.mean().rename("holistic").reset_index()
    hh = H[H.condition == "honest_haiku"].merge(hol, on=["setting", "world", "profile"])

    def sp(df, a):
        return np.mean([spearmanr(g[a], g.oracle).correlation for _, g in df.groupby("setting")])

    bars = [("S2 with true links\n(reader's parse of the essay)", sp(H.dropna(subset=["truelink"]), "truelink")),
            ("S2, exact expectation", sp(H, "S_exp")),
            ("S2, one realized outcome draw", sp(H, "S_real")),
            ("S2 on unseen questions\n(structured)", sp(hh, "S_exp_unseen")),
            ("Holistic: reader also sees\nother outcomes", sp(hh, "holistic")),
            ("LLM judge rating", sp(H.dropna(subset=["judge"]), "judge"))]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(bars))[::-1]
    cols = [BLUE, BLUE, BLUE, BLUE, ORANGE, ORANGE]
    ax.barh(y, [b[1] for b in bars], color=cols, height=0.62)
    for yi, (_, v) in zip(y, bars):
        ax.text(max(v, 0) + 0.015, yi, f"{v:.2f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [b[0] for b in bars])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spearman correlation with the author's true information")
    ax.set_title("E1. Which scores track what the author actually knows")
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "essays_e1_correlations.png")


def e1_reader(cells):
    t = cells[cells.condition == "truth"].copy()
    t["dir"] = np.where(t.p_true_model > 0.5, "yes more likely", "yes less likely")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = sorted(t.p_true_model.unique())
    for x in xs:
        g = t[t.p_true_model == x]
        ax.scatter(x + rng.normal(0, 0.012, len(g)), g.p, s=14, color=BLUE, alpha=0.35, edgecolor="none")
        ax.plot([x - 0.035, x + 0.035], [g.p.mean()] * 2, color=INK, lw=2.5)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(0.525, 0.575, "perfect reader", color=INK2, fontsize=8.5, ha="center", rotation=38)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(xs, [f"{x:.2f}" for x in xs])
    ax.set_xlabel("true P(yes) given the facts (by design)")
    ax.set_ylabel("reader's P(yes), given the true facts as certain")
    ax.set_title("E1. Even told the truth, the reader blurs the links")
    ax.grid(color=GRID, lw=0.6)
    fig.text(0.01, 0.01, "Dots: one question in one world; bars: means. The reader barely separates 0.65\nfrom "
             "0.85 (or 0.40 from 0.20), and points the wrong way on about 20% of questions.", fontsize=8.5,
             color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / "essays_e1_reader_links.png", dpi=170)
    plt.close(fig)


def e1_manipulations(R):
    def paired(a, b, col):
        A = R[R.condition == a].set_index(["setting", "world", "profile"])[col]
        B = R[R.condition == b].set_index(["setting", "world", "profile"])[col]
        return (A - B).dropna()

    s2 = [("Eloquent restyle\n(same claims)", ci(paired("eloquent", "honest_haiku", "S_exp"))),
          ("Overconfident\n(true world)", ci(paired("overconfident", "honest_haiku", "S_exp"))),
          ("Overconfident\n(author's own beliefs)", ci(paired("overconfident", "honest_haiku", "belief"))),
          ("List of predictions,\nquestions it listed", ci(paired("list", "honest_haiku", "S_exp_seen"))),
          ("List of predictions,\nunseen questions", ci(paired("list", "honest_haiku", "S_exp_unseen")))]
    jd = [("Eloquent restyle", ci(paired("eloquent", "honest_haiku", "judge"))),
          ("Overconfident", ci(paired("overconfident", "honest_haiku", "judge")))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    hbar_ci(axes[0], [a for a, _ in s2], [b for _, b in s2], BLUE)
    axes[0].set_xlabel("change in S2 vs. the honest essay (nats/question)")
    axes[0].set_title("S2: small, proper responses")
    hbar_ci(axes[1], [a for a, _ in jd], [b for _, b in jd], ORANGE)
    axes[1].set_xlabel("change in judge rating (points)")
    axes[1].set_title("Judge: penalizes both")
    fig.suptitle("E1. Manipulating the same author's essay (80 pairs each, 95% CIs)", x=0.01, ha="left",
                 fontsize=11, color=INK2)
    save(fig, "essays_e1_manipulations.png")


# ------------------------------------------------------------------ E3
def e3():
    d = pd.read_json(RES / "essays_optimize" / "rounds.json")
    key = ["setting", "world", "profile"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, kind, title in [(axes[0], "S_train", "Optimizing S2 on 15 resolved questions"),
                            (axes[1], "judge", "Optimizing the judge's rating")]:
        g = d[d.reward == kind]
        base = g[g["round"] == 0].set_index(key)
        rounds = sorted(g["round"].unique())
        lines = [("held-out score (truth, unseen questions)", "heldout_expected", BLUE)]
        if kind == "S_train":
            lines.insert(0, ("training score (the reward)", "reward_value", ORANGE))
        for lab, col, colr in lines:
            ch = [ci((g[g["round"] == r].set_index(key)[col] - base[col]).values) for r in rounds]
            m = [c[0] for c in ch]
            ax.fill_between(rounds, [c[1] for c in ch], [c[2] for c in ch], color=colr, alpha=0.15, lw=0)
            ax.plot(rounds, m, color=colr, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=lab)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(rounds)
        ax.set_xlabel("round of hill climbing (best of 4 edits)")
        ax.set_title(title)
        ax.grid(color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    jg = d[d.reward == "judge"]
    jgain = (jg[jg["round"] == 4].set_index(key).judge - jg[jg["round"] == 0].set_index(key).judge).mean()
    axes[1].text(0.97, 0.05, f"judge rating itself: {jgain:+.2f} points", transform=axes[1].transAxes,
                 ha="right", fontsize=9, color=INK2)
    axes[0].set_ylabel("change from round 0 (nats/question)")
    fig.suptitle("E3. A small-sample reward overfits; neither reward collapses at this pressure "
                 "(8 runs, 95% CIs)", x=0.01, ha="left", fontsize=11, color=INK2)
    save(fig, "essays_e3_optimization.png")


# ------------------------------------------------------------------ E2
def e2():
    E = pd.read_csv(RES / "essays_history" / "essays_scored.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    rows = [("S1, the essay's topic", ci(E.S1)), ("S1, placebo topics", ci(E.S1_placebo)),
            ("S2, the essay's topic", ci(E.S2))]
    hbar_ci(axes[0], [a for a, _ in rows], [b for _, b in rows], BLUE)
    axes[0].set_xlabel("gain over the date-only reader\n(nats/question, mean over 75 essays)")
    axes[0].set_title("Essays don't help on average")
    x = E.dropna(subset=["S1_a", "S1_b"])
    axes[1].scatter(x.S1_a, x.S1_b, s=22, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=0.6)
    axes[1].set_xlabel("essay's gain on 10 questions")
    axes[1].set_ylabel("same essay, 10 other questions")
    axes[1].set_title("…and per-essay scores don't replicate")
    axes[1].text(0.03, 0.95, f"r = {np.corrcoef(x.S1_a, x.S1_b)[0, 1]:.2f}", transform=axes[1].transAxes,
                 va="top")
    x = E.dropna(subset=["judge", "S2"])
    axes[2].scatter(x.judge + rng.normal(0, 0.08, len(x)), x.S2, s=22, color=ORANGE, alpha=0.75,
                    edgecolor=SURFACE, linewidth=0.6)
    axes[2].set_xlabel("judge rating (jittered)")
    axes[2].set_ylabel("S2 gain (nats/question)")
    axes[2].set_title("The judge disagrees with S2")
    axes[2].text(0.03, 0.95, f"Spearman {spearmanr(x.judge, x.S2).correlation:.2f}", transform=axes[2].transAxes,
                 va="top")
    for ax in axes:
        ax.grid(color=GRID, lw=0.6)
    fig.suptitle("E2. 75 LessWrong essays (Sep 2025-Feb 2026), each scored on 20 later Manifold questions",
                 x=0.01, ha="left", fontsize=11, color=INK2)
    save(fig, "essays_e2_real_essays.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    e0()
    R, cells = e1_tables()
    e1_tracking(R, cells)
    e1_reader(cells)
    e1_manipulations(R)
    e3()
    e2()
    print("written to", FIG)
