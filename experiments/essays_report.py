"""Analysis of experiments/essays.py: held-out scores, mixtures, and selection by S vs by an LLM judge.

    PYTHONPATH=. python experiments/essays_report.py [--judge anthropic/claude-sonnet-4-5]

Writes results/essays/summary.md, summary.json and figures.
"""

import argparse
import asyncio
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import cvtestbed as cvt

from lvpm.essays import score as S
from lvpm.essays import writers as W

OUT = Path(__file__).resolve().parents[1] / "results" / "essays"  # overridden by --out
KINDS = ["empty", "examples", "uninformed", "informed", "mixture"]


def mean_log(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))


def item_logs(cells, reports):
    """Per (report, test item) log score of the report's marginal prediction - for paired comparisons."""
    out = []
    for r in reports:
        c = cells[(cells.family == r["family"]) & (cells.report == r["id"]) & (cells.split == "test")]
        if c.empty:
            continue
        piv = c.pivot_table(index="hyp", columns="item", values="p")
        y = c.drop_duplicates("item").set_index("item").loc[piv.columns, "y"].to_numpy()
        m = S.marginals(np.array(r["weights"], dtype=float), piv.to_numpy())
        m = np.clip(m, 1e-12, 1 - 1e-12)
        for it, mi, yi in zip(piv.columns, m, y):
            out.append({"family": r["family"], "id": r["id"], "kind": r["kind"], "item": it,
                        "log": float(np.log(mi) if yi == 1 else np.log1p(-mi))})
    return pd.DataFrame(out)


def paired_gains(il, n_boot=4000, seed=0):
    """Mean gain over the no-essay baseline per (family, kind), with a bootstrap CI over test items."""
    rng = np.random.default_rng(seed)
    rows = []
    for fam, g in il.groupby("family"):
        base = g[g.kind == "empty"].set_index("item").log
        for kind, gk in g[g.kind != "empty"].groupby("kind"):
            d = gk.groupby("item").log.mean() - base  # mean over the kind's reports, per item
            d = d.dropna().to_numpy()
            boots = rng.choice(d, size=(n_boot, len(d))).mean(axis=1)
            rows.append({"family": fam, "kind": kind, "gain": d.mean(), "lo": np.quantile(boots, 0.025),
                         "hi": np.quantile(boots, 0.975), "items": len(d)})
    return pd.DataFrame(rows)


def report_scores(cells, reports):
    """One row per report: mean per-item log score and accuracy on dev and test (mixtures: prior and updated)."""
    rows = []
    for r in reports:
        c = cells[(cells.family == r["family"]) & (cells.report == r["id"])]
        if c.empty:
            continue
        piv = c.pivot_table(index="hyp", columns="item", values="p")
        items = list(piv.columns)
        y = c.drop_duplicates("item").set_index("item").loc[items, "y"].to_numpy()
        split = c.drop_duplicates("item").set_index("item").loc[items, "split"].to_numpy()
        P, w = piv.to_numpy(), np.array(r["weights"], dtype=float)
        dev, test = split == "dev", split == "test"
        m = S.marginals(w, P)
        row = {"family": r["family"], "id": r["id"], "kind": r["kind"], "writer": r["writer"], "sample": r["sample"],
               "k": len(w), "words": int(np.sum(r["words"])),
               "dev_log": mean_log(m[dev], y[dev]), "test_log": mean_log(m[test], y[test]),
               "test_acc": float(np.mean((m[test] > 0.5) == (y[test] == 1))),
               "dev_acc": float(np.mean((m[dev] > 0.5) == (y[dev] == 1)))}
        if len(w) > 1:
            # the latent at work: update on DEV outcomes, predict TEST
            y_dev = np.where(dev, y, np.nan)
            for tau in (1.0, 0.9):
                pred = S.predictive(w, P, y_dev, tau)
                row[f"test_log_updated_tau{tau:g}"] = mean_log(pred[test], y[test])
            row["test_heldout_joint_per_item"] = S.held_out_score(w, P, np.where(dev | test, y, np.nan), test) / test.sum()
            row["I_joint_bits"] = S.information(w, P[:, test], n_samples=2000)["I_joint_bits"]
            row["weight_entropy_bits"] = float(-(w * np.log2(np.maximum(w, 1e-12))).sum())
            row["post_max_weight"] = float(S.posterior(w, P, y_dev).max())
        rows.append(row)
    return pd.DataFrame(rows)


def best_of_k(dev, test, k):
    """Exact E[test score of the dev-argmax among a uniformly random k-subset] (ties: mean)."""
    n = len(dev)
    order = np.argsort(-np.asarray(dev), kind="stable")
    t = np.asarray(test)[order]
    # P(rank-i candidate is the best of a random k-subset) = C(n-1-i, k-1) / C(n, k)
    probs = np.array([comb(n - 1 - i, k - 1) for i in range(n)], dtype=float) / comb(n, k)
    return float(probs @ t)


async def judge_all(df, reports, judge_model):
    by_id = {(r["family"], r["id"]): r for r in reports}
    todo = df[df.kind == "informed"]
    cards = cvt.families()

    async def one(row):
        r = by_id[(row.family, row.id)]
        s, c = await W.judge_essay(judge_model, cards[cvt.ALIASES.get(row.family, row.family)].card, r["hypotheses"][0])
        return s, c

    res = await asyncio.gather(*[one(row) for row in todo.itertuples()])
    df = df.copy()
    df.loc[todo.index, "judge"] = [s for s, _ in res]
    return df, sum(c for _, c in res)


def selection_curves(df):
    out = []
    for fam, g in df[df.kind == "informed"].groupby("family"):
        g = g.dropna(subset=["judge"])
        n = len(g)
        for k in range(1, n + 1):
            out.append({"family": fam, "k": k,
                        "by_dev_S": best_of_k(g.dev_log.to_numpy(), g.test_log.to_numpy(), k),
                        "by_judge": best_of_k(g.judge.to_numpy() + 1e-6 * g.dev_log.to_numpy(), g.test_log.to_numpy(), k),
                        "random": float(g.test_log.mean())})
    return pd.DataFrame(out)


SERIES = {"examples": "#2a78d6", "uninformed": "#eb6834", "informed": "#1baf7a", "mixture": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _axes_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def figures(df, curves, gains):
    """Forest plot of held-out gains (95% CIs over test items) and best-of-k selection curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = list(dict.fromkeys(df.family))
    kinds = [k for k in SERIES if k in set(gains.kind)]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(fams) + 1.2))
    _axes_style(ax)
    off = np.linspace(-0.27, 0.27, len(kinds))
    for j, k in enumerate(kinds):
        g = gains[gains.kind == k].set_index("family")
        ys = [i + off[j] for i, f in enumerate(fams) if f in g.index]
        rows = [g.loc[f] for f in fams if f in g.index]
        ax.hlines(ys, [r.lo for r in rows], [r.hi for r in rows], color=SERIES[k], lw=2)
        ax.plot([r.gain for r in rows], ys, "o", ms=6, color=SERIES[k], mec="white", mew=1.5, label=k)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(fams)), fams, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("held-out log score gain over the reader alone (nats per item, 95% CI)", color=MUTED, fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(kinds))
    fig.tight_layout()
    fig.savefig(OUT / "essay_value.png", dpi=160)
    plt.close(fig)
    if curves is not None and not curves.empty:
        fams = list(dict.fromkeys(curves.family))
        fig, axes = plt.subplots(1, len(fams), figsize=(2.6 * len(fams), 2.8))
        axes = np.atleast_1d(axes)
        for ax, fam in zip(axes, fams):
            _axes_style(ax)
            ax.grid(axis="y", color=GRID, lw=0.8)
            g = curves[curves.family == fam]
            ax.plot(g.k, g.by_dev_S, color="#2a78d6", lw=2, label="select by dev $S$")
            ax.plot(g.k, g.by_judge, color="#eb6834", lw=2, label="select by LLM judge")
            ax.plot(g.k, g.random, color=MUTED, lw=1.2, ls="--", label="no selection")
            ax.set_title(fam, fontsize=9, color=INK)
            ax.set_xlabel("candidates (k)", fontsize=8, color=MUTED)
        axes[0].set_ylabel("held-out log score per item", fontsize=8, color=MUTED)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "selection.png", dpi=160)
        plt.close(fig)


def main(args):
    global OUT
    OUT = Path(args.out)
    cells = pd.read_parquet(OUT / "cells.parquet")
    reports = json.loads((OUT / "reports.json").read_text())
    df = report_scores(cells, reports)
    df, jcost = asyncio.run(judge_all(df, reports, args.judge))
    curves = selection_curves(df)
    df.to_csv(OUT / "reports_scored.csv", index=False)
    curves.to_csv(OUT / "selection_curves.csv", index=False)
    gains = paired_gains(item_logs(cells, reports))
    gains.to_csv(OUT / "gains.csv", index=False)
    figures(df, curves, gains)
    agg = (df.groupby(["family", "kind"]).agg(n=("id", "size"), test_log=("test_log", "mean"),
                                              test_acc=("test_acc", "mean"), words=("words", "mean"))
           .reset_index())
    base = agg[agg.kind == "empty"].set_index("family").test_log
    agg["gain_vs_empty"] = agg.test_log - agg.family.map(base)
    lines = ["# Essays as latents: first results", "",
             f"Reader: {json.loads((OUT / 'meta.json').read_text())['reader']}. Judge: {args.judge} (cost ${jcost:.2f}).", "",
             "## Held-out value of reports (test items, mean log score per item, nats)", "",
             agg.round(3).to_markdown(index=False), "",
             "## Gain over the no-essay reader, with 95% bootstrap CIs over test items (nats per item)", "",
             gains.round(3).to_markdown(index=False), ""]
    mix = df[df.kind == "mixture"]
    if not mix.empty:
        cols = ["family", "writer", "test_log", "test_log_updated_tau1", "test_log_updated_tau0.9", "I_joint_bits",
                "post_max_weight"]
        lines += ["## Mixtures: prior vs updated on dev outcomes", "", mix[cols].round(3).to_markdown(index=False), ""]
    inf = df[df.kind == "informed"]
    if not inf.empty:
        rho = inf.groupby("family").apply(lambda g: pd.Series({
            "spearman(judge, test)": g.judge.corr(g.test_log, method="spearman"),
            "spearman(dev S, test)": g.dev_log.corr(g.test_log, method="spearman")})).reset_index()
        lines += ["## Does the training signal track held-out value? (informed essays)", "",
                  rho.round(3).to_markdown(index=False), ""]
        last = curves.sort_values("k").groupby("family").tail(1)
        lines += ["## Best-of-all selection (k = all candidates)", "", last.round(3).to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="anthropic/claude-sonnet-4-5")
    ap.add_argument("--out", default=str(OUT))
    main(ap.parse_args())
