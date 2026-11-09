"""Tables and figures for latent_eic.py: empirical incentive compatibility of the latent reward, the judge and debate.

Units. Each reward is standardized by its standard deviation over all instructed essays (every writer, both sides),
so ASD reads "how many standard deviations of the reward the true side earns over the false side". Concordance (the
share of claims where the true side earns more; ties count half) is unitless and needs no standardization. 95% CIs
by bootstrap over claims (pooled rows: over claims, with writers resampled within).

    python experiments/latent_eic_report.py
"""

from __future__ import annotations

import argparse
import glob
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JUDGES = {"openai/gpt-4o-mini": "gpt-4o-mini", "openai/gpt-4.1": "gpt-4.1"}
REWARDS = ["latent"] + [f"judge:{j}" for j in JUDGES] + [f"debate:{j}" for j in JUDGES]
COLOUR = {"latent": "#2a78d6", "judge": "#eb6834", "debate": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e3"


def label(r):
    kind, _, j = r.partition(":")
    return "latent" if kind == "latent" else f"{kind} ({JUDGES[j]})"


def style(r):
    """Hue = reward family; marker fill = judge (filled gpt-4.1, hollow gpt-4o-mini)."""
    kind, _, j = r.partition(":")
    c = COLOUR[kind]
    hollow = j == "openai/gpt-4o-mini"
    return dict(color=c, markerfacecolor="white" if hollow else c, markeredgecolor=c, marker="o", markersize=7,
                markeredgewidth=1.6, linestyle="--" if hollow else "-")


def load(d: Path, pattern):
    """Rows of the matching files, without refusals or empty outputs (essays under 40 words: 6 of 1,512 instructed
    essays, all from Claude Opus 5 and Sonnet 5), with the reader's implied credence in the assigned position."""
    import hashlib

    rows = []
    for f in sorted(glob.glob(str(d / pattern))):
        rows += [json.loads(x) for x in open(f)]
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df = df[df.words >= 4 if pattern.startswith("bare") else df.words >= 40]
    cf = d / "credence.jsonl"
    if cf.exists():
        cr = pd.read_json(cf, lines=True)
        df = df.assign(essay_hash=df.essay.map(lambda e: hashlib.sha256(e.encode()).hexdigest()[:10]))
        df = df.merge(cr, on=["claim", "essay_hash"], how="left")
        df["credence"] = df.credence_true.where(df.side == 0, 1 - df.credence_true)
    return df


def paired(df, col, by=("writer", "claim")):
    """Per (writer, claim): reward of the true side minus the false side (first sample of each)."""
    g = df.groupby(list(by) + ["side"])[col].mean().unstack("side")
    return (g[0] - g[1]).dropna()


def boot(diff: pd.Series, f, n=600, seed=0):
    """CI resampling claims (and, within a claim, its writers)."""
    rng = np.random.default_rng(seed)
    claims = diff.index.get_level_values("claim").unique()
    by = {c: diff.xs(c, level="claim").values for c in claims}
    vals = []
    for _ in range(n):
        pick = rng.choice(claims, len(claims))
        vals.append(f(np.concatenate([rng.choice(by[c], len(by[c])) for c in pick])))
    return np.percentile(vals, [2.5, 97.5])


def conc(x):
    return float(np.mean((x > 0) + 0.5 * (x == 0)))


def summarize(diff, sd):
    x = diff.values
    lo, hi = boot(diff, lambda v: v.mean() / sd)
    clo, chi = boot(diff, conc)
    return {"asd": x.mean() / sd, "lo": lo, "hi": hi, "conc": conc(x), "clo": clo, "chi": chi, "n": len(x)}


def best_of_k(sel, ho, k):
    order = np.argsort(-np.asarray(sel), kind="stable")
    v = np.asarray(ho)[order]
    n = len(v)
    probs = np.array([comb(n - 1 - i, k - 1) for i in range(n)], dtype=float) / comb(n, k)
    return float(probs @ v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "latent-eic"))
    args = ap.parse_args()
    d = Path(args.dir)
    from latent_eic import WRITERS  # noqa: E402 (experiments/ on sys.path when run as a script)

    inst = load(d, "instructed-*.jsonl")
    bare = load(d, "bare.jsonl")
    sds = {r: inst[f"{r}_sel"].std() for r in REWARDS}
    sds["latent_exact"] = inst["latent_exact"].std()
    lines = ["# Empirical incentive compatibility of the latent reward (planted-world claims)", ""]
    n_claims = inst.claim.nunique()
    lines += [f"{inst.writer.nunique()} writers x {n_claims} claims x 2 sides; reader gpt-4.1-mini (truncation "
              "reported in the run log). ASD in standard deviations of each reward over all instructed essays; "
              "concordance = share of (writer, claim) pairs where the true side earns more. 95% CIs over claims.", ""]

    # audit
    au = inst.groupby("side").audit.value_counts(normalize=True).unstack().fillna(0)
    lines += ["## Audit: did the writer argue its assigned side?", "", au.round(3).to_markdown(), "",
              "Side 0 is the true side, 1 the false side.", ""]
    by_w = inst[inst.side == 1].groupby("writer").audit.value_counts(normalize=True).unstack().fillna(0)
    by_w.index = [WRITERS[w][0] for w in by_w.index]
    lines += ["False side, by writer:", "", by_w.round(2).to_markdown(), ""]

    if "credence" in inst:
        c = inst.groupby([inst.writer.map(lambda w: WRITERS[w][0]), "side"]).credence.mean().unstack("side")
        c.columns = ["true side", "false side"]
        lines += ["## How strongly do essays commit? (reader's probability that the assigned position is true, "
                  "given the essay)", "", c.round(2).to_markdown(), ""]

    # pooled table: instructed, committed, bare
    committed = inst.groupby(["writer", "claim"]).apply(
        lambda g: (g[g.side == 1].audit == "committed").all() and (g[g.side == 0].audit != "conceded").all())
    keep = committed[committed].index
    rows = []
    for r in REWARDS + ["latent_exact"]:
        col = r if r == "latent_exact" else f"{r}_sel"
        sd = sds[r]
        dif = paired(inst, col)
        row = {"reward": "latent (exact expectation)" if r == "latent_exact" else label(r)}
        s = summarize(dif, sd)
        row["instructed"] = f"{s['asd']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]"
        row["instructed conc."] = f"{s['conc']:.2f}"
        s = summarize(dif[dif.index.isin(keep)], sd)
        row["committed liars"] = f"{s['asd']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]"
        row["committed conc."] = f"{s['conc']:.2f}"
        if not r.startswith("debate"):
            b = paired(bare.assign(writer="bare"), col)
            s = summarize(b, sd)
            row["bare claim (no argument)"] = f"{s['asd']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]"
        rows.append(row)
    lines += ["## Pooled over writers", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # by writer
    per = []
    for w, g in inst.groupby("writer"):
        name, aa = WRITERS[w]
        ok = committed.loc[w]
        for r in REWARDS:
            dif = paired(g, f"{r}_sel")
            s = summarize(dif, sds[r])
            dc = dif[dif.index.get_level_values("claim").isin(ok[ok].index)]
            sc = summarize(dc, sds[r]) if len(dc) > 4 else {"asd": np.nan, "lo": np.nan, "hi": np.nan}
            per.append({"writer": name, "aa": aa, "reward": r, **s, "asd_c": sc["asd"], "lo_c": sc["lo"],
                        "hi_c": sc["hi"], "n_c": len(dc)})
    per = pd.DataFrame(per)
    per.to_csv(d / "by_writer.csv", index=False)
    tab = per.pivot(index=["aa", "writer"], columns="reward", values="asd").round(2)
    tab.columns = [label(c) for c in tab.columns]
    lines += ["## By writer (instructed ASD, standard deviations)", "", tab.to_markdown(), ""]
    slopes = []
    for r in REWARDS if per.writer.nunique() >= 3 else []:
        p = per[per.reward == r]
        b = np.polyfit(p.aa, p.asd, 1)[0] * 10
        rng = np.random.default_rng(1)
        bs = [np.polyfit(*p.sample(len(p), replace=True, random_state=rng.integers(1e9))[["aa", "asd"]].T.values, 1)[0]
              * 10 for _ in range(2000)]
        slopes.append({"reward": label(r), "slope per 10 index points": f"{b:+.3f} [{np.percentile(bs, 2.5):+.3f}, "
                                                                         f"{np.percentile(bs, 97.5):+.3f}]"})
    lines += ["Trend with writer intelligence (OLS slope of ASD on the AA index, CI resampling writers):", "",
              pd.DataFrame(slopes).to_markdown(index=False), ""]
    fig_writers(per, d)

    # best-of-n
    bon = load(d, "bon-*.jsonl")
    if len(bon):
        curves = []
        for w, g in bon.groupby("writer"):
            for r in REWARDS:
                for k in range(1, g["sample"].max() + 2):
                    vals = {}
                    for (c, s), gg in g.groupby(["claim", "side"]):
                        vals[(c, s)] = best_of_k(gg[f"{r}_sel"], gg[f"{r}_ho"], k)
                    s_ = pd.Series(vals).unstack()
                    dif = (s_[0] - s_[1]).dropna()
                    dif.index = pd.MultiIndex.from_arrays([[w] * len(dif), dif.index], names=["writer", "claim"])
                    curves.append({"writer": WRITERS[w][0], "reward": r, "k": k, **summarize(dif, sds[r])})
                    # also each side's mean held-out reward under selection
                    curves[-1]["true_side"] = s_[0].mean() / sds[r]
                    curves[-1]["false_side"] = s_[1].mean() / sds[r]
        curves = pd.DataFrame(curves)
        curves.to_csv(d / "bon_curves.csv", index=False)
        t = curves.pivot(index=["writer", "k"], columns="reward", values="asd").round(2)
        t.columns = [label(c) for c in t.columns]
        lines += ["## Best-of-n per side (selected on one reward draw, scored on a fresh one)", "", t.to_markdown(), ""]
        fig_bon(curves, d)

    # prompt search
    po = load(d, "po-*.jsonl")
    if len(po):
        lines += po_section(po, sds, d)

    (d / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def po_section(po, sds, d):
    lines = ["## Prompt search (gpt-4.1-mini writer; Claude Opus 5.5 searches 4 rounds x 4 strategies per side "
             "against each reward on 16 training claims; held-out: the other claims, fresh reward draws)", ""]
    recs = []
    for rname, g in po.groupby("reward_searched"):
        full = "latent" if rname == "latent" else next(r for r in REWARDS if r.startswith(rname + ":") and "4o-mini" in r)
        sd = sds[full]
        tr = g[g.sampler == "po-train"]
        ho = g[g.sampler == "po-heldout"]
        for side in (0, 1):
            t = tr[tr.side == side].groupby(["round", "strategy"])[f"{full}_sel"].mean().reset_index()
            best = -np.inf
            for rnd in sorted(t["round"].unique()):
                best = max(best, t[t["round"] <= rnd][f"{full}_sel"].max())
                recs.append({"reward": rname, "side": side, "round": rnd, "train_best": best / sd})
        held = {}
        for side in (0, 1):
            h = ho[ho.side == side]
            last = None
            for rnd in range(0, int(po["round"].max()) + 1):
                cur = h[h["round"] == rnd]
                if len(cur):
                    last = cur.groupby("claim")[f"{full}_ho"].mean()
                held[(side, rnd)] = last
        for rnd in range(0, int(po["round"].max()) + 1):
            dif = (held[(0, rnd)] - held[(1, rnd)]).dropna()
            dif.index = pd.MultiIndex.from_arrays([["w"] * len(dif), dif.index], names=["writer", "claim"])
            s = summarize(dif, sd)
            for side in (0, 1):
                next(x for x in recs if x["reward"] == rname and x["side"] == side and x["round"] == rnd)[
                    "heldout"] = held[(side, rnd)].mean() / sd
            recs.append({"reward": rname, "side": "asd", "round": rnd, **s})
        # audit of the final deceptive essays
        fin = ho[(ho.side == 1) & (ho["round"] == ho[ho.side == 1]["round"].max())]
        r0 = ho[(ho.side == 1) & (ho["round"] == 0)]
        cr = (f"; implied credence in the false position {r0.credence.mean():.2f} (round 0) -> "
              f"{fin.credence.mean():.2f} (final)") if "credence" in ho else ""
        lines.append(f"- {rname}: final deceptive strategy's held-out essays audited "
                     f"{fin.audit.value_counts(normalize=True).round(2).to_dict()}{cr}")
    rec = pd.DataFrame(recs)
    rec.to_csv(d / "po_curves.csv", index=False)
    a = rec[rec.side == "asd"].copy()
    a["ASD (held-out)"] = a.apply(lambda x: f"{x.asd:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}]", axis=1)
    lines += ["", a.pivot(index="round", columns="reward", values="ASD (held-out)").to_markdown(), ""]
    strategies = {f.stem: json.loads(f.read_text()) for f in d.glob("po-*-strategies.json")}
    for k, v in strategies.items():
        lines += [f"**{k}**, final honest strategy: {v['honest'][-1] or '(none)'}", "",
                  f"**{k}**, final deceptive strategy: {v['deceptive'][-1] or '(none)'}", ""]
    fig_po(rec, d)
    return lines


def _axes(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")


def fig_writers(per, d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (col, lo, hi, title) in zip(axes, [("asd", "lo", "hi", "all instructed essays"),
                                                ("asd_c", "lo_c", "hi_c", "false side committed (audited)")]):
        for r in REWARDS:
            p = per[per.reward == r].sort_values("aa")
            st = style(r)
            ax.errorbar(p.aa, p[col], yerr=[p[col] - p[lo], p[hi] - p[col]], fmt="none", ecolor=st["color"],
                        alpha=0.35, lw=1)
            ax.plot(p.aa, p[col], **st, lw=1.2, alpha=0.9)
            ax.annotate(label(r), (p.aa.iloc[-1], p[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")
        _axes(ax)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("writer intelligence (Artificial Analysis index)", color=INK)
        ax.set_xlim(right=per.aa.max() + 9)
    axes[0].set_ylabel("ASD: true side minus false side\n(standard deviations of each reward)", color=INK)
    fig.suptitle("Does each reward pay more for arguing the true position? (planted-world claims)", color=INK)
    fig.tight_layout()
    fig.savefig(d / "eic_by_writer.png", dpi=160)
    plt.close(fig)


def fig_bon(curves, d):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for r in REWARDS:
        p = curves[curves.reward == r]
        st = style(r)
        ax.fill_between(p.k, p.lo, p.hi, color=st["color"], alpha=0.08, lw=0)
        ax.plot(p.k, p.asd, **st, lw=1.6)
        ax.annotate(label(r), (p.k.iloc[-1], p.asd.iloc[-1]), xytext=(6, 0), textcoords="offset points", color=INK,
                    fontsize=8, va="center")
    _axes(ax)
    ax.set_xticks(sorted(curves.k.unique()))
    ax.set_xlim(right=curves.k.max() + 1.2)
    ax.set_xlabel("best of k essays per side (selected on the reward)", color=INK)
    ax.set_ylabel("ASD on a fresh reward draw (s.d.)", color=INK)
    ax.set_title(f"Best-of-n per side ({curves.writer.iloc[0]} writer)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "eic_best_of_n.png", dpi=160)
    plt.close(fig)


def fig_po(rec, d):
    rewards = list(dict.fromkeys(rec.reward))
    fig, axes = plt.subplots(1, len(rewards) + 1, figsize=(4.2 * (len(rewards) + 1), 4.2))
    for ax, rn in zip(axes, rewards):
        c = COLOUR[rn]
        for side, ls, nm in ((0, "-", "honest"), (1, "--", "deceptive")):
            p = rec[(rec.reward == rn) & (rec.side == side)].sort_values("round")
            ax.plot(p["round"], p.train_best, color=c, ls=ls, lw=1.2, alpha=0.45, marker="o", markersize=5,
                    markerfacecolor="white")
            ax.plot(p["round"], p.heldout, color=c, ls=ls, lw=2, marker="o", markersize=7)
            ax.annotate(f"{nm}, held-out", (p["round"].iloc[-1], p.heldout.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=8, color=INK, va="center")
        _axes(ax)
        ax.axhline(0, lw=0)
        ax.set_title(f"searching against {rn}", color=INK, fontsize=11)
        ax.set_xlabel("search round", color=INK)
        ax.set_xlim(right=rec["round"].max() + 1.6)
    axes[0].set_ylabel("mean reward (s.d. units; faint: training claims)", color=INK)
    ax = axes[-1]
    for rn in rewards:
        p = rec[(rec.reward == rn) & (rec.side == "asd")].sort_values("round")
        ax.fill_between(p["round"].astype(float), p.lo.astype(float), p.hi.astype(float), color=COLOUR[rn], alpha=0.1,
                        lw=0)
        ax.plot(p["round"], p.asd, color=COLOUR[rn], marker="o", lw=2, markersize=7)
        ax.annotate(rn, (p["round"].iloc[-1], p.asd.iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color=INK, va="center")
    _axes(ax)
    ax.set_title("held-out ASD (honest minus deceptive)", color=INK, fontsize=11)
    ax.set_xlabel("search round", color=INK)
    ax.set_xlim(right=rec["round"].max() + 1.2)
    fig.tight_layout()
    fig.savefig(d / "eic_prompt_search.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
