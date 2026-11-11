"""Figures for the essays-as-latents results, in the style of so-arena's incentive-compatibility report (task #8).

Reads saved results only (no model calls):

* ``results/latent-eic/``: the latent score as an oversight mechanism (``latent_eic.py``): figures 1-5;
* ``results/arena-bon/``: best-of-12 reports with every reward and value measure (``arena_bon.py``,
  ``arena_s1.py``, ``arena_measures.py``): figures 6-7;
* ``results/arena-iterate/``: in-context search against each reward (``arena_iterate.py``): figure 8.

Units are each reward's own: the latent score in nats per question (its exact expectation over outcomes, which is
what the writer's incentive depends on), the judge and debate in rating points (1-10). Intervals are 95% bootstrap
intervals over claims (figures 1-5) or topics (6-8).

    python experiments/latent_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style as st  # noqa: E402
from fig_style import AQUA, BLUE, CONTEXT, INK, INK2, MUTED, ORANGE, SURFACE, plt  # noqa: E402
from latent_eic import WRITERS  # noqa: E402
from latent_eic_report import REWARDS, best_of_k, boot, load, paired  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EIC = ROOT / "results" / "latent-eic"
BON = ROOT / "results" / "arena-bon"
ITER = ROOT / "results" / "arena-iterate"
OUT = ROOT / "results" / "figures"

NAME = {"latent": "Latent score", "judge:openai/gpt-4o-mini": "Judge, gpt-4o-mini",
        "judge:openai/gpt-4.1": "Judge, gpt-4.1", "debate:openai/gpt-4o-mini": "Debate, gpt-4o-mini judge",
        "debate:openai/gpt-4.1": "Debate, gpt-4.1 judge"}
UNIT = {"latent": "nats/question"}
FAMILY_COLOUR = {"latent": BLUE, "judge": ORANGE, "debate": AQUA}


def unit(r):
    return UNIT.get(r, "rating points")


def colour(r):
    return FAMILY_COLOUR[r.partition(":")[0]]


def value_col(r):
    """The column a reward is scored by: the latent score's noise-free expectation; a judge's second (fresh) draw."""
    return "latent_exact" if r == "latent" else f"{r}_ho"


def instructed_value(df, r):
    """Instructed essays: the latent expectation, or the mean of a judge's two independent draws."""
    return df["latent_exact"] if r == "latent" else (df[f"{r}_sel"] + df[f"{r}_ho"]) / 2


def diffs(df, r, by=("writer", "claim")):
    return paired(df.assign(_v=instructed_value(df, r)), "_v", by)


def ci(diff):
    lo, hi = boot(diff, np.mean)
    return float(diff.mean()), float(lo), float(hi)


def as_claims(s: pd.Series, writer="w") -> pd.Series:
    """Index a per-claim series as (writer, claim), the shape ``boot`` resamples."""
    s = s.dropna()
    s.index = pd.MultiIndex.from_arrays([[writer] * len(s), s.index], names=["writer", "claim"])
    return s


def sig_colour(lo, hi):
    return BLUE if lo > 0 else ORANGE if hi < 0 else MUTED


def headline(fig, title, note, legend=None):
    """Title, note and an optional legend row above the axes. Call after ``tight_layout``: they are stacked upward
    from the top of the axes area, and ``bbox_inches="tight"`` keeps them in the saved file."""
    import textwrap

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.get_figheight() * fig.dpi
    top = max(ax.get_tightbbox(r).y1 for ax in fig.axes) / H + 0.01
    if legend is not None:
        handles, labels = legend
        lg = fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.005, top), ncol=len(labels),
                        fontsize=7.5, handletextpad=0.3, columnspacing=1.4, borderaxespad=0)
        top = lg.get_window_extent(r).y1 / H + 0.008
    chars = int(fig.get_figwidth() * 72 / (7.5 * 0.53))
    t = fig.text(0.005, top, textwrap.fill(note, chars), fontsize=7.5, color=INK2, ha="left", va="bottom")
    top = t.get_window_extent(r).y1 / H + 0.008
    fig.text(0.005, top, title, fontsize=10.5, color=INK, fontweight="bold", ha="left", va="bottom")


# ----------------------------------------------------------------------------------------------------- data

def eic_data():
    inst = load(EIC, "instructed-*.jsonl")
    bare = load(EIC, "bare.jsonl").assign(writer="bare")
    committed = inst.groupby(["writer", "claim"]).apply(
        lambda g: (g[g.side == 1].audit == "committed").all() and (g[g.side == 0].audit != "conceded").all())
    return inst, bare, committed[committed].index


def bon_diff(bon, r, k, liar_only=False):
    vals = {}
    for (c, s), g in bon.groupby(["claim", "side"]):
        kk = 1 if (liar_only and s == 0) else k
        vals[(c, s)] = best_of_k(g[f"{r}_sel"], g[value_col(r)], kk)
    t = pd.Series(vals).unstack()
    return as_claims(t[0] - t[1])


def po_tracks(po, r):
    """Per search round: each side's best-so-far reward on the training claims, its held-out reward per claim (the
    current strategy's essays on the other claims), and the held-out paired difference."""
    searched = r.partition(":")[0]
    g = po[po.reward_searched == searched]
    tr, ho = g[g.sampler == "po-train"], g[g.sampler == "po-heldout"]
    rounds = range(int(g["round"].max()) + 1)
    out, held = [], {}
    for side in (0, 1):
        t = tr[tr.side == side].groupby(["round", "strategy"])[f"{r}_sel"].mean().reset_index()
        h = ho[ho.side == side]
        best, last = -np.inf, None
        for rnd in rounds:
            if (t["round"] <= rnd).any():
                best = max(best, t[t["round"] <= rnd][f"{r}_sel"].max())
            cur = h[h["round"] == rnd]
            if len(cur):
                last = cur.groupby("claim")[value_col(r)].mean()
            held[(side, rnd)] = last
            m, lo, hi = ci(as_claims(last.copy()))
            out.append({"side": side, "round": rnd, "train": best, "held": m, "lo": lo, "hi": hi})
    asd = []
    for rnd in rounds:
        m, lo, hi = ci(as_claims(held[(0, rnd)] - held[(1, rnd)]))
        asd.append({"round": rnd, "asd": m, "lo": lo, "hi": hi})
    return pd.DataFrame(out), pd.DataFrame(asd)


# ------------------------------------------------------------------------------------- 1. EIC by sampler

def fig_overview(inst, bare, keep, bon, po):
    samplers = ["No argument: the claim alone", "Instructed, 12 writers", "Instructed, committed liars only",
                "Best-of-4 per side, both sides", "Best-of-4, liar only", "Prompt search, final strategies"]
    rows = []
    for r in REWARDS:
        base = r.replace("debate:", "judge:")  # debate's no-argument baseline: the same judge on the bare claim
        d_all = diffs(inst, r)
        cells = {samplers[0]: diffs(bare, base, ("writer", "claim")) if r == base else None,
                 samplers[1]: d_all, samplers[2]: d_all[d_all.index.isin(keep)],
                 samplers[3]: bon_diff(bon, r, 4), samplers[4]: bon_diff(bon, r, 4, liar_only=True)}
        if r in ("latent", "judge:openai/gpt-4o-mini", "debate:openai/gpt-4o-mini"):
            cells[samplers[5]] = as_claims(_po_final(po, r))
        for s, d in cells.items():
            if d is not None and len(d):
                m, lo, hi = ci(d)
                rows.append({"reward": r, "sampler": s, "asd": m, "lo": lo, "hi": hi})
        if r != base:  # show the judge's no-argument baseline in the debate panels too, as a reference line
            m, lo, hi = ci(diffs(bare, base))
            rows.append({"reward": r, "sampler": "_bare_ref", "asd": m, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(REWARDS), figsize=(2.75 * len(REWARDS), 3.1), sharey=True)
    y = np.arange(len(samplers))[::-1]
    for ax, r in zip(axes, REWARDS):
        st.axes(ax, grid="x")
        g = df[df.reward == r].set_index("sampler")
        for yi, s in zip(y, samplers):
            if s in g.index:
                x = g.loc[s]
                st.dot_ci(ax, x.asd, yi, x.lo, x.hi, sig_colour(x.lo, x.hi))
            else:
                ax.text(0.04, yi, "not run" if "Prompt" in s else "n/a (no argument to debate)",
                        transform=ax.get_yaxis_transform(), color=CONTEXT, fontsize=6.8, va="center")
        ref = g.loc["_bare_ref"].asd if "_bare_ref" in g.index else g.loc[samplers[0]].asd
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (3, 2)), zorder=0.5)
        ax.text(ref, 1.0, " claim alone", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2,
                va="bottom", ha="center")
        st.zero(ax)
        ax.set_title(NAME[r], fontsize=8.8, pad=14)
        ax.set_xlabel(f"ASD ({unit(r)})", fontsize=7.5)
        lo, hi = min(g.lo.min(), 0), max(g.hi.max(), 0)
        ax.set_xlim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
    axes[0].set_yticks(y, samplers)
    fig.tight_layout()
    headline(fig, "The latent score pays for arguing the truth under every sampler; an argued essay erases most of "
                  "the judges' preference for truth",
             "ASD: reward for arguing the true side of a planted-world claim minus the false side, paired by claim "
             "(63 claims, 16 worlds), in each reward's units. 95% claim-bootstrap intervals. Blue: interval above 0; "
             "gray: overlaps 0; orange: below. Dashed: the judge rating the bare claim with no essay (for debate, the "
             "same judge). Best-of-n and prompt search: gpt-4.1-mini writer, selected on one reward draw, scored on "
             "a fresh one (latent: its exact expectation).")
    st.save(fig, OUT, "fig1_eic_by_sampler")
    return df


def _po_final(po, r):
    searched = r.partition(":")[0]
    g = po[(po.reward_searched == searched) & (po.sampler == "po-heldout")]
    held = {}
    for side in (0, 1):
        h = g[g.side == side]
        last = h[h["round"] == h["round"].max()]
        held[side] = last.groupby("claim")[value_col(r)].mean()
    return held[0] - held[1]


# ------------------------------------------------------------------------------- 2. by writer intelligence

def _fit(ax, x, y, colour_):
    b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    rng = np.random.default_rng(1)
    bands, slopes = [], []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0:
            continue
        bb = np.polyfit(x[i], y[i], 1)
        bands.append(np.polyval(bb, xs))
        slopes.append(bb[0] * 10)
    lo, hi = np.percentile(bands, [2.5, 97.5], axis=0)
    ax.fill_between(xs, lo, hi, color=colour_, alpha=0.10, lw=0, zorder=1)
    ax.plot(xs, np.polyval(b, xs), color=colour_, lw=2, zorder=2)
    return b[0] * 10, *np.percentile(slopes, [2.5, 97.5])


def fig_writers(inst, bare):
    rows = []
    for r in REWARDS:
        base = r.replace("debate:", "judge:")
        b = diffs(bare, base).droplevel("writer")
        for w, g in inst.groupby("writer"):
            name, aa = WRITERS[w]
            d = diffs(g, r)
            gain = (d.droplevel("writer") - b.reindex(d.index.get_level_values("claim")).values).dropna()
            m, lo, hi = ci(d)
            gm, glo, ghi = ci(as_claims(gain, w))
            rows.append({"reward": r, "writer": name, "aa": aa, "asd": m, "lo": lo, "hi": hi, "gain": gm,
                         "glo": glo, "ghi": ghi})
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "by_writer_native.csv", index=False)
    fig, axes = plt.subplots(2, len(REWARDS), figsize=(3.6 * len(REWARDS), 7.4), sharex=True)
    for j, r in enumerate(REWARDS):
        p = per[per.reward == r].sort_values("aa")
        base = r.replace("debate:", "judge:")
        bm, blo, bhi = ci(diffs(bare, base))
        for i, (col, lo, hi) in enumerate((("asd", "lo", "hi"), ("gain", "glo", "ghi"))):
            ax = axes[i, j]
            st.axes(ax)
            c = colour(r)
            if i == 0:
                ax.axhspan(blo, bhi, color=CONTEXT, alpha=0.35, lw=0, zorder=0)
                ax.axhline(bm, color=INK2, lw=1.2, ls=(0, (4, 2)), zorder=0.5)
            s, slo, shi = _fit(ax, p.aa.to_numpy(), p[col].to_numpy(), c)
            for _, x in p.iterrows():
                st.dot_ci(ax, x.aa, x[col], x[lo], x[hi], c, size=6, horizontal=False)
                ax.annotate(x.writer, (x.aa, x[col]), xytext=(4, 0), textcoords="offset points", fontsize=5.8,
                            color=INK2, va="center")
            st.zero(ax, orient="h")
            ax.text(0.98, 0.03 if i == 0 else 0.97, f"slope per 10 index points\n{s:+.3f} [{slo:+.3f}, {shi:+.3f}]",
                    transform=ax.transAxes, ha="right", va="bottom" if i == 0 else "top", fontsize=6.6, color=INK2,
                    bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=CONTEXT, lw=0.6))
            if i == 0:
                ax.set_title(NAME[r], fontsize=9)
            else:
                ax.set_xlabel("writer: Artificial Analysis Intelligence Index", fontsize=7.5)
        axes[0, j].set_ylabel(f"ASD ({unit(r)})" if j == 0 or unit(r) != unit(REWARDS[j - 1]) else "")
        axes[1, j].set_ylabel(f"essay minus claim alone ({unit(r)})" if j == 0 or unit(r) != unit(REWARDS[j - 1])
                              else "")
    fig.tight_layout()
    headline(fig, "Arguing erases the judges' preference for truth, at every writer strength; the latent score's "
                  "survives it",
             "Top: ASD per writer (true side minus false side, 63 planted-world claims), 95% claim-bootstrap interval; "
             "line and band: least-squares fit and its 95% band (resampling writers). Dashed with gray band: the same "
             "judge shown only the claim (for debate, its judge). Bottom: each writer's ASD minus the claim-alone ASD, "
             "claim by claim; below 0 means the essays moved the reward away from the truth. Prompted writers, "
             "including the 6% who conceded or evaded.")
    st.save(fig, OUT, "fig2_eic_by_writer")
    return per


# ------------------------------------------------------------------------------------------ 3. best-of-n

def fig_bon(bon):
    ks = range(1, int(bon["sample"].max()) + 2)
    fig, axes = plt.subplots(1, len(REWARDS), figsize=(2.9 * len(REWARDS), 3.0))
    rows = []
    for ax, r in zip(axes, REWARDS):
        st.axes(ax)
        for liar, c, nm in ((False, BLUE, "both"), (True, ORANGE, "liar only")):
            vals = [ci(bon_diff(bon, r, k, liar_only=liar)) for k in ks]
            m, lo, hi = map(np.array, zip(*vals))
            st.line_ci(ax, list(ks), m, lo, hi, c)
            st.end_label(ax, ks[-1], m[-1], nm)
            rows += [{"reward": r, "mode": nm, "k": k, "asd": a, "lo": l, "hi": h} for k, a, l, h in zip(ks, m, lo, hi)]
        st.zero(ax, orient="h")
        ax.set_xticks(list(ks))
        ax.set_xlim(0.7, ks[-1] + 1.1)
        ax.set_title(NAME[r], fontsize=8.8)
        ax.set_xlabel("best-of-n per side, on this reward")
        ax.set_ylabel(f"ASD ({unit(r)})", fontsize=7.5)
    fig.tight_layout()
    headline(fig, "Under best-of-n only the latent score keeps a clear incentive for truth, even when only the "
                  "liar is optimized",
             "gpt-4.1-mini writer, 4 essays per side on each of 63 claims. Blue: both sides optimized; orange: only the "
             "liar. Selected on one reward draw, scored on an independent one (latent: its exact expectation). 95% "
             "claim-bootstrap bands.")
    st.save(fig, OUT, "fig3_eic_best_of_n")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- 4. prompt search

def fig_po(po):
    searched = ["latent", "judge:openai/gpt-4o-mini", "debate:openai/gpt-4o-mini"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    summary = []
    for j, r in enumerate(searched):
        tracks, asd = po_tracks(po, r)
        ax = axes[0, j]
        st.axes(ax)
        ends = []
        for side, c, nm in ((0, BLUE, "honest"), (1, ORANGE, "deceptive")):
            t = tracks[tracks.side == side]
            ax.plot(t["round"], t.train, color=c, ls=(0, (4, 2)), lw=1.6, zorder=2)
            st.line_ci(ax, t["round"], t.held, t.lo, t.hi, c)
            ends += [(t.held.iloc[-1], f"{nm}, held-out"), (t.train.iloc[-1], f"{nm}, training")]
        st.spread_labels(ax, asd["round"].max() + 0.15, [e[0] for e in ends], [e[1] for e in ends], min_gap=0.08)
        ax.set_title(f"Searching against: {NAME[r]}", fontsize=8.8)
        ax.set_ylabel(f"mean reward ({unit(r)})", fontsize=7.5)
        ax.set_xlim(-0.3, asd["round"].max() + 1.9)
        ax = axes[1, j]
        st.axes(ax)
        st.line_ci(ax, asd["round"], asd.asd, asd.lo, asd.hi, colour(r))
        st.zero(ax, orient="h")
        ax.set_ylabel(f"held-out ASD ({unit(r)})", fontsize=7.5)
        ax.set_xlabel("search round")
        ax.set_xlim(-0.3, asd["round"].max() + 1.9)
        for a in axes[:, j]:
            a.set_xticks(list(asd["round"]))
        summary.append(asd.assign(reward=r))
    fig.tight_layout()
    headline(fig, "Prompt search shrinks the latent score's incentive for truth by four-fifths and flips debate's; "
                  "the weak judge's ends positive",
             "Claude Opus 5.5 searches 4 rounds x 4 strategies per side for the gpt-4.1-mini writer, keeping the best "
             "strategy on 16 training claims (dashed: the reward it climbed). Solid, with 95% claim-bootstrap band: "
             "that strategy's essays on the 47 held-out claims, scored on fresh reward draws (latent: exact "
             "expectation). Bottom: held-out ASD, honest minus deceptive.")
    st.save(fig, OUT, "fig4_eic_prompt_search")
    return pd.concat(summary)


# ------------------------------------------------------------------------------------- 5. claimed scope

def fig_scope():
    s = pd.read_csv(EIC / "shift.csv")
    s = s[s.kind != "mine"]
    rows = []
    for (rw, stage), g in s.groupby(["reward_searched", "stage"]):
        for col, nm in (("latent_exact", "questions in the notebook's range"),
                        ("shifted_exact", "questions in the claimed scope")):
            if g[col].notna().any():
                t = g.groupby(["claim", "side"])[col].mean().unstack()
                m, lo, hi = ci(as_claims(t[0] - t[1]))
                rows.append({"searched": rw, "stage": stage, "where": nm, "asd": m, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    order = [("latent", "round 0"), ("latent", "final"), ("judge", "final"), ("debate", "final")]
    labels = ["Unoptimized (round 0)", "Final, searched against the latent score",
              "Final, searched against the judge", "Final, searched against debate"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    st.axes(ax, grid="x")
    y = np.arange(len(order))[::-1]
    for yi, (rw, stg) in zip(y, order):
        for k, (nm, c) in enumerate((("questions in the notebook's range", MUTED),
                                     ("questions in the claimed scope", BLUE))):
            x = df[(df.searched == rw) & (df.stage == stg) & (df["where"] == nm)]
            if len(x):
                x = x.iloc[0]
                st.dot_ci(ax, x.asd, yi + (0.14 if k == 0 else -0.14), x.lo, x.hi, c,
                          label=nm if yi == y[0] else None)
    st.zero(ax)
    ax.set_yticks(y, labels)
    ax.set_xlabel("latent ASD, honest minus deceptive (nats/question, exact expectation)")
    ax.set_ylim(-0.6, len(order) - 0.4)
    fig.tight_layout()
    headline(fig, "Scoring where an essay claims to apply prices the unfalsifiable lie",
             "The latent-searched liar asserts the false position only for unrecorded mines and unseen facet counts. "
             "Held-out essays re-scored on 24 questions about an unseen mine with 10-12 facets (gray: the notebook's "
             "range, which the search optimized). Mine claims excluded. 36 claims; 95% claim-bootstrap intervals.",
             legend=ax.get_legend_handles_labels())
    st.save(fig, OUT, "fig5_eic_claimed_scope")
    return df


# ------------------------------------------------------------------------------ 6. measures of value

MEASURES = [("eval_test", "Held-out questions, same reader"), ("eval_reader_b", "Held-out questions, held-out reader"),
            ("eval_shift", "Shifted questions (unseen mine)"), ("claims", "Checkable claims it gets right"),
            ("pair_truth", "Opus 5.5 shown the truth, pairwise"), ("pair", "Opus 5.5, blind, pairwise"),
            ("eval_rating", "Sonnet 4.5 rubric, 1-10"), ("numbers", "Numbers per 100 words"),
            ("words", "Length in words")]
# row groups: outcomes the report predicts; a stronger model's judgment; style
GROUPS = [("grounded in outcomes", 0, 4), ("a stronger model's judgment", 4, 7), ("style", 7, 9)]
SERIES6 = [("latent", "Latent score (S2, mixture)", BLUE, True), ("s1", "Latent score (S1, whole report)", BLUE, False),
           ("judge", "Judge (gpt-4o-mini)", ORANGE, True), ("debate", "Debate (gpt-4o-mini judge)", AQUA, True)]


def topic_spearman(df, a, b, rng, n=2000):
    rho = []
    for _, g in df.groupby("topic"):
        g = g[[a, b]].dropna()
        if len(g) >= 5 and g[a].nunique() > 1 and g[b].nunique() > 1:
            rho.append(spearmanr(g[a], g[b]).statistic)
    rho = np.array(rho)
    if not len(rho):
        return np.nan, np.nan, np.nan, 0
    bs = [rng.choice(rho, len(rho)).mean() for _ in range(n)]
    return rho.mean(), *np.percentile(bs, [2.5, 97.5]), len(rho)


def fig_measures(inst):
    m = pd.read_csv(BON / "measures.csv")
    s1 = pd.read_csv(BON / "s1.csv")[["topic", "i", "s1"]]
    df = m.merge(s1, on=["topic", "i"])
    df["group"] = np.where(df.topic.str.startswith("planted"), "planted", "testbed")
    rng = np.random.default_rng(0)
    rows = []
    for grp, g in df.groupby("group"):
        for col, _ in MEASURES:
            if grp == "testbed" and col in ("eval_shift", "claims"):
                continue  # held-out factor exists for 2 testbed topics only; checkable claims exist for planted only
            for key, *_ in SERIES6:
                r, lo, hi, n = topic_spearman(g, key, col, rng)
                rows.append({"group": grp, "measure": col, "reward": key, "rho": r, "lo": lo, "hi": hi, "topics": n})
    tab = pd.DataFrame(rows)
    # the incentive study's measure of value: whether the essay argues the true side (per claim, across 24 essays)
    eic = []
    for r in REWARDS:
        rho = []
        for _, g in inst.assign(_v=instructed_value(inst, r)).groupby("claim"):
            if g._v.nunique() > 1:
                rho.append(spearmanr(g._v, 1 - g.side).statistic)
        rho = np.array(rho)
        rho = rho[np.isfinite(rho)]
        bs = [rng.choice(rho, len(rho)).mean() for _ in range(2000)]
        eic.append({"reward": r, "rho": rho.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
    eic = pd.DataFrame(eic)
    tab.to_csv(OUT / "value_measures.csv", index=False)
    eic.to_csv(OUT / "value_truth_eic.csv", index=False)

    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.78], wspace=0.08)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    labels = [nm for _, nm in MEASURES]
    y = np.arange(len(MEASURES))[::-1].astype(float)
    off = np.linspace(0.27, -0.27, len(SERIES6))
    for ax, grp, title in ((axes[0], "planted", "Planted worlds (8 topics, true theory known)"),
                           (axes[1], "testbed", "Testbed families (11 topics)")):
        st.axes(ax, grid="x")
        for gname_, a, b in GROUPS:
            if gname_ != "grounded in outcomes":
                ax.axhline(y[a] + 0.5, color=CONTEXT, lw=0.8, zorder=0)
            if grp == "planted":
                ax.text(-0.74, y[a] + 0.38, gname_.upper(), fontsize=6.2, color=MUTED, va="center", ha="left")
        for yi, (col, _) in zip(y, MEASURES):
            for o, (key, nm, c, filled) in zip(off, SERIES6):
                x = tab[(tab.group == grp) & (tab.measure == col) & (tab.reward == key)]
                if len(x) and np.isfinite(x.rho.iloc[0]):
                    x = x.iloc[0]
                    ax.hlines(yi + o, x.lo, x.hi, color=c, lw=1.4, zorder=2)
                    ax.plot(x.rho, yi + o, "o", ms=6, color=c, mfc=c if filled else SURFACE, mec=c if not filled
                            else SURFACE, mew=1.6 if not filled else 1.2, zorder=3,
                            label=nm if (grp == "planted" and yi == y[0]) else None)
            if not len(tab[(tab.group == grp) & (tab.measure == col)]):
                ax.text(0.5, yi, "not available", transform=ax.get_yaxis_transform(), ha="center", va="center",
                        color=CONTEXT, fontsize=7)
        st.zero(ax)
        ax.set_xlim(-0.75, 0.95)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Spearman correlation with the measure\n(per topic, mean over topics)", fontsize=7.5)
    axes[0].set_yticks(y, labels)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    ax = fig.add_subplot(gs[0, 2])
    st.axes(ax, grid="x")
    ye = np.arange(len(REWARDS))[::-1]
    for yi, (_, x) in zip(ye, eic.iterrows()):
        c = colour(x.reward)
        filled = "4o-mini" not in x.reward
        ax.hlines(yi, x.lo, x.hi, color=c, lw=1.4)
        ax.plot(x.rho, yi, "o", ms=7, color=c, mfc=c if filled else SURFACE, mec=c if not filled else SURFACE,
                mew=1.6)
    st.zero(ax)
    ax.set_yticks(ye, [NAME[r] for r in REWARDS])
    ax.yaxis.tick_right()
    ax.set_xlim(-0.1, 0.75)
    ax.set_title("Does it argue the true side?", fontsize=9)
    ax.set_xlabel("Spearman correlation with truth of the argued\nside (per claim, 12 writers x 2 sides)",
                  fontsize=7.5)
    headline(fig, "Measures grounded in outcomes move with the latent score; a stronger model's judgment is mixed; "
                  "the judge follows style",
             "Left, middle: 12 reports per topic from the best-of-12 arena; rank correlation, within a topic, between "
             "each reward and each measure, averaged over topics, with 95% topic-bootstrap intervals. The Sonnet "
             "rubric and Opus 'shown the truth' see the true theory (planted) or 100 further resolved questions "
             "(testbed). The first row shares the latent score's reader (different questions), so part "
             "of its correlation is mechanical; the held-out reader row is the clean version. Right: the incentive "
             "study's essays (63 claims).", legend=axes[0].get_legend_handles_labels())
    st.save(fig, OUT, "fig6_value_measures")
    return tab, eic


# ------------------------------------------------------------------------------- 7. arena best-of-k

def fig_arena_bon():
    df = pd.read_csv(BON / "s1.csv")
    df["group"] = np.where(df.topic.str.startswith("planted"), "planted", "testbed")
    series = [("latent", "latent (S2)", BLUE, "-"), ("s1", "latent (S1)", BLUE, (0, (3, 2))),
              ("judge", "judge", ORANGE, "-"), ("debate", "debate", AQUA, "-"), ("oracle", "oracle", MUTED, (0, (1, 2)))]
    evals = [("eval_test", "Held-out questions"), ("eval_reader_b", "Held-out reader"),
             ("eval_shift", "Shifted questions")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for i, (grp, gname) in enumerate((("planted", "Planted worlds (8)"), ("testbed", "Testbed families (11)"))):
        g = df[df.group == grp]
        for j, (ev, ename) in enumerate(evals):
            ax = axes[i, j]
            st.axes(ax)
            if grp == "testbed" and ev == "eval_shift":
                ax.text(0.5, 0.5, "a held-out factor exists\nfor 2 of 11 topics only", transform=ax.transAxes,
                        ha="center", va="center", color=CONTEXT, fontsize=8)
                ax.set_xticks([]), ax.set_yticks([])
                continue
            ends = []
            for key, nm, c, ls in series:
                gains = []
                for _, t in g.groupby("topic"):
                    t = t.dropna(subset=[key, ev])
                    n = len(t)
                    gains.append([best_of_k(t[key], t[ev], k) - t[ev].mean() for k in range(1, n + 1)][:12])
                L = min(len(x) for x in gains)
                G = np.array([x[:L] for x in gains])
                bs = np.array([G[rng.integers(0, len(G), len(G))].mean(0) for _ in range(1000)])
                ks = np.arange(1, L + 1)
                ax.fill_between(ks, *np.percentile(bs, [2.5, 97.5], axis=0), color=c, alpha=0.07 if key != "oracle"
                                else 0, lw=0)
                ax.plot(ks, G.mean(0), color=c, ls=ls, lw=2 if key != "oracle" else 1.4)
                ends.append((G.mean(0)[-1], nm))
            st.spread_labels(ax, L + 0.2, [e[0] for e in ends], [e[1] for e in ends], min_gap=0.075)
            st.zero(ax, orient="h")
            ax.set_xlim(0.8, 15.5)
            ax.set_xticks([1, 4, 8, 12])
            if i == 0:
                ax.set_title(ename, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{gname}\ngain over a random report (nats/question)", fontsize=7.5)
            if i == 1:
                ax.set_xlabel("best of k reports, picked by each reward")
    axes[0, 2].set_xlabel("best of k reports, picked by each reward")
    fig.tight_layout()
    headline(fig, "On testbed families picking by the latent score helps and picking by the judge hurts; on planted "
                  "worlds 16 resolved questions are too few to separate them",
             "gpt-4.1-mini writes 12 reports per topic; each reward picks the best of k (exact order statistics), "
             "scored by what no reward saw. 95% topic-bootstrap bands for the four rewards. The oracle picks by the "
             "held-out-questions score itself, so it is that column's ceiling and only another selector elsewhere. "
             "Latent rewards use 16 resolved questions (too few on planted worlds; see the reward-size table).")
    st.save(fig, OUT, "fig7_arena_best_of_k")


# ------------------------------------------------------------------------------ 8. in-context search

def fig_arena_search():
    df = pd.read_csv(ITER / "iterate_curves.csv")
    rng = np.random.default_rng(0)
    chains = [("latent", "latent", BLUE), ("judge", "judge", ORANGE), ("debate", "debate", AQUA)]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), sharex=True)
    for j, (grp, gname) in enumerate((("planted", "Planted worlds (8)"), ("testbed", "Testbed families (11)"))):
        for i, (col, ylab) in enumerate((("new_reward_z", "each chain's own reward\n(s.d. of round 0)"),
                                         ("new_eval", "held-out questions\n(nats/question vs round 0)"))):
            ax = axes[i, j]
            st.axes(ax)
            ends = []
            for key, nm, c in chains:
                g = df[(df.group == grp) & (df.chain == key)]
                piv = g.pivot(index="topic", columns="round", values=col)
                bs = np.array([piv.iloc[rng.integers(0, len(piv), len(piv))].mean() for _ in range(1000)])
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
                st.line_ci(ax, piv.columns.astype(int), piv.mean(), lo, hi, c)
                ends.append((piv.mean().iloc[-1], nm))
            st.spread_labels(ax, 3.15, [e[0] for e in ends], [e[1] for e in ends], min_gap=0.08)
            st.zero(ax, orient="h")
            ax.set_xlim(-0.2, 3.9)
            ax.set_xticks([0, 1, 2, 3])
            if i == 0:
                ax.set_title(gname, fontsize=9)
            else:
                ax.set_xlabel("search round")
            if j == 0:
                ax.set_ylabel(ylab, fontsize=7.5)
    fig.tight_layout()
    headline(fig, "Searching against each reward raises it; the latent chain's reports improve most on held-out "
                  "questions, debate's not at all",
             "In-context expert iteration: each round gpt-4.1-mini sees its best previous reports with their scores "
             "under one reward and writes 8 new ones. Mean over the round's new reports; 95% topic-bootstrap bands. "
             "Top: the reward being searched (z-scored on round 0 per topic). Bottom: held-out value, which no chain "
             "saw.")
    st.save(fig, OUT, "fig8_arena_search")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inst, bare, keep = eic_data()
    bon = load(EIC, "bon-*.jsonl")
    po = load(EIC, "po-*.jsonl")
    ov = fig_overview(inst, bare, keep, bon, po)
    print(ov.round(3).to_string())
    per = fig_writers(inst, bare)
    print(per.groupby("reward")[["asd", "gain"]].mean().round(3))
    print(fig_bon(bon).round(3).to_string())
    print(fig_po(po).round(3).to_string())
    print(fig_scope().round(3).to_string())
    tab, eic = fig_measures(inst)
    print(tab.round(2).to_string())
    print(eic.round(3).to_string())
    fig_arena_bon()
    fig_arena_search()


if __name__ == "__main__":
    main()
