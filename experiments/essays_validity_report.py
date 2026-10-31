"""Analysis of E0 (experiments/essays_validity.py): writes results/essays_validity/summary.md.

    PYTHONPATH=. python experiments/essays_validity_report.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "essays_validity"
rng = np.random.default_rng(0)


def ll(p, y):
    return np.where(y == 1, np.log(p), np.log(1 - p))


def boot(x, n=4000):
    x = np.asarray(x, float)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main():
    d = pd.read_parquet(D / "cells.parquet")
    d["ll"] = ll(d.p, d.y)
    empty = d[d.essay == "empty"].set_index("item").p
    e = d[d.essay != "empty"]
    W = e.pivot_table(index=["family", "essay", "item"], columns="version", values="p").reset_index()
    W["empty"] = W.item.map(empty)
    paras = [c for c in W.columns if str(c).startswith("paraphrase")]
    reads = [c for c in W.columns if str(c).startswith("reading")]

    # 1. how much each change moves predictions (mean |dp| per essay x item)
    move = {
        "the essay itself (vs. the reader alone)": (W.original - W.empty).abs(),
        "paraphrase (vs. original)": pd.concat([(W[c] - W.original).abs() for c in paras]),
        "another reading of the same text (temperature 0.7, pairwise)":
            pd.concat([(W[a] - W[b]).abs() for a, b in combinations(reads, 2)]),
        "author framed as respected (vs. original)": (W.respected - W.original).abs(),
        "author framed as unreliable (vs. original)": (W.unreliable - W.original).abs(),
        "respected vs. unreliable": (W.respected - W.unreliable).abs(),
    }
    lines = ["# E0: is the reader good enough to be a semantics?", "",
             "Reader: `openrouter/anthropic/claude-haiku-4.5` (reasoning, then a stated probability; both answer "
             "orders). Essays: the first experiment's informed essays, the first essay of two different writers "
             "per family (14 essays), each read on the family's 32 DEV+TEST items.", "",
             "## 1. How far each change moves the reader's predictions", "",
             "Mean |change in P(option A)| per (essay, item), with 95% bootstrap CIs.", "",
             "| change | mean |dp| |", "|---|---|"]
    lines += [f"| {k} | {fmt(boot(v))} |" for k, v in move.items()]

    # 2. evidential leakage: does the 'unreliable' framing pull predictions back toward the reader's prior?
    push = W.original - W.empty
    m = push.abs() > 0.1
    lam = ((W.respected - W.unreliable)[m] / push[m]).clip(-2, 2)
    lines += ["", "## 2. Literal vs. evidential reading", "",
              "Leakage = (P_respected - P_unreliable) / (P_essay - P_reader alone), over cells where the essay "
              f"moves the prediction by more than 0.1 ({m.sum()} cells). 0 = the reader takes the essay "
              "literally; 1 = the 'unreliable' framing fully undoes the essay.", "",
              f"Leakage: {fmt(boot(lam))}"]

    # 3. score invariance under paraphrase: intraclass correlation of per-essay gains
    rows = []
    for (fam, ess), g in e.groupby(["family", "essay"]):
        base = d[(d.essay == "empty") & d.item.isin(g.item.unique())].ll.mean()
        for v in ["original"] + paras:
            rows.append({"essay": ess, "version": v, "gain": g[g.version == v].ll.mean() - base})
    G = pd.DataFrame(rows)
    within = G.groupby("essay").gain.var().mean()
    between = G.groupby("essay").gain.mean().var()
    icc = between / (between + within)
    sd_para = G.groupby("essay").gain.std().mean()
    lines += ["", "## 3. Do paraphrases get the same score?", "",
              "Per-essay gain over the reader alone (mean log score per item, nats), for the original and its 3 "
              "paraphrases.", "",
              f"- intraclass correlation (between-essay variance / total): **{icc:.2f}** (pre-registered pass: > 0.8)",
              f"- mean SD of an essay's gain across its paraphrases: {sd_para:.3f} nats/item; SD of gains between "
              f"essays: {np.sqrt(between):.3f}"]

    # 4. ambiguity: scoring each reading vs. scoring the pooled reading
    amb = []
    for ess, g in W.groupby("essay"):
        y = e[(e.essay == ess) & (e.version == "original")].set_index("item").y.reindex(g.item).values
        R = g[reads].values
        each = np.mean([ll(R[:, j], y).mean() for j in range(R.shape[1])])
        pooled = ll(R.mean(1), y).mean()
        amb.append({"essay": ess, "E_r log": each, "log E_r": pooled, "gap": pooled - each,
                    "sd": R.std(1).mean()})
    A = pd.DataFrame(amb)
    lines += ["", "## 4. Ambiguity", "",
              f"Across 3 readings at temperature 0.7: mean per-cell SD of P {A.sd.mean():.3f}; scoring each reading "
              f"and averaging (E_r log) is below scoring the pooled reading (log E_r) by {A.gap.mean():.3f} "
              f"nats/item on average (max {A.gap.max():.3f}). That gap is the price of ambiguity."]

    # 5. calibration of the reader alone, and agreement with the old reader on essay gains
    cal = []
    for fam, g in d[d.essay == "empty"].groupby("family"):
        cal.append(f"| {fam} | {np.mean((g.p > 0.5) == (g.y == 1)):.2f} | {g.ll.mean():.3f} |")
    lines += ["", "## 5. The reader alone", "", "| family | accuracy | log score (nats/item) |", "|---|---|---|"]
    lines += cal
    old_cells = ROOT / "results" / "essays" / "cells.parquet"
    if old_cells.exists():
        oc = pd.read_parquet(old_cells)
        reps = json.loads((ROOT / "results" / "essays" / "reports.json").read_text())
        rid = {f"{r['family']}|{r['writer']}|{r['sample']}": r["id"] for r in reps if r["kind"] == "informed"}
        emp = {r["family"]: r["id"] for r in reps if r["kind"] == "empty"}
        oc["ll"] = ll(oc.p.clip(0.01, 0.99), oc.y)
        pairs = []
        for ess in G.essay.unique():
            fam = ess.split("|")[0]
            o = oc[(oc.family == fam) & (oc.report == rid[ess])].ll.mean() - \
                oc[(oc.family == fam) & (oc.report == emp[fam])].ll.mean()
            pairs.append((o, G[(G.essay == ess) & (G.version == "original")].gain.iloc[0]))
        P = np.array(pairs)
        r = np.corrcoef(P.T)[0, 1]
        same = sum(len(set(oc[oc.family == f].item) & set(d[d.family == f].item)) == d[d.family == f].item.nunique()
                   for f in d.family.unique())
        lines += ["", f"Essay gains under this reader vs. the first experiment's reader (gpt-4.1-mini), same "
                  f"essays: Pearson r = {r:.2f} over {len(P)} essays. Only {same} of {d.family.nunique()} families "
                  "were read on the same items (the testbed's item ids have changed since); for the rest the "
                  "items are a different draw from the same family, so this mixes reader dependence with item "
                  f"sampling: r = {r:.2f} is a lower bound on how far the two readers agree."]
    spend = json.loads((D / "spend.json").read_text())["usd"] if (D / "spend.json").exists() else None
    if spend:
        lines += ["", f"OpenRouter spend: ${spend:.2f}."]
    (D / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
