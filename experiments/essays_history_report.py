"""Analysis of E2 (experiments/essays_history.py): writes results/essays_history/summary.md.

Gains are in nats per question, relative to the same reader given only the essay's date ("base").

    PYTHONPATH=. python experiments/essays_history_report.py
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lvpm.essays import score as S

D = Path(__file__).resolve().parents[1] / "results" / "essays_history"
rng = np.random.default_rng(0)


def boot(x, n=4000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return f"{x.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]"


def ll(p, y):
    """Log score of each prediction; keeps the index when given Series."""
    v = np.where(np.asarray(y) == 1, np.log(p), np.log(1 - np.asarray(p)))
    return pd.Series(v, index=p.index) if isinstance(p, pd.Series) else v


def essay_scores(g, qs):
    """Per-question gains over base for S1 and S2 on the questions ``qs``."""
    b = g[(g.condition == "base")].set_index("q")
    s1 = g[(g.condition == "S1")].set_index("q")
    out = {"S1": (ll(s1.p, s1.y) - ll(b.p, b.y)).reindex(qs).mean()}
    s2 = g[g.condition == "S2"]
    if len(s2) and set(qs) <= set(s2.q):
        hyps = sorted(s2.hyp.unique())
        P = np.array([s2[s2.hyp == h].set_index("q").p.reindex(qs).values for h in hyps])
        w = np.array([s2[s2.hyp == h].weight.iloc[0] for h in hyps])
        y = b.y.reindex(qs).values
        out["S2"] = S.s_tau(w, P, y, 1.0) / len(qs) - ll(b.p, b.y).reindex(qs).mean()
    return out


def main():
    d = pd.read_parquet(D / "cells.parquet")
    recs = pd.DataFrame(json.loads((D / "essays.json").read_text())).set_index("id")
    rows = []
    for eid, g in d.groupby("essay"):
        topic = sorted(g[(g.kind == "topic") & (g.condition == "base")].q)
        plac = sorted(g[(g.kind == "placebo") & (g.condition == "base")].q)
        r = {"essay": eid, "author": g.author.iloc[0], "topic": g.topic.iloc[0], "date": g.date.iloc[0],
             "title": recs.loc[eid, "title"], "judge": recs.loc[eid, "judge"],
             "K": len(recs.loc[eid, "scenarios"])}
        sc = essay_scores(g, topic)
        r["S1"], r["S2"] = sc["S1"], sc.get("S2", np.nan)
        r["S1_placebo"] = essay_scores(g, plac)["S1"]
        r["S1_adj"] = r["S1"] - r["S1_placebo"]
        half = random.Random(f"half|{eid}").sample(topic, len(topic) // 2)
        other = [q for q in topic if q not in half]
        for name, qs in (("a", half), ("b", other)):
            hs = essay_scores(g, qs)
            r[f"S1_{name}"], r[f"S2_{name}"] = hs["S1"], hs.get("S2", np.nan)
        rows.append(r)
    E = pd.DataFrame(rows)

    def sb(a, b):  # split-half correlation, Spearman-Brown corrected to the full 20 questions
        x = E[[a, b]].dropna()
        r = np.corrcoef(x[a], x[b])[0, 1]
        return r, 2 * r / (1 + r)

    L = ["# E2: real essays, scored on questions created after them", "",
         f"{len(E)} LessWrong essays (Sep 2025 - Feb 2026; {E.author.nunique()} authors; topics "
         f"{E.topic.value_counts().to_dict()}), truncated to 3,000 words. Each is scored on 20 resolved Manifold "
         "questions in its topic created after it (and 10 placebo questions from other topics). Reader: "
         "claude-haiku-4.5 (training cutoff about July 2025, before every essay and outcome). Gains: nats per "
         "question over the same reader given only the essay's date.", "",
         "## 1. Do essays help the reader?", "", "| score | mean gain over the date-only reader [95% CI] |",
         "|---|---|",
         f"| S1 (whole essay), topic questions | {boot(E.S1)} |",
         f"| S1, placebo questions (other topics) | {boot(E.S1_placebo)} |",
         f"| S1, topic minus placebo | {boot(E.S1_adj)} |",
         f"| S2 (parsed scenarios), topic questions | {boot(E.S2)} |", "",
         f"Parse: {E.K.value_counts().sort_index().to_dict()} essays with K scenarios.", "",
         "## 2. Is the score a stable property of the essay?", ""]
    for s in ("S1", "S2"):
        r, rsb = sb(f"{s}_a", f"{s}_b")
        L.append(f"- {s}: split-half correlation across essays {r:.2f} (Spearman-Brown, 20 questions: {rsb:.2f})")
    multi = E.groupby("author").filter(lambda g: len(g) >= 3)
    for s in ("S1", "S2", "judge"):
        x = multi.dropna(subset=[s])
        between = x.groupby("author")[s].mean().var()
        within = x.groupby("author")[s].var().mean()
        L.append(f"- {s}: share of variance between authors (authors with 3+ essays, n={x.author.nunique()}): "
                 f"{between / (between + within):.2f}")
    L += ["", "## 3. Does the judge agree?", ""]
    for s in ("S1", "S2", "S1_adj"):
        x = E.dropna(subset=[s, "judge"])
        L.append(f"- Spearman(judge, {s}) = {spearmanr(x.judge, x[s]).correlation:.2f} (n={len(x)})")
    E["rank_gap"] = E.S1_adj.rank(pct=True) - E.judge.rank(pct=True)
    L += ["", "Largest disagreements (S1 topic-minus-placebo percentile minus judge percentile):", "",
          "| essay | author | judge | S1 adj. |", "|---|---|---|---|"]
    for _, r in pd.concat([E.nlargest(4, "rank_gap"), E.nsmallest(4, "rank_gap")]).drop_duplicates("essay").iterrows():
        L.append(f"| {r.title[:70]} | {r.author} | {r.judge} | {r.S1_adj:+.3f} |")
    spend = json.loads((D / "spend.json").read_text())["usd"] if (D / "spend.json").exists() else None
    if spend:
        L += ["", f"OpenRouter spend during this run (shared with any concurrent run): ${spend:.2f}."]
    E.to_csv(D / "essays_scored.csv", index=False)
    (D / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
