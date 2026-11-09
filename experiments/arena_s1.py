"""S1 against S2 as rewards for the best-of-12 reports (arena_bon.py).

The reports are three weighted hypotheses. Task #9's names for the two latent scorers:

* **S2** (the arena's ``latent``): one reader call per (hypothesis, question); the report's score is
  log sum_k w_k prod_q p(y_q | h_k)^tau, so the hypotheses act as values of a latent and the score rewards explaining
  which outcomes come true together;
* **S1**: the reader reads the *whole* report (all hypotheses with their credences) once per question; the score is
  sum_q log p(y_q | report). Correlations between questions earn nothing beyond what the reader's per-question
  probabilities carry.

Both use the same 16 reward questions and sampled outcomes. Evaluations: ``eval_test`` (the mixture marginal on held-
out questions, as before), ``eval_s1_test`` (the whole report read on the held-out questions: S1's own notion of
value), ``eval_reader_b`` and, where computed, the measures of arena_measures.py.

    python experiments/arena_s1.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cvtestbed.llm import LLM

from lvpm.essays import arena as A
from lvpm.essays.reader import Reader

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "arena-bon"


def whole(rep):
    return {"texts": [A.report_text(rep)], "weights": [1.0]}


async def main():
    llm = LLM(concurrency=192)
    reader = Reader("openai/gpt-4.1-mini", llm=llm)
    topics = {t.name: t for t in A.topics()}
    reps = [json.loads(x) for x in (DIR / "reports.jsonl").open()]

    async def one(r):
        t = topics[r["topic"]]
        s1, s1_test = await asyncio.gather(A.latent_score(reader, whole(r), t.reward_qs, soft=False),
                                           A.latent_score(reader, whole(r), t.test_qs, t.soft))
        return {"topic": r["topic"], "i": r["i"], "s1": s1, "eval_s1_test": s1_test}

    rows = pd.DataFrame(await asyncio.gather(*[one(r) for r in reps]))
    print(f"reader ${reader.cost:.2f}, truncation {reader.truncation_rate:.4f}")
    d = pd.DataFrame(reps).drop(columns=["texts", "weights"], errors="ignore").merge(rows, on=["topic", "i"])
    m = DIR / "measures.csv"
    if m.exists():
        d = d.merge(pd.read_csv(m)[["topic", "i", "pair", "pair_truth"] +
                                   [c for c in ("claims",) if c in pd.read_csv(m, nrows=1).columns]],
                    on=["topic", "i"], how="left")
    d.to_csv(DIR / "s1.csv", index=False)
    report(d)


def report(d):
    rewards = ["latent", "s1", "judge", "debate"]
    evals = [e for e in ["eval_test", "eval_s1_test", "eval_reader_b", "claims", "pair_truth"] if e in d]
    lines = ["# S1 vs S2 as rewards (best-of-12 reports)", "",
             "`latent` is S2 (one reader call per hypothesis and question, tempered mixture); `s1` reads the whole "
             "report per question. Best-of-12 gain over a random report (nats/question for the eval_ columns; "
             "claims-score and Bradley-Terry units otherwise), mean over topics, 95% bootstrap CI over topics.", ""]
    for group, sel in [("planted worlds", d.topic.str.startswith("planted")),
                       ("testbed families", ~d.topic.str.startswith("planted"))]:
        g = d[sel]
        lines += [f"## {group}", "", "| reward | " + " | ".join(evals) + " | Spearman with eval_test |",
                  "|---" * (len(evals) + 2) + "|"]
        for r in rewards:
            cells = []
            for e in evals:
                if g[e].isna().all():
                    cells.append("")
                    continue
                gains = np.array([A.best_of_k(t[r], t[e], len(t)) - t[e].mean() for t in
                                  (x.dropna(subset=[r, e]) for _, x in g.groupby("topic")) if len(t) > 1])
                bs = [np.random.default_rng(b).choice(gains, len(gains)).mean() for b in range(2000)]
                cells.append(f"{gains.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
            rho = np.nanmean([spearmanr(t[r], t["eval_test"], nan_policy="omit").statistic for _, t in g.groupby("topic")
                              if t[r].nunique() > 1])
            lines.append(f"| {r} | " + " | ".join(cells) + f" | {rho:+.2f} |")
        dif = np.array([A.best_of_k(t["latent"], t["eval_test"], len(t)) - A.best_of_k(t["s1"], t["eval_test"], len(t))
                        for t in (x.dropna(subset=["latent", "s1", "eval_test"]) for _, x in g.groupby("topic"))])
        bs = [np.random.default_rng(b).choice(dif, len(dif)).mean() for b in range(2000)]
        lines += ["", f"S2 minus S1, best-of-12 on eval_test, paired by topic: {dif.mean():+.3f} "
                      f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]", ""]
    (DIR / "s1.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    import sys

    if "--report" in sys.argv:
        report(pd.read_csv(DIR / "s1.csv"))
    else:
        asyncio.run(main())
