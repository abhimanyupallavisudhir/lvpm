"""The essay arena under best-of-n: which reward, when optimized, selects better reports?

For every topic (planted worlds and testbed families, lvpm.essays.arena) the writer samples N reports. Each is scored
by every reward (latent S_tau on reward questions, a weak judge, debate, the oracle) and every evaluation (held-out
test and shifted questions, a second reader, a strong model's rubric). Best-of-k curves are exact order statistics.
References: the reader with no report, and (planted worlds) with the true theory.

    PYTHONPATH=. python experiments/arena_bon.py --n 16 --out results/arena-bon
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cvtestbed.llm import LLM

from lvpm.essays import arena as A
from lvpm.essays.reader import Reader

REWARDS = ["latent", "judge", "debate", "oracle"]
EVALS = ["eval_test", "eval_shift", "eval_reader_b", "eval_rating"]


async def score_report(t, rep, i, args, llm, ra, rb):
    key = f"{t.name}|{i}"
    lat, test, shift, rb_test, jd, db, rub = await asyncio.gather(
        A.latent_score(ra, rep, t.reward_qs, soft=False),  # the mechanism sees only sampled outcomes
        A.latent_score(ra, rep, t.test_qs, t.soft),
        A.latent_score(ra, rep, t.shift_qs, t.soft),
        A.latent_score(rb, rep, t.test_qs, t.soft),
        A.judge_score(args.judge, t, rep, llm, "judge|" + key),
        A.debate_score(args.writer, args.judge, t, rep, llm, "debate|" + key),
        A.rubric_score(args.rubric, t, rep, llm, "rubric|" + key))
    return {"topic": t.name, "i": i, "k_hyps": len(rep["texts"]), "words": sum(len(h.split()) for h in rep["texts"]),
            "latent": lat, "judge": jd, "debate": db, "oracle": test, "eval_test": test, "eval_shift": shift,
            "eval_reader_b": rb_test, "eval_rating": rub, "texts": rep["texts"], "weights": rep["weights"]}


async def references(t, ra, rb):
    out = {}
    empty = {"texts": [""], "weights": [1.0]}
    out["empty_test"] = await A.latent_score(ra, empty, t.test_qs, t.soft)
    out["empty_shift"] = await A.latent_score(ra, empty, t.shift_qs, t.soft)
    out["empty_reader_b"] = await A.latent_score(rb, empty, t.test_qs, t.soft)
    if t.true_theory:
        tru = {"texts": [t.true_theory], "weights": [1.0]}
        out["true_test"] = await A.latent_score(ra, tru, t.test_qs, t.soft)
        out["true_shift"] = await A.latent_score(ra, tru, t.shift_qs, t.soft)
        b = np.array([A._p_a(q) for q in t.test_qs])
        out["bayes_test"] = float(np.mean(b * np.log(np.clip(b, 1e-6, 1)) + (1 - b) * np.log(np.clip(1 - b, 1e-6, 1))))
    return out


async def run(args):
    llm = LLM(concurrency=args.concurrency)
    ra, rb = Reader(args.reader_a, llm=llm), Reader(args.reader_b, llm=llm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, refs = [], []
    planted = range(args.planted) if args.planted else []
    for t in A.topics(planted=planted, testbed=args.testbed):
        reps = await asyncio.gather(*[A.write_report(args.writer, t, i, llm) for i in range(args.n)])
        reps = [(i, r) for i, r in enumerate(reps) if r is not None]
        got = await asyncio.gather(*[score_report(t, r, i, args, llm, ra, rb) for i, r in reps])
        rows += got
        refs.append({"topic": t.name, **await references(t, ra, rb)})
        pd.DataFrame(rows).to_json(out / "reports.jsonl", orient="records", lines=True)
        pd.DataFrame(refs).to_csv(out / "references.csv", index=False)
        print(f"{t.name}: {len(reps)} reports; reader A ${ra.cost:.2f}, reader B ${rb.cost:.2f}", flush=True)
    curves(pd.DataFrame(rows), out)
    (out / "meta.json").write_text(json.dumps({**vars(args), "reader_a_cost": ra.cost, "reader_b_cost": rb.cost,
                                               "reader_a_truncation": ra.truncation_rate,
                                               "reader_b_truncation": rb.truncation_rate}, indent=1))


def curves(df, out):
    rng = np.random.default_rng(0)
    rows = []
    for topic, g in df.groupby("topic"):
        g = g.dropna(subset=["latent", "eval_test"])
        n = len(g)
        for k in range(1, n + 1):
            for r in REWARDS + ["random"]:
                rew = rng.random(n) if r == "random" else g[r].to_numpy()
                for e in EVALS:
                    v = g[e].to_numpy()
                    if np.isnan(v).all():
                        continue
                    rows.append({"topic": topic, "k": k, "reward": r, "eval": e,
                                 "value": A.best_of_k(rew, np.nan_to_num(v, nan=np.nanmean(v)), k)})
    c = pd.DataFrame(rows)
    c.to_csv(out / "curves.csv", index=False)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--planted", type=int, default=8)
    ap.add_argument("--testbed", nargs="*", default=A.DEFAULT_TESTBED)
    ap.add_argument("--writer", default="openai/gpt-4.1-mini")
    ap.add_argument("--reader-a", default="openai/gpt-4.1-mini")
    ap.add_argument("--reader-b", default="anthropic/claude-haiku-4-5")
    ap.add_argument("--judge", default="openai/gpt-4o-mini")
    ap.add_argument("--rubric", default="anthropic/claude-sonnet-4-5")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--out", default="results/arena-bon")
    asyncio.run(run(ap.parse_args()))
