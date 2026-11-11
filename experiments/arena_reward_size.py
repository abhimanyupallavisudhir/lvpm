"""Does the latent reward get better as more of its questions resolve? (planted worlds, the arena's reports)

The latent reward of the essay arena sees only sampled outcomes of its reward questions, so its noise falls as
1/sqrt(n). For the reports the arena already sampled on planted worlds, recompute S_tau on n = 16, 32, 64, 128
reward questions (fresh crystals, disjoint from the test questions) and report, against the held-out test score:
the best-of-12 gain, the share of the gap to the true theory closed, and the Spearman correlation. Also the same
reward computed on the exact conditionals (no outcome noise) - the limit as n grows.

    PYTHONPATH=. python experiments/arena_reward_size.py --dir results/arena-bon
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cvtestbed as cvt

from lvpm.essays import arena as A
from lvpm.essays.reader import Reader

SIZES = [16, 32, 64, 128]


def reward_pool(k: int, n: int):
    """In-distribution questions for world k beyond those the arena used (it used the first 46 of seed 1)."""
    qs = cvt.load([f"planted/world_{k}"], n=1200, seed=1)
    ins = [q for q in qs if not q.params["shifted"]]
    return ins[46:46 + n]


def boot(x, n=4000):
    x = np.asarray(x, dtype=float)
    m = np.random.default_rng(0).choice(x, (n, len(x))).mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


async def run(d: Path, reader_model: str, concurrency: int):
    df = pd.read_json(d / "reports.jsonl", lines=True)
    refs = pd.read_csv(d / "references.csv").set_index("topic")
    from cvtestbed.llm import LLM

    reader = Reader(reader_model, llm=LLM(concurrency=concurrency))
    rows = []
    for topic, g in df[df.topic.str.startswith("planted/")].groupby("topic"):
        k = int(topic.rsplit("_", 1)[1])
        pool = reward_pool(k, max(SIZES))
        reps = [{"texts": t, "weights": w} for t, w in zip(g.texts, g.weights)]
        test = g.eval_test.to_numpy()
        gap = refs.loc[topic, "true_test"] - test.mean()
        for n in SIZES + ["exact"]:
            qs = pool if n == "exact" else pool[:n]
            rew = np.array(await asyncio.gather(*[A.latent_score(reader, r, qs, soft=(n == "exact")) for r in reps]))
            gain = A.best_of_k(rew, test, len(test)) - test.mean()
            rows.append({"topic": topic, "n_reward": n, "gain": gain, "share_closed": gain / gap,
                         "rho": spearmanr(rew, test)[0]})
        print(f"{topic}: done (reader ${reader.cost:.2f}, truncation {reader.truncation_rate:.3f})", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(d / "reward_size.csv", index=False)
    lines = ["# Latent reward vs number of resolved reward questions (planted worlds)", "",
             f"Reader {reader_model}; {R.topic.nunique()} worlds x 12 reports. `exact`: S on the true conditionals "
             f"of all {max(SIZES)} questions (no outcome noise). 95% bootstrap CIs over worlds.", "",
             "| reward questions | best-of-12 gain on held-out test (nats/q) | share of gap to true theory closed | "
             "Spearman with test |", "|---|---|---|---|"]
    for n in SIZES + ["exact"]:
        s = R[R.n_reward == n]
        lines.append(f"| {n} | " + " | ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*boot(s[c]))
                                              for c in ("gain", "share_closed", "rho")) + " |")
    (d / "reward_size.md").write_text("\n".join(lines))
    (d / "reward_size_meta.json").write_text(json.dumps({"reader": reader_model, "cost": reader.cost,
                                                        "truncation": reader.truncation_rate}))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/arena-bon")
    ap.add_argument("--reader", default="openai/gpt-4.1-mini")
    ap.add_argument("--concurrency", type=int, default=192)
    a = ap.parse_args()
    asyncio.run(run(Path(a.dir), a.reader, a.concurrency))
