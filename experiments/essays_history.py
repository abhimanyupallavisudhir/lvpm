"""E2: real essays scored on real questions created after them (notes/latent-essays-plan.md).

Essays: LessWrong posts from Sep 2025 - Feb 2026 (experiments/essays_history_data.py), truncated to 3,000 words.
Questions: resolved Manifold binary markets (>= 15 traders) created after the essay, in the essay's topic
(20 per essay), plus 10 from other topics (placebo). The reader's training cutoff (about July 2025) precedes
every essay and every outcome.

Per essay, the frozen reader gives:
* base  - the essay's date alone ("Written on ..."): the reader's own forecast, with the same date information;
* S1    - the whole essay;
* S2    - the essay parsed into up to 4 weighted scenarios, read once per scenario (essay + "assume this
          scenario"), combined by S_tau;
and a strong model rates the essay (S0, the judge).

Outputs in results/essays_history/. Analysis: experiments/essays_history_report.py.

    PYTHONPATH=. python experiments/essays_history.py [--limit N]
"""

import argparse
import asyncio
import collections
import datetime as dt
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lvpm.essays import llmconf as C

from cvtestbed.llm import default_llm

from lvpm.essays.reader import Reader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "essays_history"
N_TOPIC, N_PLACEBO = 20, 10
PARSE = ("Read this essay (posted {date}). Does it describe distinct possible futures or scenarios with different "
         "implications? If so, list up to 4 mutually exclusive scenarios, each described in one or two sentences "
         "concrete enough to answer forecasting questions with, and the probability the essay gives or implies "
         "for each. If not, give one scenario that summarizes the essay's view of what will happen. Reply with "
         "only JSON: {{\"scenarios\": [{{\"description\": \"...\", \"weight\": p}}, ...]}}.\n\nESSAY:\n{text}")
JUDGE = ("Here is an essay posted on {date}.\n\nTITLE: {title}\n\n{text}\n\nRate its quality as analysis of how "
         "things will develop: insight, soundness, calibration and usefulness for forecasting. End with a line "
         "'Score: N' where N is from 1 to 10.")


@dataclass(frozen=True)
class MQ:
    id: str
    text: str
    answer: int  # 0 = YES
    options: tuple = ("Yes", "No")


def select_essays(essays):
    """Zvi's posts (every other one), every other author with 2+ essays, strong single-essay AI posts, and all
    non-AI essays."""
    essays = sorted(essays, key=lambda e: e["date"])
    by = collections.Counter(e["author"] for e in essays)
    zvi = [e for e in essays if e["author"] == "Zvi"][::2]
    repeat = [e for e in essays if e["author"] != "Zvi" and by[e["author"]] >= 2]
    nonai = [e for e in essays if e["topic"] != "ai" and e not in repeat]
    single = [e for e in essays if e["author"] != "Zvi" and by[e["author"]] == 1 and e["topic"] == "ai"
              and e["forward"] == 3]
    random.Random(0).shuffle(single)
    return zvi + repeat + nonai + single[:max(0, 75 - len(zvi) - len(repeat) - len(nonai))]


def sample_questions(essay, pool, now):
    t0 = dt.datetime.fromisoformat(essay["date"]).timestamp() * 1000 + 86_400_000
    ok = [q for q in pool if q["created"] > t0 and q["resolved"] and q["resolved"] < now]
    on = [q for q in ok if q.get("topic") == essay["topic"]]
    off = [q for q in ok if q.get("topic") in ("ai", "us_politics", "geopolitics", "economy")
           and q.get("topic") != essay["topic"]]
    rng = random.Random(f"q|{essay['id']}")
    return rng.sample(on, min(N_TOPIC, len(on))), rng.sample(off, min(N_PLACEBO, len(off)))


def mq(q):
    return MQ(q["id"], q["question"], 0 if q["resolution"] == "YES" else 1)


async def chat(model, prompt, key, max_tokens=2000):
    r = await default_llm().chat(model, prompt, max_tokens=max_tokens, temperature=0.0, key=key)
    return r.text


def scenarios(text):
    for cand in reversed(re.findall(r"\{.*\}", text or "", flags=re.S)):
        try:
            sc = json.loads(cand).get("scenarios", [])
            sc = [(str(s["description"]).strip(), max(0.0, float(s.get("weight", 1.0)))) for s in sc][:4]
            tot = sum(w for _, w in sc)
            if sc and tot > 0:
                return [(d, w / tot) for d, w in sc]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            continue
    return None


async def run_essay(e, pool, now, reader, rows, recs):
    on, off = sample_questions(e, pool, now)
    qs = [mq(q) for q in on + off]
    kind = {q["id"]: "topic" for q in on} | {q["id"]: "placebo" for q in off}
    head = f"(Written on {e['date']}.)\n\n"
    essay = head + e["text"]
    parse_text, judge_text = await asyncio.gather(
        chat(C.READER, PARSE.format(date=e["date"], text=e["text"]), "parse"),
        chat(C.STRONG, JUDGE.format(date=e["date"], title=e["title"], text=e["text"]), "judge", 800))
    sc = scenarios(parse_text) or [(None, 1.0)]
    hyps = [("base", 0, 1.0, head.strip(), qs), ("S1", 0, 1.0, essay, qs)]
    topic_qs = [q for q in qs if kind[q.id] == "topic"]
    for k, (d, w) in enumerate(sc):
        text = essay if d is None else essay + f"\n\nASSUME, IN ADDITION, THAT THIS SCENARIO HOLDS: {d}"
        hyps.append(("S2", k, w, text, topic_qs))
    mats = await asyncio.gather(*[reader.matrix([t], q) for *_, t, q in hyps])
    for (cond, k, w, _, q_list), P in zip(hyps, mats):
        for q, p in zip(q_list, P[0]):
            rows.append({"essay": e["id"], "author": e["author"], "date": e["date"], "topic": e["topic"],
                         "condition": cond, "hyp": k, "weight": w, "q": q.id, "kind": kind[q.id], "p": float(p),
                         "y": 1.0 if q.answer == 0 else 0.0})
    m = re.findall(r"score\W{0,8}(\d+(?:\.\d+)?)", judge_text, re.I)
    recs.append({**{k: e[k] for k in ("id", "title", "date", "author", "topic", "karma", "words")},
                 "judge": min(10.0, max(1.0, float(m[-1]))) if m else None,
                 "scenarios": [{"description": d, "weight": w} for d, w in sc], "n_topic_q": len(on)})


async def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    s0 = C.spend()
    essays = select_essays(json.loads((ROOT / "data" / "e2_essays.json").read_text()))[:args.limit]
    pool = json.loads((ROOT / "data" / "manifold_pool.json").read_text())
    now = dt.datetime(2026, 10, 31).timestamp() * 1000
    reader = Reader(C.READER)
    rows, recs = [], []
    for i in range(0, len(essays), 5):
        await asyncio.gather(*[run_essay(e, pool, now, reader, rows, recs) for e in essays[i:i + 5]])
        pd.DataFrame(rows).to_parquet(OUT / "cells.parquet")
        (OUT / "essays.json").write_text(json.dumps(recs, indent=1))
        print(f"{min(i + 5, len(essays))}/{len(essays)} essays; spend so far ${C.spend() - s0:.2f}", flush=True)
    (OUT / "spend.json").write_text(json.dumps({"usd": C.spend() - s0}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(main(ap.parse_args()))
