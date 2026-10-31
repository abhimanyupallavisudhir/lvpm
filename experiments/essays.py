"""Essays as latents, first experiment: do essays carry transferable structure, and is S_tau a good signal?

For each testbed family, items are split three ways (seeded): SHOWN (resolved items writers may read), DEV
(resolved items used only to select among essays) and TEST (held out from everyone until scoring). Reports:

* empty           - no essay: the reader's own predictions
* examples        - the SHOWN items verbatim with their answers (memorized data, no theory)
* uninformed      - an essay from the family card alone
* informed        - an essay from the card plus the SHOWN items (several samples per writer)
* mixture         - three weighted competing hypotheses from the card plus the SHOWN items

Every hypothesis is read by the same frozen reader on DEV and TEST items. Outputs (results/essays/):
cells.parquet (one row per hypothesis x item with p_R), reports.json (texts, weights, costs) and a summary
printed and written to summary.json. Analysis and figures: experiments/essays_report.py.

    PYTHONPATH=. python experiments/essays.py --families rlhack/sensor agents/auction --per-family 60
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cvtestbed as cvt

from lvpm.essays import writers as W
from lvpm.essays.reader import Reader, outcomes, words

OUT = Path(__file__).resolve().parents[1] / "results" / "essays"  # overridden by --out


def three_way(items, seed=0, shown=0.34, dev=0.33):
    items = list(items)
    random.Random(f"split|{seed}").shuffle(items)
    a, b = int(len(items) * shown), int(len(items) * (shown + dev))
    return items[:a], items[a:b], items[b:]


async def family_reports(family, card, shown, writers, n_informed, n_uninformed, n_mixture):
    reports, cost = [], 0.0
    reports.append({"kind": "empty", "writer": None, "sample": 0, "hypotheses": [None], "weights": [1.0]})
    reports.append({"kind": "examples", "writer": None, "sample": 0, "hypotheses": [W.examples_essay(shown)],
                    "weights": [1.0]})
    jobs = []
    for w in writers:
        for s in range(n_uninformed):
            jobs.append(("uninformed", w, s, W.write_essay(w, card, None, sample=s)))
        for s in range(n_informed):
            jobs.append(("informed", w, s, W.write_essay(w, card, shown, sample=s)))
        for s in range(n_mixture):
            jobs.append(("mixture", w, s, W.write_mixture(w, card, shown, sample=s)))
    results = await asyncio.gather(*[j[3] for j in jobs], return_exceptions=True)
    for (kind, w, s, _), res in zip(jobs, results):
        if isinstance(res, Exception):
            print(f"  ! {family} {kind} {w} {s}: {res!r}")
            continue
        out, c = res
        cost += c
        if kind == "mixture":
            if not out:
                print(f"  ! {family} mixture {w} {s}: unparsable")
                continue
            hyps, ws = [t for t, _ in out], [x for _, x in out]
        else:
            if not out:
                continue
            hyps, ws = [out], [1.0]
        reports.append({"kind": kind, "writer": w, "sample": s, "hypotheses": hyps, "weights": ws})
    return reports, cost


async def run_family(family, args, reader):
    family = cvt.ALIASES.get(family, family)
    items = cvt.load([family], n=args.per_family, seed=args.seed)
    shown, dev, test = three_way(items, args.seed)
    card = cvt.families()[family].card
    t0 = time.time()
    reports, wcost = await family_reports(family, card, shown, args.writers, args.n_informed, args.n_uninformed,
                                          args.n_mixture)
    scored = dev + test
    split = {it.id: ("dev" if it in dev else "test") for it in scored}
    y = outcomes(scored)
    rcost0 = reader.cost
    rows = []
    hyps = []
    for ri, r in enumerate(reports):
        for hi, h in enumerate(r["hypotheses"]):
            hyps.append((ri, hi, h))
    P = await reader.matrix([h for _, _, h in hyps], scored)
    for (ri, hi, h), prow in zip(hyps, P):
        for it, p, yi in zip(scored, prow, y):
            rows.append({"family": family, "report": ri, "hyp": hi, "item": it.id, "split": split[it.id],
                         "p": float(p), "y": float(yi)})
    for ri, r in enumerate(reports):
        r.update(family=family, id=ri, words=[words(h) for h in r["hypotheses"]])
    print(f"{family}: {len(reports)} reports, {len(hyps)} hypotheses x {len(scored)} items; writers ${wcost:.2f}, "
          f"reader ${reader.cost - rcost0:.2f}, {time.time() - t0:.0f}s")
    return rows, reports, {"family": family, "n_shown": len(shown), "n_dev": len(dev), "n_test": len(test),
                           "writer_cost": wcost, "reader_cost": reader.cost - rcost0}


async def main(args):
    global OUT
    OUT = Path(args.out)
    reader = Reader(args.reader)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, all_reports, meta = [], [], []
    for fam in args.families:
        rows, reports, m = await run_family(fam, args, reader)
        all_rows += rows
        all_reports += reports
        meta.append(m)
        pd.DataFrame(all_rows).to_parquet(OUT / "cells.parquet")
        (OUT / "reports.json").write_text(json.dumps(all_reports, indent=1))
        (OUT / "meta.json").write_text(json.dumps({"reader": reader.describe(), "args": vars(args), "families": meta},
                                                  indent=1))
    print(f"total reader cost ${reader.cost:.2f} over {reader.calls} calls")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["rewardhacking/gridworld_sensor", "multiagent/auction",
                                                       "sycophancy/wrong_answer", "numbers/prime_race",
                                                       "decisions/newcomb"])
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--writers", nargs="+", default=["openai/gpt-4.1-mini", "openai/gpt-5-mini",
                                                      "anthropic/claude-sonnet-4-5"])
    ap.add_argument("--reader", default="openai/gpt-4.1-mini")
    ap.add_argument("--n-informed", type=int, default=4)
    ap.add_argument("--n-uninformed", type=int, default=2)
    ap.add_argument("--n-mixture", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    asyncio.run(main(ap.parse_args()))
