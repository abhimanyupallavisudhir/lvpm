"""Other measures of essay value, and whether they agree with the latent reward (best-of-12 reports, arena_bon.py).

None of these is a validated measure of quality; the question is which of them move with the latent incentive and
which move with the judge's. For each report in ``results/arena-bon/reports.jsonl``:

* ``claims``: planted worlds only. Every verifiable claim the world settles (``planted.all_claims``: 21-29 per
  world: colour majorities, links between tests, mine effects, facet trends). gpt-4.1 reads each hypothesis and says
  which side of each claim it implies, if any. Score: credence-weighted (implies true side) - (implies false side),
  averaged over claims; ``claims_stated`` is the credence-weighted share of claims it takes a side on at all;
* ``pair``: Claude Opus 5.5 compares the report with 4 others on the same topic ("which is the better account?"),
  Bradley-Terry strength; ``pair_truth``: the same, shown the true theory (planted) or 100 further resolved questions
  (testbed families) - a frontier model with privileged information as a stand-in for an expert;
* ``numbers`` (quantitative statements per 100 words), ``words``.

Output: ``results/arena-bon/measures.csv`` and ``measures.md`` (per-topic Spearman correlations, mean over topics
with bootstrap CIs).

    python experiments/arena_measures.py
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cvtestbed as cvt
from cvtestbed.domains import planted as P
from cvtestbed.llm import LLM

from lvpm.essays.arena import report_text
from lvpm.essays.theories import variants
from lvpm.essays.writers import _examples_block

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "arena-bon"
CLASSIFIER = "openai/gpt-4.1"
PAIR = "anthropic/claude-opus-5-5"

CLAIMS = ("Here is a hypothesis about a fictional mineralogy:\n\n{hyp}\n\nBelow are pairs of contrary statements "
          "about the same world. For each pair, say which statement the hypothesis states or clearly implies: 'A', "
          "'B', or '-' if it implies neither (it does not address it, or is compatible with both).\n\n{pairs}\n\n"
          "Reply with only a JSON object mapping each pair number to 'A', 'B' or '-'.")
PAIRWISE = ("Two reports propose competing hypotheses, with credences, about what determines outcomes in a domain. "
            "Which is the better account - more correct, capturing the structure that actually determines outcomes, "
            "better calibrated, and more useful for predicting new cases?\n\nDOMAIN:\n{card}\n\n{extra}"
            "REPORT 1:\n{a}\n\nREPORT 2:\n{b}\n\nThink it through, then end with a line 'Better: 1' or 'Better: 2'.")


def numbers_per_100(text):
    return 100 * len(re.findall(r"\d+(?:\.\d+)?%?", text)) / max(1, len(text.split()))


async def claim_score(llm, rep, world):
    cs = P.all_claims(world)
    flips = [random.Random(f"flip|{c['id']}").random() < 0.5 for c in cs]  # which side is shown as A
    pairs = "\n".join(f"{i + 1}. (A) {c['sides'][f]}  (B) {c['sides'][1 - f]}" for i, (c, f) in
                      enumerate(zip(cs, flips)))
    acc = stated = 0.0
    for text, w in zip(rep["texts"], rep["weights"]):
        r = await llm.chat(CLASSIFIER, CLAIMS.format(hyp=text, pairs=pairs), max_tokens=800, temperature=0.0,
                           key="measures-claims")
        try:
            lab = json.loads(r.text[r.text.index("{"): r.text.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            lab = {}
        for i, f in enumerate(flips):
            x = str(lab.get(str(i + 1), "-")).strip().upper()
            if x in ("A", "B"):
                true_shown = "A" if f == 0 else "B"
                acc += w * (1 if x == true_shown else -1) / len(cs)
                stated += w / len(cs)
    return acc, stated


def bradley_terry(n, wins, iters=200):
    """wins: list of (i, j) meaning i beat j. MM algorithm with a small prior (one tie against a virtual peer)."""
    s = np.ones(n)
    w = np.zeros(n)
    for i, _ in wins:
        w[i] += 1
    for _ in range(iters):
        denom = np.full(n, 1.0 / (1 + 1))  # prior: one game against a peer of strength 1
        for i, j in wins:
            denom[i] += 1 / (s[i] + s[j])
            denom[j] += 1 / (s[i] + s[j])
        s = (w + 0.5) / denom
        s /= np.exp(np.mean(np.log(s)))
    return np.log(s)


async def pairwise(llm, topic, reps, truth):
    card = cvt.families()[topic].card
    if truth and topic.startswith("planted/"):
        extra = f"THE TRUE THEORY OF THIS DOMAIN (known to you, not to the authors):\n{variants(int(topic[-1]))['true']}\n\n"
    elif truth:
        qs = cvt.load([topic], n=400)
        random.Random(f"measures|{topic}").shuffle(qs)
        extra = ("RESOLVED QUESTIONS FROM THIS DOMAIN (not seen by the authors), with their correct answers:\n\n"
                 + _examples_block(qs[-100:], limit_chars=40000) + "\n\n")
    else:
        extra = ""
    rng = random.Random(f"pairs|{topic}")
    n = len(reps)
    pairs = set()
    for i in range(n):
        for j in rng.sample([x for x in range(n) if x != i], 2):
            pairs.add((min(i, j), max(i, j)))

    async def one(i, j):
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        r = await llm.chat(PAIR, PAIRWISE.format(card=card, extra=extra, a=report_text(reps[a]), b=report_text(reps[b])),
                           max_tokens=4000, temperature=0.0, key=f"measures-pair|{int(truth)}")
        m = re.findall(r"better\W{0,3}([12])", r.text, re.I)
        if not m:
            return None
        return (a, b) if m[-1] == "1" else (b, a)

    wins = [x for x in await asyncio.gather(*[one(i, j) for i, j in sorted(pairs)]) if x]
    return bradley_terry(n, wins)


async def main():
    llm = LLM(concurrency=64)
    reps = [json.loads(line) for line in (DIR / "reports.jsonl").open()]
    df = pd.DataFrame(reps)
    out = []
    for topic, g in df.groupby("topic", sort=False):
        rs = g.to_dict("records")
        jobs = [pairwise(llm, topic, rs, False), pairwise(llm, topic, rs, True)]
        if topic.startswith("planted/"):
            jobs += [claim_score(llm, r, int(topic.rsplit("_", 1)[1])) for r in rs]
        res = await asyncio.gather(*jobs)
        for idx, r in enumerate(rs):
            row = {k: r[k] for k in ("topic", "i", "latent", "judge", "debate", "eval_test", "eval_reader_b",
                                     "eval_shift", "eval_rating", "words")}
            row["pair"], row["pair_truth"] = res[0][idx], res[1][idx]
            row["numbers"] = numbers_per_100(" ".join(r["texts"]))
            if topic.startswith("planted/"):
                row["claims"], row["claims_stated"] = res[2 + idx]
            out.append(row)
        print(topic, "done", flush=True)
    m = pd.DataFrame(out)
    m.to_csv(DIR / "measures.csv", index=False)
    report(m)


def report(m: pd.DataFrame):
    measures = ["claims", "claims_stated", "pair", "pair_truth", "eval_rating", "numbers", "words"]
    against = ["latent", "judge", "debate", "eval_test", "eval_reader_b"]
    lines = ["# Other measures of report value: per-topic Spearman correlations",
             "", "Mean over topics of the per-topic Spearman correlation (12 reports per topic), 95% bootstrap CI over "
             "topics. `claims` and `claims_stated` exist for the 8 planted worlds only.", ""]
    for group, sel in [("planted worlds", m.topic.str.startswith("planted")),
                       ("testbed families", ~m.topic.str.startswith("planted"))]:
        lines += [f"## {group}", "", "| measure | " + " | ".join(against) + " |", "|---" * (len(against) + 1) + "|"]
        for meas in measures:
            if m.loc[sel, meas].isna().all():
                continue
            cells = []
            for a in against:
                rs = []
                for _, g in m[sel].groupby("topic"):
                    if g[meas].nunique() > 1 and g[a].nunique() > 1:
                        rs.append(spearmanr(g[meas], g[a]).statistic)
                rs = np.array(rs)
                boot = [np.mean(np.random.default_rng(b).choice(rs, len(rs))) for b in range(2000)]
                cells.append(f"{rs.mean():+.2f} [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]")
            lines.append(f"| {meas} | " + " | ".join(cells) + " |")
        lines.append("")
    (DIR / "measures.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
