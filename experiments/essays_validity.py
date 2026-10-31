"""E0: is the frozen reader good enough to be a semantics? (notes/latent-essays-plan.md)

On the first experiment's informed essays (two per family, from different writers), the reader is asked about the
same DEV+TEST items under controlled changes to the essay:

* paraphrase   - 3 meaning-preserving rewrites (gauge): predictions should not move;
* attribution  - the essay framed as from a respected expert vs. from an unreliable source (literal
                 reading): predictions should not move;
* readings     - 3 independent readings at temperature 0.7 (ambiguity): how much one text's meaning varies.

Also the reader alone (calibration). Outputs in results/essays_validity/: cells.parquet (one row per
version x item), versions.json (all texts), summary.md.

    PYTHONPATH=. python experiments/essays_validity.py [families...]
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lvpm.essays import llmconf as C  # noqa: F401  (keys, prices, cache)

import cvtestbed as cvt
from cvtestbed.llm import default_llm

from lvpm.essays.reader import Reader, outcomes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from essays import three_way  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "essays_validity"
SRC = Path(__file__).resolve().parents[1] / "results" / "essays" / "reports.json"
WRITER_ORDER = ["openai/gpt-5-mini", "anthropic/claude-sonnet-4-5", "openai/gpt-4.1-mini"]
ATTRIB = {
    "respected": "The following theory was written by a leading expert in this area, whose past predictions "
                 "about it have been consistently accurate.\n\n",
    "unreliable": "The following theory was written by an anonymous forum user with a poor track record; "
                  "their past predictions about this area have mostly been wrong.\n\n",
}
PARA = ("Rewrite the following text completely: different wording, different sentence structure and a different "
        "order of points where the order does not matter. Keep every claim, every number, every condition and "
        "exactly the same degree of confidence; add nothing and drop nothing. Reply with the rewritten text "
        "only.\n\nTEXT:\n{essay}")


async def paraphrases(essay, k=3):
    llm = default_llm()
    rs = await asyncio.gather(*[llm.chat(C.STRONG, PARA.format(essay=essay), max_tokens=2000, temperature=1.0,
                                         key=f"para|{i}") for i in range(k)])
    return [r.text.strip() for r in rs]


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s0 = C.spend()
    reports = json.loads(SRC.read_text())
    reader = Reader(C.READER)
    readings = [Reader(C.READER, temperature=0.7, reading=r, llm=reader.llm) for r in (1, 2, 3)]
    rows, versions = [], []
    for family in sys.argv[1:] or dict.fromkeys(r["family"] for r in reports):
        items = cvt.load([cvt.ALIASES.get(family, family)], n=48, seed=0)
        _, dev, test = three_way(items, 0)
        scored = dev + test
        y = outcomes(scored)
        informed = [r for r in reports if r["family"] == family and r["kind"] == "informed"]
        picked = list({r["writer"]: r for r in reversed(informed)}.values())  # first essay of each writer
        picked = sorted(picked, key=lambda r: (r["writer"] not in WRITER_ORDER,
                                               WRITER_ORDER.index(r["writer"]) if r["writer"] in WRITER_ORDER
                                               else 0))[:2]
        jobs = [("empty", None, "original", reader, None)]
        for r in picked:
            essay = r["hypotheses"][0]
            eid = f"{family}|{r['writer']}|{r['sample']}"
            paras = await paraphrases(essay)
            jobs.append((eid, essay, "original", reader, None))
            jobs += [(eid, p, f"paraphrase{i + 1}", reader, None) for i, p in enumerate(paras)]
            jobs += [(eid, ATTRIB[a] + essay, a, reader, None) for a in ATTRIB]
            jobs += [(eid, essay, f"reading{rd.reading}", rd, None) for rd in readings]
        mats = await asyncio.gather(*[rd.matrix([text], scored) for _, text, _, rd, _ in jobs])
        for (eid, text, version, _, _), P in zip(jobs, mats):
            versions.append({"family": family, "essay": eid, "version": version, "text": text})
            for it, p, yi in zip(scored, P[0], y):
                rows.append({"family": family, "essay": eid, "version": version, "item": it.id, "p": float(p),
                             "y": float(yi)})
        pd.DataFrame(rows).to_parquet(OUT / "cells.parquet")
        (OUT / "versions.json").write_text(json.dumps(versions, indent=1))
        print(f"{family}: {len(jobs)} versions x {len(scored)} items; spend so far ${C.spend() - s0:.2f}",
              flush=True)
    (OUT / "spend.json").write_text(json.dumps({"usd": C.spend() - s0}))
    print(f"done; OpenRouter spend ${C.spend() - s0:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
