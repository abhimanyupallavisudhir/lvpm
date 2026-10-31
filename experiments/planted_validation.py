"""Does S_tau through a frozen reader recover the true theory of a world whose truth is known?

For each planted world (cvtestbed ``planted/world_k``), the true theory and nine degraded or exploitative variants
(lvpm.essays.theories) are read by a frozen reader on questions about new crystals, including shifted ones
(an unseen mine, more facets than any notebook crystal). Reported, per variant:

* mean log score per question (in-distribution and shifted), against the exact Bayes-optimal score;
* the same after recalibration: a logistic map of the reader's probabilities fitted, per variant and world, on
  half of the questions and applied to the other half - text that only moves confidence gains nothing then;
* the same with a second, held-out reader - content means the same to competent readers, exploits do not.

Mixture checks: reports [true, alternative] with weight w on the truth; does the posterior weight on the true
theory rise as questions resolve, and is the joint score maximized near w = 1?

    PYTHONPATH=. python experiments/planted_validation.py --out results/planted
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cvtestbed as cvt

from lvpm.essays import score as S
from lvpm.essays.reader import Reader, outcomes
from lvpm.essays.theories import variants

OUT = Path(__file__).resolve().parents[1] / "results" / "planted"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def recalibrated_log(p, y, seed=0):
    """Mean (expected) log score on one half after a logistic recalibration a*logit(p)+b fitted on the other (both
    ways). ``y`` may be soft: the probability that option A is correct."""
    from scipy.optimize import minimize

    idx = np.random.default_rng(seed).permutation(len(p))
    halves = [idx[: len(p) // 2], idx[len(p) // 2:]]
    out = np.zeros(len(p))
    for fit, ev in (halves, halves[::-1]):
        x, t = logit(p[fit]), y[fit]

        def nll(ab):
            q = 1 / (1 + np.exp(-(ab[0] * x + ab[1])))
            q = np.clip(q, 1e-6, 1 - 1e-6)
            return -np.mean(t * np.log(q) + (1 - t) * np.log1p(-q))

        a, b = minimize(nll, [1.0, 0.0], method="Nelder-Mead").x
        q = np.clip(1 / (1 + np.exp(-(a * logit(p[ev]) + b))), 1e-6, 1 - 1e-6)
        out[ev] = y[ev] * np.log(q) + (1 - y[ev]) * np.log1p(-q)
    return float(out.mean())


def mean_log(p, y):
    """Mean log score; ``y`` may be soft (the probability that option A is correct): then it is the expected log
    score under the true conditional distribution - exact in planted worlds, free of outcome noise."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


async def run(args):
    rows, mix_rows = [], []
    from cvtestbed.llm import LLM

    llm, cost, trunc, reads = LLM(), {"A": 0.0, "B": 0.0}, {}, {}
    for k in range(args.worlds):
        qs = cvt.load([f"planted/world_{k}"], n=args.questions, seed=0)
        y = outcomes(qs)
        # bayes_p is P(pass); turn it into P(option A is correct)
        p_a = np.array([q.data["bayes_p"] if q.options[0].startswith("Yes") else 1 - q.data["bayes_p"] for q in qs])
        shifted = np.array([q.params["shifted"] for q in qs])
        V = {"empty": None, **variants(k)}
        names = list(V)
        # one reader object per (reader, variant), sharing the cached client, so truncation is counted per variant
        ra = {n: Reader(args.reader, max_reasoning=args.max_reasoning, llm=llm) for n in names}
        rb = {n: Reader(args.reader_b, max_reasoning=args.max_reasoning, llm=llm) for n in args.b_variants or []}
        P = np.vstack(await asyncio.gather(*[ra[n].matrix([V[n]], qs) for n in names]))
        PB = np.vstack(await asyncio.gather(*[rb[n].matrix([V[n]], qs) for n in rb])) if rb else None
        for tag, rs in (("A", ra), ("B", rb)):
            for n, r in rs.items():
                cost[tag] += r.cost
                trunc[(tag, n)] = trunc.get((tag, n), 0) + r.truncated
                reads[(tag, n)] = reads.get((tag, n), 0) + r.readings
        for i, n in enumerate(names):
            p = P[i]
            rows.append({"world": k, "variant": n, "reader": "A",
                         "log_in": mean_log(p[~shifted], p_a[~shifted]),
                         "log_shifted": mean_log(p[shifted], p_a[shifted]),
                         "log_all": mean_log(p, p_a), "log_sampled": mean_log(p, y),
                         "acc_vs_bayes": float(np.mean((p > 0.5) == (p_a > 0.5))),
                         "log_recal": recalibrated_log(p, p_a, seed=k), "bayes_all": mean_log(p_a, p_a),
                         "bayes_in": mean_log(p_a[~shifted], p_a[~shifted]),
                         "bayes_shifted": mean_log(p_a[shifted], p_a[shifted]), "words": len((V[n] or "").split())})
            rows[-1]["truncated"] = ra[n].truncation_rate
        for j, n in enumerate(args.b_variants or []):
            p = PB[j]
            rows.append({"world": k, "variant": n, "reader": "B", "log_all": mean_log(p, p_a),
                         "log_in": mean_log(p[~shifted], p_a[~shifted]),
                         "log_shifted": mean_log(p[shifted], p_a[shifted]),
                         "acc_vs_bayes": float(np.mean((p > 0.5) == (p_a > 0.5))),
                         "log_recal": recalibrated_log(p, p_a, seed=k), "bayes_all": mean_log(p_a, p_a),
                         "truncated": rb[n].truncation_rate})
        # mixtures [true, alt]: posterior on the truth as questions resolve, and the joint score by weight
        for alt in ("merged", "swapped", "split", "no_mine"):
            Pm = np.vstack([P[names.index("true")], P[names.index(alt)]])
            for w in (0.1, 0.5, 0.9):
                post = [float(S.posterior([w, 1 - w], Pm[:, :m], y[:m])[0]) for m in (0, 10, 20, 40, len(y))]
                mix_rows.append({"world": k, "alt": alt, "w_true": w, "S1": S.s_tau([w, 1 - w], Pm, y, 1.0) / len(y),
                                 "S0.9": S.s_tau([w, 1 - w], Pm, y, 0.9) / len(y),
                                 **{f"post_after_{m}": v for m, v in zip((0, 10, 20, 40, len(y)), post)}})
        print(f"world {k}: done; reader A ${cost['A']:.2f}, reader B ${cost['B']:.2f}", flush=True)
    df, mix = pd.DataFrame(rows), pd.DataFrame(mix_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "variants.csv", index=False)
    mix.to_csv(OUT / "mixtures.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({**vars(args), "cost_A": cost["A"], "cost_B": cost["B"],
                                              "truncation": {f"{t}/{n}": trunc[(t, n)] / max(1, reads[(t, n)])
                                                             for t, n in trunc}},
                                              indent=1))
    summary(df, mix)


def summary(df, mix):
    a = df[df.reader == "A"].groupby("variant")[["log_all", "log_in", "log_shifted", "log_recal", "log_sampled",
                                                 "acc_vs_bayes", "bayes_all", "words", "truncated"]].mean()
    a["gap_to_bayes"] = a.log_all - a.bayes_all
    a = a.sort_values("log_all", ascending=False)
    b = df[df.reader == "B"].groupby("variant")[["log_all", "log_recal", "acc_vs_bayes", "truncated"]].mean().add_suffix("_B")
    m = mix.groupby(["alt", "w_true"])[[c for c in mix.columns if c.startswith(("S", "post"))]].mean()
    lines = ["# Planted-latent validation", "",
             "## Variants (mean over worlds; expected log score per question under the true conditional, nats)", "",
             a.join(b).round(3).to_markdown(), "", "## Mixtures [true, alternative]", "", m.round(3).to_markdown()]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--reader", default="openai/gpt-4.1-mini")
    ap.add_argument("--reader-b", default="anthropic/claude-haiku-4-5")
    ap.add_argument("--b-variants", nargs="*", default=["empty", "true", "qualitative", "cheat_sheet", "nudge",
                                                        "injection", "swapped"])
    ap.add_argument("--max-reasoning", type=int, default=4000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    asyncio.run(run(a))
