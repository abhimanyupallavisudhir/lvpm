"""Write elicitation prompts for the pilot (one file per cluster x protocol).

The forecaster sees only what was knowable on 2026-07-01: question text, resolution
criteria, deadline and the Polymarket price at t0.  Marginals are given, so every
protocol elicits *dependence structure* only (the part ordinary benchmarks never score).

  A  latent net: name the dominant latent variable Y ~ N(0,1), fix its orientation, and
     give P(YES | Y = +1 sd), P(YES | Y = -1 sd) per question  (2 numbers per question).
  B  pairwise: P(j YES | i YES), P(j YES | i NO) for every pair  (n(n-1) numbers).
  C  label -> loadings ("explain, then simulate"): a fresh forecaster gets only another
     forecaster's protocol-A label and must reproduce the conditional probabilities.

Usage: python 03_make_prompts.py            -> prompts for A and B
       python 03_make_prompts.py C A1       -> prompts for C using labels from responses/A1_*"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "data" / "pilot_clusters.json").read_text())
PD = ROOT / "data" / "elicitation" / "prompts"
PD.mkdir(parents=True, exist_ok=True)

RULES = """Rules: today is 2026-07-01. Do NOT use any tools (no web search, no file reading, nothing) -
answer purely from your own knowledge and judgement, reasoning as carefully as you like first.
The market probabilities are Polymarket prices at 2026-07-01 00:00 UTC; treat them as the
marginal probabilities (you are only asked about how the questions move *together*).
End your reply with a single fenced ```json block in exactly the requested format."""


def question_block(qs):
    lines = []
    for k, q in enumerate(qs):
        desc = " ".join(q["description"].split())[:450]
        lines.append(f"[{k}] {q['question']}\n    market P(YES) = {q['p_t0']:.2f}\n    resolution: {desc}")
    return "\n".join(lines)


def prompt_A(c, qs):
    return f"""You are an expert forecaster building a one-factor latent variable model (item response
theory) for a cluster of related forecasting questions, topic "{c.replace('_', ' ')}".

{RULES}

Questions:
{question_block(qs)}

Task:
1. Identify the single latent variable Y that best explains why these questions' outcomes are
   correlated (the common cause / "how things go" dimension). Give it a short label and say
   which direction is HIGH. Think of Y as standard normal across the scenarios you consider
   plausible as of today: Y = +1 is a world 1 sd towards HIGH, Y = -1 is 1 sd towards LOW.
2. For every question give P(YES | Y = +1) and P(YES | Y = -1). They should average to roughly
   the market probability. A question unrelated to Y gets equal values; one that is more
   likely when Y is LOW gets p_hi < p_lo. Stronger dependence = bigger gap.

Format:
```json
{{"label": "<short name of Y>", "high_means": "<what Y = +1 looks like>", "low_means": "<what Y = -1 looks like>",
  "p_hi": [<P(YES|Y=+1) for question 0..{len(qs) - 1}>], "p_lo": [<P(YES|Y=-1) for question 0..{len(qs) - 1}>]}}
```"""


def prompt_B(c, qs):
    n = len(qs)
    return f"""You are an expert forecaster estimating how a cluster of related forecasting questions
(topic "{c.replace('_', ' ')}") depend on each other.

{RULES}

Questions:
{question_block(qs)}

Task: for every pair (i, j) with i < j, give P(question j resolves YES | question i resolves YES)
and P(question j resolves YES | question i resolves NO). If you think i and j are unrelated,
both equal the market probability of j. Consider what a world in which i resolves YES (or NO)
looks like and how j fares in it. There are {n * (n - 1) // 2} pairs.

Format (one entry per pair, all pairs, i < j):
```json
{{"pairs": [[i, j, P(j YES | i YES), P(j YES | i NO)], ...]}}
```"""


def prompt_C(c, qs, label):
    return f"""You are an expert forecaster. Another forecaster defined a latent variable Y for a cluster of
forecasting questions (topic "{c.replace('_', ' ')}"):

    Y = "{label['label']}"
    HIGH (Y = +1 sd): {label['high_means']}
    LOW  (Y = -1 sd): {label['low_means']}

Y is standard normal across plausible scenarios as of today.

{RULES}

Questions:
{question_block(qs)}

Task: using the definition of Y above (do not redefine it), give for every question
P(YES | Y = +1) and P(YES | Y = -1). They should average to roughly the market probability;
unrelated questions get equal values; questions more likely when Y is LOW get p_hi < p_lo.

Format:
```json
{{"p_hi": [<question 0..{len(qs) - 1}>], "p_lo": [<question 0..{len(qs) - 1}>]}}
```"""


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "C":
        src = sys.argv[2]
        for c, qs in D["clusters"].items():
            lab = json.loads((ROOT / "data" / "elicitation" / "responses" / f"{src}_{c}.json").read_text())
            (PD / f"C_{c}.md").write_text(prompt_C(c, qs, lab))
    else:
        for c, qs in D["clusters"].items():
            (PD / f"A_{c}.md").write_text(prompt_A(c, qs))
            (PD / f"B_{c}.md").write_text(prompt_B(c, qs))
    print(sorted(p.name for p in PD.iterdir()))
