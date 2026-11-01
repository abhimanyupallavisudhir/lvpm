# Results: scoring essays as latent models (E0–E3)

This note reports the experiments planned in [latent-essays-plan.md](latent-essays-plan.md), which test the
proposal in [latent-essays.md](latent-essays.md). The score names follow the plan:
- **S0:** an LLM judge;
- **S1:** the frozen reader given the whole essay;
- **S2:** parsed scenarios, separate per-question reader calls, and the tempered log score.

The reader is `claude-haiku-4.5` through OpenRouter. It reasons, then states a probability; both answer
orders are asked; probabilities are clipped to [0.01, 0.99]. All scores are in nats per question.

Each experiment has its own summary file:
- [E0](../results/essays_validity/summary.md)
- [E1](../results/essays_worlds/summary.md)
- [E2](../results/essays_history/summary.md)
- [E3](../results/essays_optimize/summary.md)

## Summary

1. **The reader is a usable semantics** (E0). Paraphrases and author framing move its predictions no more
   than re-reading the same text does. Ambiguity costs 0.01 nats per question.
2. **Where the truth is known, S2 tracks it and the judge does not** (E1). Measured as Spearman correlation
   with the author's exact information, in synthetic worlds:

   | score | correlation |
   |---|---|
   | S2, one realized outcome draw | 0.65 |
   | S2, exact expectation | 0.71 |
   | S2 with true links | 0.82 |
   | strong-model judge | 0.02 |

   Two of the theory's specific predictions hold:
   - **Properness:** overstating your information loses expected score under your own beliefs.
   - **Separate queries matter:** letting the reader see other outcomes, as a joint query would, drops the
     correlation from 0.68 to 0.43.
3. **The interpreter is the bottleneck, not the scoring rule.** Even when told the true facts, the reader
   reaches only −0.672 against a best possible −0.541. It gets the direction wrong on about 20% of questions,
   and it does not distinguish strongly from weakly linked questions. This one weakness explains every
   anomaly below.
4. **One prediction failed.** I expected a list of predictions for questions the writer was shown to be
   useless on unseen questions. It was *better* than an honest essay there (+0.036). With few latents and
   redundant questions, a list of predictions *is* an efficient encoding of the latent. Held-out questions
   separate theories from lists only when the unseen questions cannot be inferred from the listed ones.
5. **Optimization at this scale** (E3, 8 runs, 4 rounds of best-of-4):
   - Optimizing S on 15 resolved outcomes overfits: +0.083 on the training outcomes, +0.006 held out
     [−0.012, +0.023].
   - Optimizing the judge gave a small held-out gain, +0.017 [+0.002, +0.032].
   - Neither collapsed. The experiment is underpowered for comparing the two.
6. **Real essays: a null result** (E2; 75 LessWrong essays scored on 1,500 later Manifold questions).
   - On average, essays slightly *hurt* the reader on their own topic (−0.030 [−0.050, −0.011]) and do nothing
     on placebo topics.
   - With 20 questions per essay, the per-essay score has no split-half reliability (≈0).
   - The judge does not agree with S (ρ ≈ 0). The judge is much more consistent *within an author* (50% of
     variance between authors, vs. 16–24% for S). Together with E1, where the judge ignored the information,
     this suggests the judge rates authors and style, not predictive content.

## E0: is the reader a semantics?

These are 14 essays from the first experiment, two per family, each read on 32 items.

| change to the essay | mean shift in predicted probability |
|---|---|
| the essay itself, vs. the reader alone | 0.513 |
| paraphrase (3 per essay) | 0.055 |
| another reading of the same text (temperature 0.7) | 0.052 |
| framed as from a respected expert, vs. original | 0.045 |
| framed as from an unreliable source, vs. original | 0.056 |

**Literal reading.** Leakage measures how much of an essay's effect the "unreliable author" framing undoes.
It is −0.04 [−0.07, −0.01], against 1 if the framing undid the essay entirely. So the reader reads
literally.

**Paraphrase invariance.** Essay gains have an intraclass correlation of 0.96 across paraphrases, above the
pre-registered 0.8.

**Ambiguity.** Scoring each of three readings and averaging, instead of scoring the pooled reading, costs
0.013 nats per question.

**Reader dependence.** Against the first experiment's reader (gpt-4.1-mini), essay gains correlate at
r = 0.65. Five of the seven families were read on different items, so this is a lower bound on how far the
two readers agree.

## E1: synthetic worlds with known latents

**Setup.**
- Two fictional settings: Valdoria, a republic, and Corvane Therapeutics, a biotech.
- Each has 3 hidden yes/no factors, with all 8 combinations used.
- Each setting has 30 questions, and each question depends on one factor with designed probabilities.
- Five author profiles, from uninformed to confidently misled, so every author's information quality is
  known exactly.
- In total: 80 (world, author) cells, 6 essay conditions, 2 writer models, about 23,000 reader calls.

| test | result | verdict |
|---|---|---|
| S2 tracks information | Spearman 0.65 (realized), 0.71 (expected); true links 0.82 | supported |
| the judge tracks information | Spearman 0.02 | the judge fails |
| style: eloquent restyle, same claims | S2 −0.009 [−0.015, −0.003]; judge −0.41 | S2 nearly invariant; the judge reacts to style, and negatively |
| properness: overconfident vs. honest, under the author's own beliefs | −0.026 [−0.035, −0.017] | supported |
| separate queries vs. letting the reader see outcomes | correlation with information 0.68 → 0.43. The reader with outcomes but no essay (−0.783) beats honest essays read alone (−0.825) | supported: a joint or holistic score credits the reader's own inference |
| list of predictions on unseen questions | +0.036 [+0.016, +0.055] better than honest essays | prediction failed (see summary point 4) |
| scenario mixtures updating on resolved questions | +0.012 [−0.006, +0.030] on unseen | not detectable |

**Why "strong" authors' essays score below "weak" ones.** This is the exception to S2 tracking information.
Their exact information is better (−0.549 vs. −0.601), but S2 is worse (−0.719 vs. −0.680). The reader turns
a confident essay into predictions that are equally extreme on weak and strong questions, so confident
essays pay for the reader's miscalibration. The true-link score, which reads the essay's factor
probabilities and applies the true conditionals, has no such inversion.

## E2: real essays, real questions

**Setup.**
- **Essays:** 75 LessWrong posts from September 2025 to February 2026, by 29 authors. They were chosen by a
  cheap model as forward-looking and grouped by topic: 60 AI, 15 other. Each is truncated to 3,000 words.
- **Questions:** each essay gets 20 Manifold questions in its topic, plus 10 placebo questions. All are
  binary, have at least 15 traders, were created after the essay, and resolved by October 2026. Question
  topics come from a cheap-model classifier.
- **Temporal hygiene:** the reader's training data ends around July 2025, before every essay and every
  outcome.
- **Baseline:** the same reader given only the essay's date.

| score | gain over date-only reader |
|---|---|
| S1, topic questions | −0.030 [−0.050, −0.011] |
| S1, placebo questions | −0.000 [−0.021, +0.022] |
| S2, topic questions | −0.032 [−0.052, −0.013] |

- **Parse:** most essays split into 2–4 scenarios; only 4 parsed as a single scenario.
- **Reliability:** split-half correlation of per-essay S1 and S2 is −0.09 and 0.04.
- **Judge vs. S:** Spearman 0.00 with S1 and −0.09 with S2 (n = 75).

**Reading.** These essays were not written to inform forecasts of Manifold questions. A 20-question sample
cannot resolve differences between them. The negative mean is consistent with the reader taking each
essay's framing too literally on questions the essay does not address.

That last point is the price of literal reading noted in [latent-essays.md](latent-essays.md) §2. An essay
scored on questions it never discusses is charged for whatever its framing implies about them.

## E3: optimizing against each score

**Setup.**
- 8 synthetic (world, author) runs.
- An editor holding the author's information proposes 4 revisions per round; the best under the reward is
  kept.
- Held-out evaluation is the exact expected score on 15 unseen questions.

| reward | reward gain (4 rounds) | held-out gain |
|---|---|---|
| S2 on 15 resolved training questions | +0.083 [+0.049, +0.128] | +0.006 [−0.012, +0.023] |
| judge rating | +0.44 [+0.25, +0.63] | +0.017 [+0.002, +0.032] |

S2's training gain is mostly overfitting to 15 noisy outcomes, which is the expected failure of a
small-sample reward. The judge's gain did not hurt here, because the editor knew the truth and every edit
stayed honest. A real test needs writers without access to the truth, and more rounds.

## What this means for the proposal

- **The mechanism behaved as the theory says wherever the truth was available:**
  - properness;
  - the gain from separate queries over holistic ones;
  - invariance to paraphrase and attribution;
  - the judge failing to track information.
- **The binding constraints are the interpreter and statistical power:**
  - **Interpreter.** Its link error costs as much as most information differences between essays. E1's
    generator gives unlimited conditional questions with known answers, which is exactly the data needed to
    *train or calibrate* a reader. That is the obvious next step.
  - **Power.** On real essays, 20 questions per essay is far too few. Either use hundreds of questions per
    essay, or compare essays aggregated over many authors.
- **One claim of the note needs qualifying.** Held-out questions do not make lists of predictions worthless;
  they make them worth exactly what they imply about the latent. To separate theory from data, held-out
  questions must include *shifted* kinds of question that cannot be inferred from listed answers.

## Costs and data

- **Total OpenRouter spend:** $348. The breakdown is approximate, because E1–E3 ran concurrently on one key:
  - E0: $39;
  - pilots: $34;
  - E1, E2 and E3 together: about $275;
  - classification: under $1.
- **Blocked sources:**
  - OpenAI models are refused on both OpenRouter keys (an account-level block), so no OpenAI reader or writer
    was used.
  - The Metaculus key authenticates, but the API returns no resolutions for it, so E2 uses Manifold.
- **Data:**
  - `data/lw_posts_2025-09_2026-02.json`: LessWrong GraphQL `posts(view:"new", after:2025-09-01,
    before:2026-03-01, karmaThreshold:20)`.
  - `data/manifold_pool.json`: Manifold `search-markets` (resolved binary markets created since 2025-09-01,
    at least 15 traders, topic search terms in the data script), with topics reassigned by a cheap model.
  - `data/e2_essays.json`: the classified, truncated essays.
- **Judge fix-ups:** some judge calls hit the 800-token limit before writing a score: 10 of 400 in E1 and 26
  of 75 in E2. They were rescored with a larger limit and the same rubric. Affected records carry a
  `judge_note`.
