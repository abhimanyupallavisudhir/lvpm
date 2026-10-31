# E0: is the reader good enough to be a semantics?

Reader: `openrouter/anthropic/claude-haiku-4.5` (reasoning, then a stated probability; both answer orders). Essays: the first experiment's informed essays, the first essay of two different writers per family (14 essays), each read on the family's 32 DEV+TEST items.

## 1. How far each change moves the reader's predictions

Mean |change in P(option A)| per (essay, item), with 95% bootstrap CIs.

| change | mean |dp| |
|---|---|
| the essay itself (vs. the reader alone) | 0.513 [0.481, 0.544] |
| paraphrase (vs. original) | 0.055 [0.050, 0.061] |
| another reading of the same text (temperature 0.7, pairwise) | 0.052 [0.047, 0.058] |
| author framed as respected (vs. original) | 0.045 [0.038, 0.054] |
| author framed as unreliable (vs. original) | 0.056 [0.049, 0.064] |
| respected vs. unreliable | 0.058 [0.051, 0.065] |

## 2. Literal vs. evidential reading

Leakage = (P_respected - P_unreliable) / (P_essay - P_reader alone), over cells where the essay moves the prediction by more than 0.1 (407 cells). 0 = the reader takes the essay literally; 1 = the 'unreliable' framing fully undoes the essay.

Leakage: -0.038 [-0.068, -0.007]

## 3. Do paraphrases get the same score?

Per-essay gain over the reader alone (mean log score per item, nats), for the original and its 3 paraphrases.

- intraclass correlation (between-essay variance / total): **0.96** (pre-registered pass: > 0.8)
- mean SD of an essay's gain across its paraphrases: 0.030 nats/item; SD of gains between essays: 0.191

## 4. Ambiguity

Across 3 readings at temperature 0.7: mean per-cell SD of P 0.035; scoring each reading and averaging (E_r log) is below scoring the pooled reading (log E_r) by 0.013 nats/item on average (max 0.037). That gap is the price of ambiguity.

## 5. The reader alone

| family | accuracy | log score (nats/item) |
|---|---|---|
| agents/auction | 0.38 | -0.920 |
| agents/persona | 0.41 | -0.799 |
| agents/sycophancy | 0.88 | -0.424 |
| decisions/newcomb | 0.91 | -0.160 |
| numbers/prime_race | 0.50 | -0.803 |
| rlhack/fake_goal | 0.66 | -0.643 |
| rlhack/sensor | 0.69 | -0.620 |

Essay gains under this reader vs. the first experiment's reader (gpt-4.1-mini), same essays: Pearson r = 0.65 over 14 essays. Only 2 of 7 families were read on the same items (the testbed's item ids have changed since); for the rest the items are a different draw from the same family, so this mixes reader dependence with item sampling: r = {r:.2f} is a lower bound on how far the two readers agree.

OpenRouter spend: $39.08.
