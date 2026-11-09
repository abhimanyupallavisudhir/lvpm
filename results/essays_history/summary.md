# E2: real essays, scored on questions created after them

75 LessWrong essays (Sep 2025 - Feb 2026; 29 authors; topics {'ai': 60, 'us_politics': 7, 'economy': 5, 'geopolitics': 3}), truncated to 3,000 words. Each is scored on 20 resolved Manifold questions in its topic created after it (and 10 placebo questions from other topics). Reader: claude-haiku-4.5 (training cutoff about July 2025, before every essay and outcome). Gains: nats per question over the same reader given only the essay's date.

## 1. Do essays help the reader?

| score | mean gain over the date-only reader [95% CI] |
|---|---|
| S1 (whole essay), topic questions | -0.030 [-0.050, -0.011] |
| S1, placebo questions (other topics) | -0.000 [-0.021, +0.022] |
| S1, topic minus placebo | -0.030 [-0.062, +0.000] |
| S2 (parsed scenarios), topic questions | -0.032 [-0.052, -0.013] |

Parse: {1: 4, 2: 21, 3: 28, 4: 22} essays with K scenarios.

## 2. Is the score a stable property of the essay?

- S1: split-half correlation across essays -0.09 (Spearman-Brown, 20 questions: -0.21)
- S2: split-half correlation across essays 0.04 (Spearman-Brown, 20 questions: 0.08)
- S1: share of variance between authors (authors with 3+ essays, n=7): 0.16
- S2: share of variance between authors (authors with 3+ essays, n=7): 0.24
- judge: share of variance between authors (authors with 3+ essays, n=7): 0.61

## 3. Does the judge agree?

- Spearman(judge, S1) = -0.00 (n=75)
- Spearman(judge, S2) = -0.09 (n=75)
- Spearman(judge, S1_adj) = -0.15 (n=75)

Largest disagreements (S1 topic-minus-placebo percentile minus judge percentile):

| essay | author | judge | S1 adj. |
|---|---|---|---|
| Little Echo | Zvi | 2.0 | +0.139 |
| Please, Don't Roll Your Own Metaethics | Wei Dai | 3.0 | +0.102 |
| MAGA speakers at NatCon were mostly against AI | Remmelt | 4.0 | +0.204 |
| Alignment Faking Demo for Congressional Staffers | Alice Blair | 4.0 | +0.188 |
| Please Do Not Sell B30A Chips to China | Zvi | 6.0 | -0.345 |
| Focus transparency on risk reports, not safety cases | ryan_greenblatt | 6.0 | -0.200 |
| The Economics of Replacing Call Center Workers With AIs | michaelwaves | 6.0 | -0.196 |
| Anthropic and the DoW: Anthropic Responds | Zvi | 6.0 | -0.192 |

OpenRouter spend during this run (shared with any concurrent run): $195.16.
