# E1: essay scores against known latent truth (synthetic worlds)

8 worlds x 5 author profiles x 2 settings. Scores: nats per question; 'expected' = exact expectation over outcomes in the true world; 'realized' = one draw.

## 0. The reader as a link function

| | expected score |
|---|---|
| best possible (the true probabilities) | -0.541 |
| reader given the true factors as facts (truth essay) | -0.672 |
| true factors through true links (= best possible) | -0.542 |

- reader alone, valdoria: expected score -0.746 (averaged over worlds)
- reader alone, corvane: expected score -0.751 (averaged over worlds)

## 1. Does each score track true information? (honest essays)

Spearman correlation with the author's oracle information (the expected score of their exact posterior), within setting, averaged over the two settings.

| score | Spearman with oracle |
|---|---|
| S2, realized (one outcome draw, all 30 questions) | 0.65 |
| S2, expected (no outcome noise) | 0.71 |
| S2 with true links (parse -> generator probabilities) | 0.82 |
| S0: strong-model judge rating | 0.02 |

Mean by profile (expected S2 / oracle / judge):

| profile | S2 expected | oracle | judge |
|---|---|---|---|
| misled | -1.187 | -1.041 | 5.2 |
| partial | -0.762 | -0.642 | 4.6 |
| strong | -0.719 | -0.549 | 5.1 |
| uninformed | -0.789 | -0.692 | 4.8 |
| weak | -0.680 | -0.601 | 4.8 |

## 2. Manipulations (paired differences, 95% bootstrap CIs)

| comparison | S2 expected | judge |
|---|---|---|
| eloquent restyle - honest | -0.009 [-0.015, -0.003] (n=80) | -0.412 [-0.606, -0.212] (n=80) |
| overconfident - honest | -0.025 [-0.039, -0.012] (n=80) | -0.475 [-0.694, -0.256] (n=80) |

Properness: expected score *under the author's own beliefs* (worlds drawn from their posterior), overconfident - honest: -0.026 [-0.035, -0.017] (n=80). Proper scoring predicts < 0: an author who believes their own information loses by overstating it.

List of predictions (for the SEEN half only) - honest essay, expected score:

- on SEEN questions: +0.015 [-0.004, +0.033] (n=80)
- on UNSEEN questions: +0.036 [+0.016, +0.055] (n=80)

## 3. Letting the reader use other outcomes (holistic) vs. structured scoring

On the UNSEEN questions. Holistic = the reader is given the essay plus the realized SEEN outcomes.

- reader alone + SEEN outcomes (no essay): expected -0.783
- honest essay, structured: expected -0.825; Spearman with oracle 0.68
- honest essay + SEEN outcomes, holistic: expected -0.771; Spearman with oracle 0.43
- scenario mixtures, structured update on SEEN outcomes (Bayes over the writer's own scenarios): gain on UNSEEN +0.012 [-0.006, +0.030] (n=80) nats/question

OpenRouter spend: $283.77.
