# Essays as latent variables: what the idea is, what can go wrong, and the best work to do

This note thinks the mechanism through from the theory ([latent-meaning.md](latent-meaning.md)). It names every
way it can fail, gives a principled fix for each, and lays out the experiments that would settle whether it
works, with sizes and costs. [essays.md](essays.md) describes the first (exploratory) experiment.

## 1. The mechanism, stated exactly

**Latent scoring.** Outcomes $X_1,\dots,X_n$ are uncertain, checkable and correlated. A report gives a latent
$Y$ with prior $P(Y)$ and conditionals $P(X_i\mid Y)$, implying the joint $m(x)=\sum_y P(y)\prod_i P(x_i\mid y)$.

Under the joint log score, a report whose $m$ equals the forecaster's belief beats reporting only the marginals
$\prod_i P(x_i)$ by that belief's total correlation, $\mathrm{KL}\big(P(X)\,\|\,\prod_iP(X_i)\big)$. Scored
one outcome at a time, the latent earns nothing. The tempered score
$$S_\tau=\log\sum_yP(y)\prod_iP(x_i\mid y)^\tau,\qquad \tau<1,$$
adds a preference, among latents that explain the dependence equally well, for the one carrying the fewest
bits. That rules out the trivial latent $Y=X$.

**Essays.** The values of $Y$ are texts $h_1,\dots,h_K$: hypotheses, theories, positions. The writer reports which
hypotheses exist and the weights $w_k=P(Y=h_k)$. It does **not** report the conditionals. A fixed reader model
$R$ computes them by answering each question as if $h_k$ were true: $P(x_i\mid h_k)=p_R(x_i\mid h_k)$, one call
per (hypothesis, question).

So the reader supplies the semantics, and the writer cannot choose conditionals freely, which was the gap in
LVPM. A piece of conceptual reasoning is **a set of competing positions plus credences in them**:
- the positions carry the *content*: which distinctions matter, what follows from what;
- the credences are the *beliefs*.

Two consequences:
- **A single essay ($K=1$) is a degenerate latent.** It can improve the reader's conditionals, but it cannot
  express correlation: the reader sees one question at a time, so an essay saying "70% that mechanism A
  holds" is applied as a 70/30 average to every question independently. Credences in positions, and the
  correlation they imply, exist only when the positions are separate hypotheses.
- **The natural $Y$ for a conceptual question is the answer to it.** For "do LLM agents collude tacitly?" the
  hypotheses are the positions ("yes, through signalling in their bids", "no, they bid noisily but
  competitively", ...), each stated precisely enough for the reader to derive predictions. The credences are
  then exactly the conceptual beliefs, and they are scored by the verifiable questions the positions bear on.
  This is the precise form of "latent beliefs are made meaningful by downstream tasks".

## 2. What a training signal for conceptual essays should reward

| desideratum | how $S_\tau$ treats it |
|---|---|
| true implications | directly: the log score of the reader's predictions |
| calibrated credences across positions | directly: the weights are scored jointly; the posterior after resolved questions is the update |
| generality: holds beyond the questions scored | only if questions come from a broad distribution the writer does not control (§3, T2) |
| compression, insight | $\tau<1$ penalizes table-like mixtures; length is a separate, explicit cost |
| usefulness to other reasoners | by construction: value is what the text does for a reader |
| saying only what one means | unpriced content earns nothing, and can be detected (§3, T4) |

## 3. How it can fail, and the principled fix for each

**T1. Reader exploitation.** Text that moves this reader's predictions for reasons other than content:
calibration nudges ("be confident"), instructions, prompt injection, quirks of one model.
- *Recalibration-invariant scoring.* Score after the best monotone (logistic) recalibration of the reader's
  probabilities, fitted on separate questions for each hypothesis. Then text that only shifts confidence
  earns nothing; only discrimination (resolution) pays.
- *Reader panels.* Score with several readers, or train against one reader and evaluate with a held-out reader.
  Content that means something transfers across competent readers; quirks do not. The gap between the training
  reader and a held-out reader directly measures exploitation.

**T2. Narrow questions (your concern 2a).** An essay is distinguished only as far as the questions distinguish
it. "LLMs will never cause existential risk" can score perfectly on questions that never test it.
- Questions come from a broad distribution fixed before writing, and the writer never sees which ones will be
  scored.
- *Shifted kinds* of questions held out entirely: new environment types, new models, new auction formats. A
  theory that captured the concept transfers; a theory tuned to the reward questions does not. This is the
  main generalization measure.
- *Adversarial, still incentive-compatible, selection.* Choose questions where rival reports disagree most,
  sampled with probabilities tilted towards disagreement, and reweight each scored subset by
  $q_{\text{base}}/q_{\text{adaptive}}$. Every writer's expected score is then unchanged, so incentives stay
  proper (`score.adaptive_inclusion`, `score.importance_weight`, tested). Our experiment catalog makes this
  real: questions are experiments that can be generated and run *after* the essays are committed.

**T3. Cheat sheets.** A hypothesis that lists answers.
- Impossible when the question space is large and parametric: a rule is the only short way to answer
  thousands of possible experiments, and a rule is a compression.
- Mixtures that enumerate outcome patterns tie at $\tau=1$ and lose at $\tau<1$ (tested).
- Length is reported and can be charged.

**T4. Unpriced content.** Claims with no implication for any question available are invisible to the score.
That is correct (they are not yet meaningful in this sense), but it should be *measured*. Ablate each claim and
measure the change in the reader's predictions; the priced fraction of an essay is a diagnostic in its own
right. A position that is priced nowhere is either verbal or needs new experiments.

**T5. Unidentified credences.** The weights on two positions are pinned only if the positions disagree on at least
three conditionally independent questions (Kruskal; latent-meaning.md §5). Otherwise their relative weight is
gauge-like. Two positions that predict everything identically are observationally equivalent, which is a
detector for verbal disputes. Where they disagree on fewer than three questions, the fix is experiment design:
generate questions that separate them.

**T6. The writer cannot see the questions.** This is a feature, not a bug: it is what forces generality, as
long as the writer knows the *distribution* (the family card).

## 4. The decisive validation: worlds whose latent structure is known

Before asking whether $S_\tau$-optimized essays are better on real conceptual topics, establish that the
mechanism recovers the truth when the truth is known. Two kinds of world:

**(a) Planted-latent "alien science" worlds.**
- *The world.* Fictional objects (devices, organisms, materials) with a hidden type $z\in\{1,\dots,K\}$, plus
  observable descriptors, some informative about $z$. Questions: "will object $o$ pass test $t$?". Each
  outcome depends on the type and the test's parameters, so the hidden type correlates the questions about
  one object.
- *The true theory* is writable exactly: "there are 3 kinds; kind is signalled by colour; kind 1 passes heat
  tests above 300 K; ...". Because the worlds are fictional, no model has prior knowledge, which isolates
  inference from data.
- *Writers* see resolved observations and commit theories (mixtures of positions with credences). Scoring
  uses held-out objects and tests.
- *What it settles:*
  1. Does $S_\tau$ through a frozen reader rank the true theory above degraded variants (a missing kind, merged
     kinds, spurious extra kinds, a cheat sheet, a vague theory, a calibration-only nudge, a prompt
     injection)?
  2. With $\tau<1$, is the fewest-kind correct theory preferred over an over-split one?
  3. Do mixture weights converge to the truth as data accrue, and do per-object latent beliefs ("this object
     is kind 2") score properly on a held-out indicator?
  4. Under optimization (best-of-$n$, expert iteration), do writers recover the true theory, or drift to text
     that exploits the reader?
  5. The same under an LLM judge's rating as the reward.

**(b) Real families with known rules.** The testbed's decision families (the exact payoff formula), prime races
(Chebyshev's bias), RL reward hacking (a 12-feature model reaches 90%) and the toy-ML families. Their true
theories can be written and serve as known optima.

Cost: about \$20–60 of API calls (reader gpt-4o-mini or gpt-4.1-mini), CPU for the worlds.

## 5. The main experiment: which reward produces better conceptual essays?

This one experiment answers both program questions: (2) does debate or oversight training help, and (3) does the
latent mechanism help.

- **Topics.** The testbed's experiment families (36, of which about 20 have runnable experiments) plus the
  planted worlds. Each has a card, a training pool of resolved questions, held-out in-distribution questions,
  and held-out shifted kinds.
- **Writer.** Produces a report: $K\le4$ positions plus credences, or a single essay as a baseline.
- **Rewards compared:**
  1. LLM judge rating (RLAIF);
  2. debate on the essay (critic vs writer, weak judge);
  3. consultancy;
  4. market making;
  5. $S_\tau$ on training questions through reader A (recalibrated);
  6. oracle: $S_\tau$ on held-out questions (the upper bound);
  7. no optimization.
- **Optimization ladder:**
  - best-of-$n$, with $n$ up to 64; curves computed exactly from pools;
  - expert iteration: 3 rounds of fine-tuning on the best samples;
  - RL (§7).
- **Evaluation, never used as a reward:**
  1. $S_\tau$ on held-out in-distribution questions;
  2. the same on shifted kinds;
  3. the same with held-out reader B;
  4. independent quality ratings: a strong-model rubric, human spot checks, later LMCA-style experts;
  5. on planted and known-rule worlds, distance to the true theory;
  6. diagnostics: length, priced fraction, calibration-only gain.
- **Outputs.**
  - A Goodhart curve for each reward: ground truth against optimization pressure. The claim to test is that
    $S_\tau$ stays aligned with held-out value while the judge's rating diverges.
  - The same curves split by writer-overseer knowledge gap.
  - Concordance of each reward with held-out value over natural samples: so-arena's ASD measures, applied at
    the level of essays.

## 6. Real conceptual questions with active settlement

On AI-behaviour questions where experiments are runnable, run it as science:
- rival theories, from writers or from the literature;
- experiments chosen from the catalog where the theories disagree, sampled with the incentive-preserving
  weights of T2, and run after the theories are committed;
- settlement by $S_\tau$.

Example topics: when do LLM agents collude; when is chain-of-thought faithful under pressure; when does RL find
the reward bug; when do monitors miss backdoors.

Measures:
- information gained per dollar of experiments, against random experiment choice;
- whether the market's winning theories transfer to new experiment kinds;
- how human-written theories (paper conclusions) fare against model-written ones.

Then recursion: the value of theory $T_1$ includes the gain in the scores of theories written by readers of
$T_1$, which is the measured-influence version of "PageRank for arguments".

## 7. Training: what "RL" means here, and what it costs

My earlier figures (\$50–150 API-only, \$200–1,000 for RL) were too low for a proper study. Revised:

**API-only, first.** Best-of-$n$ and expert iteration (a standard RL proxy).
- Writer: gpt-4.1-mini, fine-tuned through the OpenAI API.
- Reader: gpt-4o-mini.
- One (hypothesis, question) evaluation, with reasoning, a stated probability and both answer orders, costs
  about \$0.0015.
- *Best-of-$n$ study:* 24 topics × 32 reports × 3 hypotheses × 48 questions, about \$170, plus judges and
  debates (about \$20).
- *Expert iteration:* 3 rounds × 4 rewards. Each round samples 8 reports per topic, scores them, fine-tunes on
  the best and evaluates on held-out questions: about \$90 per round per reward, so about \$1.1k. A pilot with
  2 rewards ($S_\tau$ vs judge) and 2 rounds is about \$350.
- *Practical constraint:* OpenAI fine-tuning queues can be slow (our test job sat in the queue for over
  90 minutes).

**Real RL, on open weights with rented GPUs.**
- GRPO (TRL or verl) with LoRA on Qwen3-8B as the writer, with a frozen reader (Qwen3-8B served by vLLM).
- One step: 64 prompts × 8 samples = 512 reports, about 16k reader generations. On 2 × H100 that is roughly
  5 minutes, so about 150 steps is roughly 12 hours, or about 25 GPU-hours (\$60–80 at \$2.5–3/hour) per run.
- Five rewards × two seeds, plus the usual factor of 2–3 for setup and failed runs: about **\$1.5–3k of GPU**.
- A pilot (Qwen3-4B, $S_\tau$ vs judge vs oracle, one seed): about **\$300–600**.
- It needs a GPU provider account (RunPod, Lambda or Modal) and an API key in the vault. None is available to
  me now.

**Debate training proper** (for the oversight question). The same GRPO setup with zero-sum self-play debaters
(Qwen3-8B) and a frozen weak judge (Qwen3-1.7B), on items selected for a large gap. Each debate is about 5
generations, so about 2.5k per step: roughly \$500–1.5k.

## 8. Order of work and decision points

1. **§4: planted and known-rule worlds**, about \$50, a few days. Decision point: does the mechanism recover
   known truth and resist the listed exploits? If not, fix the reader or the scoring before anything else.
2. **§5 with best-of-$n$**, about \$200. This gives the first Goodhart curves for $S_\tau$ against judge,
   debate and oracle.
3. **§5 with expert iteration**: \$350 for the pilot, \$1.1k for the full study.
4. **RL**: \$300–600 pilot, \$1.5–3k full; needs a GPU account.
5. **§6: active settlement on AI-behaviour questions**: about \$100–300 in experiments.

**What would show the idea is wrong:**
- Best-of-$n$ or training on $S_\tau$ raising the training-reader score while held-out-reader or shifted-kind
  scores fall, faster than the judge-trained curve does.
- $S_\tau$ failing to prefer true planted theories over exploits.
