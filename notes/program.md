# Training and evaluating conceptual reasoning: the program, what is built, what we found

Goal: principled training and evaluation signals for conceptual arguments and beliefs, meaning questions with
no answer key. Every check of a conceptual claim reduces to one of three signals:
- **consequences**: what it implies about things that do get checked;
- **dialectic**: whether it survives the best objections;
- **coherence**: whether a set of beliefs fits together.

Each signal fails alone. So the program builds one place where all three can be *measured against hidden
truth*, then tests methods for each there.

| project | question | where | status |
|---|---|---|---|
| 1. Testbed | questions that must be argued about, with hidden ground truth from computation, proof or experiment - mostly *about the outcomes of alignment experiments*, specified so they can be run at the click of a button | [conceptual-validation-testbed](https://github.com/abhimanyupallavisudhir/conceptual-validation-testbed) (`cvtestbed`, CLI `cvt`) | 36 families across 17 research streams; ~1,150 labelled experiment questions, plus unlimited generated ones; expensive experiments (fine-tuning) defined and runnable, unlabelled |
| 2. Oversight evaluations | do debate-like protocols make a weak judge more accurate, and stay so under optimization? | `so-arena`: `mechanisms/recursive.py`, `experiments/oversight_*.py`, [docs/oversight-results.md](https://github.com/abhimanyupallavisudhir/so-arena/blob/master/docs/oversight-results.md) | 5 protocols × 4 debater models × 160 items; best-of-$n$ curves |
| 3. Essays as latents | is an essay valuable in proportion to how it improves a frozen reader's predictions of questions its writer never saw? | `lvpm`: `lvpm/essays/`, [essays.md](essays.md) | 7 families, 150 reports, 28k reader calls |

Theory underneath project 3: [latent-meaning.md](latent-meaning.md). The tempered log score $S_\tau$ is the
latent bottleneck with the encoder optimized out. So scoring weighted natural-language hypotheses by $S_\tau$ is
a proper elicitation of latent structure, relative to the reader.

## What we found (2026-10, first pass)

1. **Argument transfers knowledge; it doesn't create it.** Across 40 (family, debater model) cells, debate's
   gain over the weak judge alone correlates with the knowledge gap between debater and judge ($r=0.52$). It
   helps where the debaters know the answer (Newcomb with a Sonnet debater: 0.62 → 0.97) and hurts where they
   don't. On average no protocol beat a reasoning weak judge alone. *Audit (2026-10-31):* prompted liars
   often concede (8–58% of transcripts); on items where the liar actually argued, every protocol scored below
   the judge alone, so the apparent gain with debater strength was mostly a weak-liar artifact
   (so-arena `docs/oversight-results.md`).
2. **Optimization pressure separates protocols.** Under best-of-$n$ against the judge's reward:
   - a single argument for a false answer fools the judge 59% → 96% of the time ($n=1\to8$);
   - debate with only the liar optimized degrades from 0.66 to 0.45;
   - debate with both sides optimized improves from 0.66 to 0.71.
3. **Prover-estimator debate's guarantee rests on a coherent estimator,** and prompted LLMs are incoherent.
   They price subclaims in isolation, so $P(\text{all subclaims})>P(\text{claim})$, and a lying prover profits
   from that. Stating the constraint helps partly. Training the estimator is the open experiment.
4. **Essays carry transferable knowledge where the reader's prior is wrong.** On LLM-auction behaviour the
   reader alone is right 19% of the time. A ~550-word essay written from 16 resolved experiments lifts it to
   71% (+1.1 nats/item, CI [0.63, 1.50]); a card-only essay gives +0.45. Where the reader already reasons well,
   essays add nothing.
5. **Latent mixtures track cross-task structure.** Updating hypothesis weights on resolved tasks improves
   predictions of unresolved ones: +0.04 nats/item, CI [0.006, 0.076], in 29 of 38 mixtures.
6. **The testbed discriminates.** Model intuitions about LLM-agent behaviour are often anti-correlated with
   measurements. RL reward-hacking outcomes follow learnable structure (a 12-feature model reaches 90%) that
   current models don't apply.

7. **$S_\tau$ recovers known truth.** On 8 planted worlds whose true theory is known, $S_\tau$ through a frozen
   reader ranks the true theory first with two readers; wrong theories (kinds merged or swapped), cheat sheets,
   calibration nudges and prompt injections lose decisively; predictively equivalent variants tie, as they should.
   The reader must be able to finish applying a theory: a 700-token reasoning budget truncated 46% of readings
   and reversed the ranking ([essays-plan.md](essays-plan.md) §4).
8. **As a reward for essays, $S_\tau$ selects better reports and an LLM judge does not.** Best-of-12 over writer
   reports on 11 testbed families: $S_\tau$ selection gains +0.065 nats per held-out question (83% of the
   oracle's gain), transferring to a held-out reader; the judge's selection is worse than random through that
   reader; $S_\tau$ beats the judge by +0.097 [0.048, 0.155] and debate by +0.062 [0.019, 0.118]. On planted
   worlds 16 resolved questions are too few to separate rewards (§5 of essays-plan).

Total API spend: about \$125 (\$3 + \$3 testbed experiments, \$25 essays, \$87 protocol comparisons, \$5
optimization, about \$1 pilots) in the first pass; since then about \$300 (planted validation \$49 including
the truncated first run, essay arena \$212, liar audit \$2, reward-size follow-up ~\$40). The fine-tuning grid
(~\$115) has sat in OpenAI's queue since submission and has not run.

## One arena for questions 2 and 3 (approved 2026-10-31; first results in finding 8)

The first experiments answered narrower questions than the program's. The oversight comparison measured
protocols as *decision procedures* with prompted debaters arguing assigned sides of yes/no questions. The essay
experiment measured whether essays carry transferable information, not whether a mechanism *incentivizes* better
essays. The program's questions are:

- (2) will training against debate or another oversight protocol produce better conceptual essays?
- (3) will training against a latent-variable mechanism with verifiable downstream tasks produce better ones?

Both ask *which reward signal, when optimized, yields better conceptual outputs*. One experiment answers both:

- **Policy:** a writer that produces an essay (a theory plus conclusions) on a conceptual topic with checkable
  implications. Examples: "when does RL find reward hacks?", "when do LLM agents collude?", "when should an agent
  one-box?", "when are chain-of-thought monitors reliable?".
- **Rewards compared,** all with weak overseers:
  - an LLM judge's rating (the RLAIF baseline);
  - debate on the essay (writer vs critic);
  - consultancy;
  - market making;
  - the latent mechanism: $S_\tau$ of the essay through a frozen reader on *training* downstream questions;
  - an oracle upper bound.
- **Optimization:** best-of-$n$, then expert iteration, then RL on an open model.
- **Ground-truth evaluation,** never used as a reward:
  - held-out downstream questions, including shifted kinds of question, so that memorizing the reward
    questions does not pay;
  - independent quality ratings (a strong-model rubric, later experts or LMCA-style ratings);
  - correctness of the essay's checkable conclusions.
- **Outputs:** Goodhart curves (ground truth against optimization pressure) for each reward signal, stratified
  by the writer-overseer knowledge gap. The essay-level analogue of so-arena's ASD measures sits alongside:
  concordance of each reward with ground truth over natural samples of essays.

## Next steps

The essays-as-latents concept, its failure modes, and the experiments that would settle it are worked out in
[essays-plan.md](essays-plan.md), with revised training costs.

 in order of expected value

1. **Train, don't select.** Run RL or iterated best-of-$n$ on essay writers with DEV $S_\tau$ as the reward, and
   plot held-out $S$ along training (the Goodhart curve). Do the same for debaters in self-play, and for the
   prover-estimator estimator on its own payoff.
2. **Power up essays** on families where the reader's prior is weak: 60+ TEST items per family, plus new
   LLM-behaviour experiment families. This is cheap, since cached work is free.
3. **Tier-B anchor.** Load LMCA once its data is available (access is by request form), then test whether the
   protocols' and essays' rankings agree with expert ratings and whether those ratings predict tier-A outcomes.
4. **More families,** each a self-run experiment generator:
   - fine-tuning experiments (e.g. narrow fine-tunes and their effect on decision-theoretic attitudes);
   - population-ethics axiom consistency (SAT);
   - research-outcome prediction from post-cutoff safety papers.
5. **Recursion and credit assignment.** An essay's value includes the gain of essays written by readers of
   it: measured influence on a DAG, the principled version of "PageRank for arguments".
