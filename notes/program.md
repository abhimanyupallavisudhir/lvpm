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
| 1. Testbed | questions that must be argued about, with hidden ground truth from computation, proof or experiment | `so-arena`: `so_arena.domains.testbed`, [docs/testbed.md](https://github.com/abhimanyupallavisudhir/so-arena/blob/master/docs/testbed.md) | 21 families, ~2,000 items. 600 RL and 120 LLM-agent experiments run and committed as data |
| 2. Oversight evaluations | do debate-like protocols make a weak judge more accurate, and stay so under optimization? | `so-arena`: `mechanisms/recursive.py`, `experiments/oversight_*.py`, [docs/oversight-results.md](https://github.com/abhimanyupallavisudhir/so-arena/blob/master/docs/oversight-results.md) | 5 protocols × 4 debater models × 160 items; best-of-$n$ curves |
| 3. Essays as latents | is an essay valuable in proportion to how it improves a frozen reader's predictions of questions its writer never saw? | `lvpm`: `lvpm/essays/`, [essays.md](essays.md) | 7 families, 150 reports, 28k reader calls |

Theory underneath project 3: [latent-meaning.md](latent-meaning.md). The tempered log score $S_\tau$ is the
latent bottleneck with the encoder optimized out. So scoring weighted natural-language hypotheses by $S_\tau$ is
a proper elicitation of latent structure, relative to the reader.

## What we found (2026-10, first pass)

1. **Argument transfers knowledge; it doesn't create it.** Across 40 (family, debater model) cells, debate's
   gain over the weak judge alone correlates with the knowledge gap between debater and judge ($r=0.52$). It
   helps where the debaters know the answer (Newcomb with a Sonnet debater: 0.62 → 0.97) and hurts where they
   don't. On average no protocol beat a reasoning weak judge alone.
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

Total API spend: about \$120 (\$3 testbed experiments, \$25 essays, \$87 protocol comparisons, \$5
optimization, under \$1 pilots).

## Next steps, in order of expected value

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
