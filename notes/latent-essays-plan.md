# Prototype and experiments: scoring essays as latent models

Companion to [latent-essays.md](latent-essays.md). The scorers are named as
follows:

| name | what it is |
|---|---|
| **S0** | an LLM judge grading against a rubric (cheap answer A) |
| **S1** | the interpreter's forecast given the whole essay, with no parse (cheap answer B); asked per question, or jointly |
| **S2** | the proposal: parsed scenarios, separate link queries, tempered log score $S_\tau$ |

All scores are reported as increments over a baseline. There are two
baselines: the interpreter with no essay, and the crowd forecast on the
essay's date.

## The question the experiments must answer

Does S2 measure what the theory says it measures, namely the essay's
information about the world, better than S0 and S1? And is S2 immune to the
manipulations the theory says it should ignore?

A gate comes before that: is the interpreter good enough to serve as a
semantics?

## What to build

Code goes in `lvpm/essays/`, and experiments in `experiments/essays_*`. Every
LLM call is cached, keyed by a hash of (model, prompt, inputs). This makes
reruns free and results reproducible.

1. **Question bank** (`qbank.py`)
   - **Sources:** Metaculus and Manifold, through their public APIs.
   - **Questions:** binary questions with a clear resolution.
   - **Fields per question:** text, resolution criteria, creation date,
     resolution date, outcome, the crowd-probability history, and topic tags.
   - **Topic tags** come from a fixed taxonomy. The classifier that assigns
     them is frozen before any essay is scored.
   - **Filters:** drop N/A resolutions, and drop questions with too few
     forecasters.
2. **Essays** (`corpus.py`)
   - **Historical:** dated posts (LessWrong and EA Forum through their GraphQL
     APIs, plus Substack) on topics the bank covers well: AI progress,
     Russia–Ukraine, US politics, macroeconomics, science and replication.
   - **Synthetic:** generated inside controlled worlds (E1).
3. **Interpreter** (`interpreter.py`)
   - **Configuration:** a fixed model, version, temperature and set of prompts,
     all hashed.
   - **Temporal hygiene:** no tools and no retrieval. The date is stated in
     every prompt. The model's knowledge cutoff must precede every resolution
     used in evaluation.
   - **Parse prompt.** Turns an essay into JSON: mutually exclusive, exhaustive
     scenarios with descriptions and weights, plus the latent claims, in DAG
     form if the essay argues in chains. An essay with no scenarios becomes one
     scenario. The parse is stored and shown to the author.
   - **Link prompt.** Input: (essay, scenario, question). Output: one
     probability. It is asked separately for each question. The instruction:
     "Assume this essay's account and this scenario are correct. Do not
     consider who wrote it."
   - **Ensemble:** $K$ readings, from different samples or prompt variants.
     These measure ambiguity.
   - **Baseline prompt:** the same question with no essay, as of the essay's
     date.
4. **Sampler** (`sampling.py`)
   - The measure $\nu$ is uniform over the bank questions that:
     - fall in the essay's topic tags;
     - were created after the essay's date;
     - resolved after the interpreter's cutoff.
   - Draw $m\approx30$ questions per essay.
   - The coarse topic tag is the only part of $\nu$ that depends on the essay.
5. **Scorers** (`scoring.py`)
   - S0, S1 (per-question and joint), and S2 with
     $\tau\in\{1,0.95,0.9\}$. S2 averages $\mathbb E_r[\log]$ over the $K$
     readings.
   - Claim-level weights of evidence.
   - Diagnostics: the rank of the scenario-by-question link matrix on resolved
     questions, and the number of independent consequences per claim.
   - The arithmetic is a log-sum-exp over scenarios. Claims in DAG form use
     `LatentDAG`.
6. **Harness**
   - A command-line tool to score one essay.
   - Experiment scripts that write parquet files.
   - A short report per experiment, with its predictions and thresholds
     registered before the run.

## Experiments

### E0. Interpreter validity (gate)

Each criterion is fixed before the run. If the interpreter fails, fix it or
replace it before going further.

| test | what is measured | passes if |
|---|---|---|
| calibration | baseline log score on bank questions resolved after the cutoff, compared with the crowd | not far behind the crowd |
| responsiveness | essays stating a mechanism vs. its negation | links move in opposite directions (AUC high) |
| literal reading | the same essay attributed to a respected vs. a disreputable author | mean link change ≈ 0 |
| paraphrase invariance (gauge) | 5 meaning-preserving paraphrases per essay | S2 intraclass correlation across paraphrases > 0.8 |
| parse stability | scenario sets across re-parses | high agreement |

### E1. Synthetic worlds with known latents (the core test)

**Why synthetic.** In a fictional world we know the latent truth. Each world
can be realized many times, so *expected* scores can be estimated and
properness can actually be tested. With one real world this is impossible.
And invented entities give the interpreter no prior knowledge that could leak.

**Setup.**
- A generator draws a latent DAG over 3–5 factors, which may be binary or
  continuous. It then writes 100–300 natural-language questions whose outcomes
  follow known conditionals.
- Each world is realized about 200 times, with fresh latent draws each time.
- **Authors** are LLM agents. Each gets a private signal (a noisy view of some
  latents) and writes an analytical essay without seeing the questions.
- Signal quality is varied by design. So every essay's *oracle quality* is
  known exactly: the expected log score, on $\nu$, of the author's Bayesian
  posterior predictive.

**Essay conditions:**
- honest essays, across a range of signal quality;
- **list** essays: the same information, written as answers to the questions
  the author was shown, with no mechanism;
- **style** variants: the same content rewritten as confident, authoritative
  or eloquent, vs. plain;
- **proliferation:** hedges spread over many scenarios;
- **distortion:** the author is told to overstate a claim they don't believe.

**Primary outcome.** Spearman correlation between each scorer and oracle
quality across essays, pooled over worlds. The prediction: S2 ≥ S1 > S0.
S2's advantage over joint S1 should be largest when essays omit common causes.

**Secondary outcomes.** Each one tests a specific design choice:

| prediction | design choice it tests |
|---|---|
| list essays win on questions they saw and lose on held-out ones | views sampled after writing |
| for essays with right marginals but no mechanism, (joint S1 − S2) tracks the true dependence the essay leaves unexplained | separate link queries |
| style changes move S0, not S2 | gauge / literal semantics |
| distorted essays lose expected S2 across realizations | properness |
| the sign of a claim's weight of evidence is right more often as its number of independent consequences grows from 1 to 3+ | Duhem–Quine rule |
| proliferation is penalized at $\tau<1$ and not at $\tau=1$ | $\tau$ |

**Ablation.** Recompute S2 with *true* links, computed from the generator for
the parsed scenarios. The gap between true-link S2 and LLM-link S2 isolates
interpreter error from mechanism error.

### E2. Historical essays (external validity)

**Setup.** 300–1000 essays from 2023–2025 on 3–4 topics. Each is scored on
questions created after it and resolved after the interpreter's cutoff.
There is no ground-truth quality, so the tests use reliability and
incremental validity:

- **Split-half reliability:** the same essay scored on two disjoint question
  samples.
- **Track record:** an author's mean score in one period predicts their mean
  in the next. Compare S2 with S0.
- **Beating the crowd:** what fraction of essays have S2 > 0 relative to the
  crowd forecast on their date.
- **Placebo:** questions from unrelated topics should give increments ≈ 0.
- **Disagreements:** sample essays where S0 and S2 disagree and inspect them by
  hand.
- **Robustness:** rerun the E0 paraphrase and attribution tests on real
  essays.

**Leakage.** The interpreter may have seen an essay, or later discussion of
it, in training. Only outcomes after the cutoff count. The placebo and the
attribution test are there to catch residual leakage.

### E3. Adversarial optimization

**Setup.** An LLM writer hill-climbs an essay against each scorer on a
training sample of questions. Then measure:
- the gain on fresh held-out questions;
- the change in oracle quality, in synthetic worlds.

**Prediction.**
- S0 gains without any oracle gain.
- S2's training gains transfer to held-out questions only when they reflect
  real information.

**The quantity of interest** is the gap between training and held-out gains,
which measures how far the interpreter can be gamed. If the gap is large, the
mitigations are an interpreter ensemble and a hidden question bank.

### E4. Market (later)

Sequential essays act as trades against the consensus. Check the payment
bookkeeping, the loss bound, and whether the consensus improves on the crowd
over time.

## Order and resources

- **Order:** build, then E0, then E1, then E3 on synthetic worlds, then E2,
  then E4. E1 can run before the question bank exists.
- **Calls per essay for S2:** about $1+|H|\cdot m\cdot K\approx1+4\cdot30\cdot3=361$.
  The essay is a shared prompt prefix, so prompt caching makes most input
  tokens cheap. A small model at batch prices puts E1 (about 2,000 essays) on
  the order of a few hundred US dollars. A pilot of 20 essays will measure the
  real cost before committing.
- **Needed from you:**
  - an API key: Anthropic, and possibly an open-weights host if a model with
    a strict, documented knowledge cutoff is wanted for E2;
  - a spend limit;
  - your choice of 3–4 topics for E2;
  - a Metaculus API token, if their API requires one for resolution data.
