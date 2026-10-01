# Eliciting trustworthy beliefs about latent variables: what to build

*Context: tailcalled's [latent variable prediction markets](https://www.lesswrong.com/posts/ufW5LvcwDuL6qjdBT/) (LVPMs). Related work is in [literature.md](literature.md); synthetic checks and the pilot are in [../reports/pilot.md](../reports/pilot.md).*

## Recommendation

Do these in order:

1. **Build an evaluation of joint ("structure") forecasts first.** Use resolved clusters of real questions, keep the marginals fixed, and score only the dependence structure. This is the missing piece. Without it you cannot say whether models already produce latent nets you can trust, and you have no training signal to improve them.
   - It is cheap: no model ever has to write out a whole joint distribution.
   - Nobody has done it. Every forecasting benchmark and every forecasting-RL paper scores marginals independently.
   - The pilot in this repo shows the pipeline works end to end.
2. **Then train on that signal.** Sums of conditional log-scores are an ordinary RL reward; nothing needs to be differentiable. In parallel, train a small, fully differentiable "amortised IRT" model that maps question text to loadings. It is a strong baseline and a stand-in for the market.
3. **Make labels earn their meaning.** Score a latent's natural-language label by *explain → simulate*: can a fresh model, given only the label, reproduce the loadings? This also fixes the gauge.
4. **Only then run agent markets on latent nets.** Questions that will never resolve then inherit credibility from the three tests above.

The rest of this document explains why, answering each of your sub-questions.

---

## 1. What an LVPM actually elicits, and what "trusting the latent" can mean

An LVPM state $\theta$ is a joint distribution over the resolvable indicators:

$$P_\theta(x_1,\dots,x_n)=\int P_\theta(y)\prod_i P_\theta(x_i\mid y)\,dy .$$

A bet that moves the state from $\phi$ to $\theta$ pays $\log P_\theta(\vec x)-\log P_\phi(\vec x)$. The latent $Y$ never appears in the payoff. Three consequences follow, and they shape everything else.

**(a) Only the joint over the indicators is incentivised.**
- A trader with belief $P^\*$ maximises $E_{P^\*}[\log P_\theta(\vec x)]$, which means moving the market to the KL projection $\arg\min_\theta \mathrm{KL}(P^\*\,\|\,P_\theta)$ within the model family.
- "The credence on $Y$" is therefore not a separate object. It is a coordinate on that joint.
- In the standard gauge $Y\sim N(0,1)$, the only gauge-invariant latent statements are relative to the indicators. Examples: $P(Y>y^\star)$ where $y^\star$ is the point at which $P(\text{Kostyantynivka falls}\mid y^\star)=\tfrac12$, the posterior over $Y$ after some indicators resolve, or predictions for *new* indicators.

**(b) The latent becomes asymptotically resolvable.**
- With $n$ indicators whose loadings are bounded away from 0, $\mathrm{Var}(Y\mid X_{1..n}) = O(1/n)$ (tailcalled's $\sqrt n$ signal-to-noise; in psychometrics, the test information function).
- So "is Russia winning?", defined as the common factor of many resolvable battlefield questions, becomes approximately a function of resolved facts.
- The irreducible subjectivity is whatever posterior variance remains, plus the gap between the label and the factor.

**(c) So "trust the agent's credence on a latent" breaks into four separately checkable claims:**

| claim | how to check it | resolves? |
|---|---|---|
| T1. The joint over indicators is good | structure score on resolution (§2) | yes |
| T2. The indicators pin down $Y$ | posterior variance or test information, computable *ex ante* | n/a |
| T3. The label means the factor | explain → simulate score (§6) | yes, via T1 |
| T4. The factor is stable as indicators are added | predict loadings of *new* questions from the label | yes |

T1 is the only one that needs ground truth, and it is exactly the piece no current benchmark measures.

## 2. "See how SOTA forecasters do. Compare against what? A whole joint seems crazy."

You never need the whole joint. Two identities let you evaluate it from small queries.

**Chain rule.** The joint log score is a sum of conditional log scores at the *realised* outcomes:

$$\log P(\vec x)=\sum_i \log P(x_i\mid x_{<i}).$$

So you can score an agent's implicit joint by asking $n$ separate questions of the form "$P(X_i \mid$ these others resolved like this$)$", posed as hypotheticals, each in a fresh context so nothing leaks. That is $n$ calls per ordering, with no parametric model at all. Pairwise conditionals $P(X_j\mid X_i=\pm)$ are the second-order version, and you can ask for both branches at once without leakage. That is protocol B in the pilot.

**Copula swap.** Hold the marginals fixed (for example at the market price) and score only the dependence:

$$S=\log P(\vec x\mid p,\,R)-\log P(\vec x\mid p,\,I),\qquad E_{P^\*}[S]=\underbrace{\mathrm{TC}(P^\*)}_{\text{total correlation}}-\mathrm{KL}(P^\*\,\|\,Q).$$

$S$ separates structure skill from marginal skill. Marginal skill is already well measured, and the swap cancels most of its noise. $S$ also has a natural zero (independence) and a natural ceiling (the total correlation).

**What to compare against.**
- **Independence ($S\equiv 0$).** The null.
- **The market's own co-movement.** This is model-free (§3), but noisy.
- **An amortised text→loadings model trained on past clusters** (§4).
- **Humans** on the same pairwise protocol (FRI-style conditional trees).
- **Cross-protocol agreement** (latent net vs pairwise vs label→loadings). This is a label-free check in the spirit of consistency checks.

Nothing gives an absolute "good" here. What you get is a ranking with confidence intervals, plus a covariance-calibration slope where 1 means calibrated.

**How much data.** From [`01_synthetic.py`](../experiments/01_synthetic.py), with 8 questions per cluster:

| comparison | resolved clusters needed |
|---|---|
| oracle structure vs independence | ≈ 64 |
| oracle vs a forecaster with right signs at half strength | ≈ 114 |
| oracle vs one with loading noise of sd 0.25 | ≈ 140 |
| oracle vs independence, 16 questions per cluster | ≈ 33 |

So the benchmark has to be hundreds of clusters. That is feasible: Polymarket alone resolves hundreds of markets a month, and a ForecastBench-style rolling, prospective design removes contamination entirely.

## 3. "Infer latents from how prediction markets move"

There is a clean identity that does not seem to be in the literature. Augenblick & Rabin (QJE 2021) give the univariate case, and Chernov–Elenev–Song (2025) do PCA on a filtered latent state rather than on probabilities. The identity: if prices are martingales, $p^i_t=E_t[X_i]$, then increments are orthogonal over time, so

$$\mathrm{Cov}_s(X_i,X_j)=E_s\Big[\sum_{s<t\le T}\Delta p^i_t\,\Delta p^j_t\Big].$$

**Realised quadratic covariation of belief paths is an unbiased, model-free estimate of the market-implied outcome covariance.** Factor-analysing it is "PCA for prediction markets", done correctly. The synthetic check (figure `synthetic_identity.png`) confirms it holds, with slope 0.95 and 0.89 under the two news regimes.

Two practical lessons, one from the synthetic data and one from the pilot:

- **Past co-movement reveals the structure of *past news*, not necessarily the *residual* structure of the future.**
  - When news is mostly about the common factor, the factor is largely priced in by $t_0$. Pre-$t_0$ correlations then overstate the remaining dependence, and the structure score went negative in simulation.
  - The fix is a state-space treatment that tracks how much factor variance remains (as Chernov–Elenev–Song do for elections), not raw correlations.
- **On real Polymarket data, daily co-movement over 30–60 days is dominated by estimation noise.**
  - The pilot's co-movement score swings from −11.3 to +1.1 nats depending on window, shrinkage and price scale.
  - It needs principled regularisation and more liquid markets, with settings chosen on held-out clusters.

So this is a worthwhile *paper* (a multivariate Augenblick–Rabin, with an excess-co-movement test and a market factor model). It is also a *baseline* for the benchmark. On this pilot's evidence, though, it is not yet a stronger teacher than an LLM.

## 4. Training: "how do we get differentiable signal?"

**You don't need it.**
- The reward $r=\log P_\theta(\vec x)-\log P_{\text{baseline}}(\vec x)$ is a scalar per episode. That is ordinary outcome-based RL, the same setup as Turtel et al.'s forecasting GRPO, only the reward is joint.
- The chain rule lets you train on *conditional* questions. Each "$P(X_i\mid x_S)$" episode is scored by $\log P(x_i\mid x_S)$, so a long joint task breaks into short episodes.
- **Noise level.** From the power analysis, the per-cluster reward has a signal-to-noise ratio of about 0.35 at $n=8$ (mean 0.33, sd 0.94 nats for an oracle). You therefore need thousands of cluster-episodes, which means historical data, which means contamination is the binding constraint. Train only on questions after the model's cutoff, or on prospective windows.

**Where differentiability *is* useful is the amortised model ("our own architecture").**
- Predict each question's loading from text, $\lambda_i=g_\phi(e_{\text{question}_i},\,e_{\text{label}})$, keep the marginals from the market or the forecaster, and maximise $\sum_{\text{clusters}}\log P(\vec x_c\mid p_c,g_\phi)$.
- The 1–2 factor probit likelihood is exact and differentiable by Gauss–Hermite quadrature (`lvpm.model.factor_loglik`). So this is end-to-end gradient training, i.e. IRT with text-conditioned item parameters.
- The LLM does the hard part (reasoning about marginals and naming latents). The small model learns the statistical regularities of dependence across thousands of clusters.
- It serves as the benchmark's strongest cheap baseline, as a reward-shaping critic, and as the automated market maker's prior.

**Your "small bets as data points" idea is exactly the market scoring rule, and it is already a training pipeline.**
- Bets telescope: $\sum_k [\log P_{\theta_k}(\vec x)-\log P_{\theta_{k-1}}(\vec x)]=\log P_{\theta_K}(\vec x)-\log P_{\theta_0}(\vec x)$.
- Each bet's reward is the log-likelihood ratio of the state it changed, so credit assignment is exact and each bet is its own RL sample.
- A "small bet" can touch one loading or the latent's location. That is a local, one-call query: given the current market, where should $\lambda_i$ be?
- A myopic trader moves the market to its KL projection, so the mechanism is incentive-compatible within the model family.

## 5. "A fully-connected multi-layer network can represent any distribution, right?" Yes, and that's the problem

**Expressivity.** Yes.
- Deep sigmoid belief networks and RBMs/DBNs are universal approximators of distributions over $\{0,1\}^n$ (Sutskever & Hinton 2008; Le Roux & Bengio 2008).
- Trivially, so is one discrete latent with $2^n$ states, or one continuous latent with a flexible enough decoder.

But universality destroys the three properties you need.

1. **Meaning.** The gauge group grows from $O(k)$ (rotations and sign flips) to essentially every measure-preserving reparametrisation of latent space. "Conditional independence given $Y$" becomes vacuous, and the latent stops meaning anything.
2. **Learnability.** Each cluster gives one world, i.e. *one* sample of $\vec x$. Complex per-cluster structure cannot be identified from one sample. It can only be learned if it is shared across clusters, which is the amortisation in §4.
3. **Elicitability.** Nobody, human or model, can bet sensibly on thousands of hidden weights.

**Use instead:**
- low-rank structure: 1–3 factors per cluster, plus a sparse set of residual pairwise links;
- for "latents of latents", a **bifactor / higher-order factor model**: general factors that span clusters (e.g. "US geopolitical risk-taking"), plus cluster-specific factors. This is psychometrics' measurement model / structural model split, which tailcalled also recommends.

Put the neural network in the *amortised map from text to parameters*, not in the per-cluster decoder.

The pilot supports adding residual links. A single forced factor lost information that pairwise elicitation caught, for example "Trump in the champions photo ⇒ Trump attends the final" and "Cuba deal ⇒ Cuba meeting". Pairwise scored +1.72 nats across the 8 clusters against +0.44 for the 1-factor net.

## 6. "Too many parameters for one call" and interpretable latents

**The factorisation already solves the parameter count.**
- An IRT net is $P(Y)$ plus one link per question, $P(X_i\mid Y)$. That is one call per question, two numbers each.
- The catch: separate calls must agree on what $Y$ *is*. **The label is the coordinate system.** A natural-language definition with an orientation ("HIGH = Russian advances accelerate on several axes") is what makes decomposed elicitation coherent across calls, and across agents in a market.
- So interpretability is not decoration. It is what makes coordination possible.

**This also gives the test for whether a label means what it says (explain → simulate, after Bills et al. 2023 and Zhong et al. 2024).**
1. A model sees questions (and optionally fitted loadings) and writes a label.
2. A *fresh* model sees only the label and the questions, and predicts the loadings.
3. Score the agreement of the predicted loadings, and more importantly their structure score on the outcomes.

**What this does for the gauge:**
- For $k=1$, the label fixes the sign.
- For $k>1$, the label-implied loadings are a Procrustes target that fixes the rotation (`lvpm.model.align`).
- So the label *resolves* the gauge freedom you raised, and the simulate score measures *how well* it does.
- Test T4 from §1 goes further: does the label predict the loading of a question written *after* the label?

**Pilot result.**
- Given only the latent-net agent's label, a fresh agent reproduced its loadings: correlation 0.84–1.00 (median ≈ 0.93), same sign on 100% of loaded questions in 7 of 8 clusters (90% in the eighth).
- Its structure score matched (+0.46 vs +0.44 nats).
- Caveat: both agents were the same model, so this is a lower bar than cross-model simulation.

## 7. Concrete roadmap

| phase | deliverable | rough size | decides |
|---|---|---|---|
| 0 (done here) | `lvpm` library, synthetic checks, 82-question / 8-cluster post-cutoff pilot | — | the pipeline works; power requirements |
| 1 | **Structure benchmark.** 300+ resolved clusters, prospective rolling windows, protocols A/B/C plus chain-rule conditionals, baselines (independence, co-movement, amortised model, humans) | ~1–2 months | whether model latent nets can be trusted (T1) and the size of the training headroom |
| 1b | Multivariate Augenblick–Rabin paper: realised-covariation estimator, excess co-movement test, Polymarket factor model | parallel | whether markets themselves are a usable teacher |
| 2 | RL on structure and conditional log-scores; amortised text→loadings model | after 1 | whether the gap closes |
| 3 | Label science: explain→simulate, cross-model simulators, new-indicator prediction (T3, T4) | after 1 | whether latents can be named reliably |
| 4 | Agent LVPM: agents trade on shared labelled latent nets under the log market scoring rule; then attach truly unresolvable target latents | after 2–3 | the actual goal |

## Trade-offs taken in this repo

- **Gaussian copula with probit links.** It gives exact marginals and fast exact factor likelihoods, but it cannot express higher-order interactions, such as mutually exclusive sets (excluded from the pilot) or logical ladders (one question per event).
- **Market prices as marginals.** This isolates structure, but it does not measure the LLM's marginal skill, and the elicited structure was conditioned on seeing those prices.
- **Keyword clustering by event title.** It is reproducible and blind to outcomes, but two clusters are grab-bags (US and world politics), which act as near-zero-structure controls.
- **Forecaster.** The forecaster was Claude Opus 5.5 subagents with no tools (0 tool calls verified for all 24), one sample per protocol. Prompts carried condensed resolution criteria rather than the full text written by `03_make_prompts.py`, and the pilot used the condensed versions. The responses are stored verbatim.
- **8 clusters is far below the ~64 the power analysis requires.** The pilot is evidence the pipeline works and a direction to look in, not a conclusion.
