# Related work (checked 2026-10-01)

Compiled by a literature-search agent. Each item was checked against arXiv, a publisher page, Crossref, or the author's own page, and anything still uncertain is flagged. Grouped by how close it is to this project.

## Closest: dependence structure from prediction-market price co-movement

- **Augenblick & Rabin, "Belief Movement, Uncertainty Reduction, and Rational Updating"**, QJE 136(2), 2021. [pdf](https://faculty.haas.berkeley.edu/ned/AugenblickRabin_MovementUncertainty.pdf)
  - This is the *univariate* version of our identity: $E\big[\sum_t (\Delta p_t)^2\big] = p_0(1-p_0)$.
  - They test it on Betfair. There is no cross-event covariance version.
- **Chernov, Elenev, Song, "The Comovement of Voter Preferences: Insights from U.S. Presidential Election Prediction Markets Beyond Polls"**, NBER w33339, 2025. [link](https://www.nber.org/papers/w33339)
  - They fit a state-space model on Polymarket state prices, polls and fundamentals, then run PCA on the filtered latent state (two factors explain 94%).
  - Election-day correlations come from the random-walk innovation covariance of the latent state, not from probabilities.
- **Dalen, "Toward Black–Scholes for Prediction Markets"**, arXiv 2510.15205, 2025.
  - Logit jump-diffusion pricing, including correlation and covariance swaps across events.
  - It does not estimate outcome covariance from realised covariation.
- **Saguillo et al., "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"**, arXiv 2508.03474, 2025. Covers *logical* dependence (combinatorial arbitrage) between Polymarket markets.
- **Not found:** using $\sum_t \Delta p^i_t \Delta p^j_t$ as an estimator of $\mathrm{Cov}(X_i,X_j)$, or a finance-style factor model of prediction-market "returns".

## Combinatorial and latent-variable markets

- Hanson, "Combinatorial Information Market Design", *Inf. Syst. Frontiers* 2003 (introduces LMSR).
- Chen, Fortnow, Lambert, Pennock, Wortman, "Complexity of Combinatorial Market Makers", EC 2008. Pricing is #P-hard.
- Pennock & Xia, UAI 2011; Sun, Hanson, Laskey, Twardy, UAI 2012; Laskey et al., JAIR 2018 ([doi](https://doi.org/10.1613/jair.1.11249)). Bayesian-network market makers, i.e. the SciCast lineage.
- Rana et al., "ParlayMarket", arXiv 2603.22596, 2026. A pairwise (Ising-like) belief state over Kalshi combos, with no latent variable.
- **Metaculus Indexes**, live since about 2025. These are *formative* weighted sums of forecasts, not reflective factor models.
- tailcalled's Manifold market "Will Manifold have built-in support for reflective latent variables by 2025?" resolved **NO**.
- **No reflective IRT or factor latent-variable market has been implemented.**

## LLM forecasting of joint, conditional and correlated questions

- **Paleka, Pallavi Sudhir, et al., "Consistency Checks for Language Model Forecasters"**, ICLR 2025, arXiv 2412.18544.
- Paleka et al., "Pitfalls in Evaluating Language Model Forecasters", arXiv 2506.00723.
- **Karger et al., "ForecastBench"**, ICLR 2025.
  - Half the LLM question set is combination questions (Boolean combinations of question pairs).
  - LLMs do much worse on these than superforecasters.
- Andrews & Sarkar, "Dutch Books for Language Models", arXiv 2609.02797, 2026.
- **Forecasting Research Institute, "Conditional Trees"**, 2024.
  - Experts build small Bayesian networks of resolvable indicators for an unresolvable target (AI x-risk).
  - This is the closest *human* analogue to the project.
- Pieces of LLM-elicited structure exist separately:
  - LLM-abduced factors feeding a Bayesian network: BIRD (Feng et al., ICLR 2025).
  - CPTs for Bayesian networks: Nafar et al., arXiv 2505.15918.
  - Correlations via comparison questions: Kabus et al., arXiv 2605.06335.
  - Priors for statistical models: AutoElicit (Capstick et al., ICML 2025).
- **Not found:** eliciting a full LLM correlation matrix or factor model over real forecasting questions and scoring it on joint outcomes.

## RL and fine-tuning on forecasting

- Halawi et al., NeurIPS 2024 (retrieval plus fine-tuning).
- Turtel et al., "Outcome-based RL to Predict the Future", arXiv 2505.17989.
- Turtel et al., "How Proper Scoring Rules Shape LLM Forecasting", arXiv 2608.28482, 2026. Log-score rewards give the best log score and calibration.
- Chandak et al., OpenForecaster, arXiv 2512.25070.
- Damani et al., RLCR, arXiv 2507.16806.
- Thinking Machines / Mantic, "Training LLMs to Predict World Events" (blog, Mar 2026). Uses a Brier reward, chosen for stability.
- **Not found:** any training on joint or conditional log-likelihood. Every paper scores binary marginals independently.

## Mechanisms for unverifiable outcomes

- Prelec, BTS, *Science* 2004.
- Miller, Resnick, Zeckhauser, peer prediction, *Mgmt Sci* 2005.
- Kong & Schoenebeck, ACM TEAC 2019.
- Kong, DMI, ITCS 2022.
- Srinivasan, Karger, Chen, "Self-Resolving Prediction Markets for Unverifiable Outcomes", arXiv 2306.04305.
- Karger et al., "Reciprocal Scoring", SSRN 3954498.
- Baillon, Tereick, Wang, *J. Risk & Uncertainty* 2025.
- Hossain et al., "Evidence Markets", arXiv 2606.07434.
- Lehmann, review, arXiv 2409.07277. Finds empirical support for these mechanisms is weak.
- Applications to LLMs:
  - Lu et al., EC'24 (arXiv 2405.15077).
  - Peer Elicitation Games (arXiv 2505.13636).
  - Qiu, Carroll, Allen (arXiv 2601.20299).
  - BTS-based "surprisingly likely" answers (arXiv 2311.07692).
- **None connects to a measurement model**, i.e. one where a latent variable earns credibility from resolvable indicators.

## IRT

- Merkle et al., *Decision* 2016; Bo et al., *JDM* 2017. Both put the latent on *forecaster ability*, not on the world.
- tinyBenchmarks (Maia Polo et al., ICML 2024).
- "From Benchmarks to Skills", arXiv 2507.20208. Factor labels are chosen by vote, not scored by simulation.

## Explain-then-simulate label scoring

- Bills et al., "Language models can explain neurons in language models", OpenAI 2023.
- Paulo et al., arXiv 2410.13928.
- Transluce, "Scaling Automatic Neuron Description", 2024.
- **Zhong, Wang, Klein, Steinhardt, "Explaining Datasets in Words"**, NeurIPS 2024. Statistical models whose parameters are natural-language predicates; the closest template for scored latent labels.
- **Not found:** simulation-scored labels for latent factors over forecasting questions.

## Data APIs (verified)

- **Polymarket Gamma**: `/markets`, `/events`.
  - Offsets above about 2700 return 422, so page within date windows.
  - URL-encoded `:` in ISO dates returns 500, so send `:` raw.
- **Polymarket CLOB** `/prices-history?market=<token>`:
  - With `interval=max`, resolved markets need `fidelity ≥ 720`.
  - `startTs`/`endTs` windows of up to about 15 days give minute data.
- **Manifold** `/v0/bets?contractId=…` returns `probBefore`/`probAfter` for every bet, so full paths can be rebuilt.
- **Metaculus** API returns 403 without a token.
