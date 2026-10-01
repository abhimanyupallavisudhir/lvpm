# lvpm — latent variable nets for eliciting beliefs on unresolvable questions

Builds on tailcalled's [latent variable prediction markets](https://www.lesswrong.com/posts/ufW5LvcwDuL6qjdBT/). The goal is trustworthy AI credences on questions that will never resolve, by tying each one to a latent factor behind many questions that *do* resolve.

**Start here**
- [docs/proposal.md](docs/proposal.md): what to build and why, answering each design question.
- [reports/pilot.md](reports/pilot.md): synthetic checks, plus an 82-question, post-cutoff pilot scoring LLM latent nets against resolved Polymarket outcomes.
- [docs/literature.md](docs/literature.md): related work and the gaps.

## Library (`lvpm/`)

| module | what |
|---|---|
| `model.py` | Probit factor / Gaussian-copula models with exact marginals. Exact 1–2 factor likelihood by Gauss–Hermite quadrature, Genz for a general $R$. Converts conditionals and joint probabilities to tetrachoric $\rho$ and loadings. Factor fitting and gauge alignment (Procrustes). |
| `scoring.py` | Joint log-score, the copula-swap *structure score*, the covariance-calibration slope, and bootstrap CIs. |
| `covariation.py` | Dependence implied by price co-movement, using the martingale identity $\mathrm{Cov}(X_i,X_j)=E[\sum_t\Delta p^i_t\Delta p^j_t]$. |
| `data/polymarket.py` | Cached fetcher for resolved Polymarket markets and price histories. |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
python experiments/01_synthetic.py           # ~4 min: identity check + power analysis
python experiments/02_build_pilot_dataset.py # fetches Polymarket (cached in data/cache/)
python experiments/03_make_prompts.py        # elicitation prompts (responses are in data/elicitation/responses/)
python experiments/04_score_pilot.py         # scores + figures -> reports/
python experiments/05_comovement_sensitivity.py
```
