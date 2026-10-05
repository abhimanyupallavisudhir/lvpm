# lvpm

What gives a belief about a latent variable meaning, given that only
observables can be scored? See [notes/latent-meaning.md](notes/latent-meaning.md).

- `lvpm/dag.py`: latent DAG models; tempered EM fitting and the tempered log
  score that elicits the latent-bottleneck optimum.
- `lvpm/continuous.py`: continuous latents (IRT, factor analysis, mixed
  indicators, several factors), fitted and scored the same way by quadrature.
- `lvpm/latent_bottleneck.py`: information quantities and the single-latent
  bottleneck `I(X;Y) - tau * sum_i I(X;Y_i)`.
- `experiments/latent_meaning.py`: figures and bottleneck numbers (about 40 s).
- `experiments/elicitation.py`: scoring-rule, DAG and LVPM checks (about 45 s).
- `experiments/continuous.py`: continuous-latent checks (about 10 min).
- `experiments/wyner_vs_true.py`: exact Wyner-minimal latent vs the true latent
  (about 15 min).

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python experiments/latent_meaning.py
```
