# lvpm

Start with [notes/program.md](notes/program.md): the research program on training and evaluating conceptual
reasoning, what is built where, and the first results.

What gives a belief about a latent variable meaning, given that only observables can be scored? And can that
make conceptual writing scoreable? See [notes/latent-meaning.md](notes/latent-meaning.md) (theory) and
[notes/essays.md](notes/essays.md) (essays as latent variables: design and first results).

Theory and exact checks:

- `lvpm/dag.py`: latent DAG models; tempered EM fitting and the tempered log score that elicits the
  latent-bottleneck optimum.
- `lvpm/continuous.py`: continuous latents (IRT, factor analysis, mixed indicators, any number of factors),
  fitted and scored the same way (Gauss-Hermite or Sobol nodes; exact closed form for factor analysis).
- `lvpm/latent_bottleneck.py`: information quantities and the single-latent bottleneck
  `I(X;Y) - tau * sum_i I(X;Y_i)`.
- `experiments/latent_meaning.py`: figures and bottleneck numbers (about 40 s).
- `experiments/elicitation.py`: scoring-rule, DAG and LVPM checks (about 45 s).
- `experiments/continuous.py`: continuous-latent checks (about 13 min).
- `experiments/wyner_vs_true.py`: exact Wyner-minimal latent vs the true latent (about 15 min).

Essays as latents (questions and the cached LLM client come from the
[conceptual validation testbed](https://github.com/abhimanyupallavisudhir/conceptual-validation-testbed)):

- `lvpm/essays/score.py`: $S_\tau$ of weighted natural-language hypotheses on resolved tasks; posteriors,
  held-out scores, information diagnostics, incentive-preserving adaptive task selection.
- `lvpm/essays/reader.py`: the frozen LLM reader that gives a hypothesis its predictions.
- `lvpm/essays/writers.py`: essay and hypothesis-mixture writers, baselines, an LLM essay judge.
- `experiments/essays.py` then `experiments/essays_report.py`: the first experiment (7 testbed families,
  about $25 of API calls; cached, so re-running is free). Results in `results/essays/`.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python experiments/latent_meaning.py
PYTHONPATH=. .venv/bin/pytest tests
```
