"""Essays as latent variables: score natural-language theories by what they do to a frozen reader's predictions.

* ``score``: the tempered log score of a report (weighted hypotheses) on resolved binary tasks, posteriors,
  held-out scores, information diagnostics, importance-weighted adaptive task selection;
* ``reader``: the frozen LLM reader that turns a hypothesis into task probabilities;
* ``writers``: LLM writers of essays and hypothesis mixtures, plus baselines and an LLM essay judge.
"""
