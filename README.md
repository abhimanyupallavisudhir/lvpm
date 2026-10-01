# lvpm

What gives a belief about a latent variable meaning, given that only
observables can be scored? See [notes/latent-meaning.md](notes/latent-meaning.md).

- `lvpm/latent_bottleneck.py`: information quantities, the latent bottleneck
  `I(X;Y) - tau * sum_i I(X;Y_i)`, and its tempered-EM solver.
- `experiments/latent_meaning.py`: figures and numbers for the note (about 40 s).
- `experiments/wyner_vs_true.py`: exact Wyner-minimal latent vs the true latent
  (about 15 min).

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python experiments/latent_meaning.py
```
