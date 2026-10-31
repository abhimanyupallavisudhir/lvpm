# Latent reward vs number of resolved reward questions (planted worlds)

Reader openai/gpt-4.1-mini; 8 worlds x 12 reports. `exact`: S on the true conditionals of all 128 questions (no outcome noise). 95% bootstrap CIs over worlds.

| reward questions | best-of-12 gain on held-out test (nats/q) | share of gap to true theory closed | Spearman with test |
|---|---|---|---|
| 16 | +0.060 [+0.010, +0.114] | +0.215 [-0.015, +0.415] | +0.374 [+0.050, +0.638] |
| 32 | +0.058 [+0.013, +0.109] | +0.207 [+0.038, +0.372] | +0.518 [+0.291, +0.690] |
| 64 | +0.057 [+0.009, +0.109] | +0.205 [+0.022, +0.383] | +0.561 [+0.289, +0.749] |
| 128 | +0.070 [+0.030, +0.117] | +0.290 [+0.124, +0.437] | +0.517 [+0.254, +0.697] |
| exact | +0.074 [+0.037, +0.120] | +0.303 [+0.176, +0.430] | +0.662 [+0.418, +0.836] |