# E3: optimizing essays against each score (synthetic worlds)

8 (world, author) runs; hill climbing, best of 4 edits per round. Held-out = exact expected log score (nats/question) on the 15 UNSEEN questions in the true world.

| reward | round | reward value | held-out expected | judge | words |
|---|---|---|---|---|---|
| S_train | 0 | -0.731 | -0.704 | 6.12 | 251 |
| S_train | 1 | -0.671 | -0.695 | 6.00 | 238 |
| S_train | 2 | -0.662 | -0.694 | 6.12 | 238 |
| S_train | 3 | -0.660 | -0.693 | 6.12 | 241 |
| S_train | 4 | -0.648 | -0.698 | 6.12 | 238 |
| judge | 0 | 6.125 | -0.704 | 6.12 | 251 |
| judge | 1 | 6.375 | -0.686 | 6.38 | 245 |
| judge | 2 | 6.500 | -0.681 | 6.50 | 241 |
| judge | 3 | 6.500 | -0.681 | 6.50 | 241 |
| judge | 4 | 6.562 | -0.687 | 6.56 | 238 |

Held-out change after 4 rounds, optimizing S_train: mean +0.006, 6/8 runs improved.

Held-out change after 4 rounds, optimizing judge: mean +0.017, 5/8 runs improved.

OpenRouter spend during this run (shared with any concurrent run): $71.49.
