# Essay arena: best-of-n

19 topics, 227 reports (up to 12 per topic). Gains are best-of-12 minus the mean report, averaged over topics, with 95% bootstrap CIs over topics. `oracle` on held-out questions is in-sample (an upper bound).

## planted: gain of best-of-12 over a random report

| reward | held-out questions | shifted questions | held-out reader | expert rubric (1-10) |
|---|---|---|---|---|
| latent S_τ | +0.046 [-0.016, +0.108] | +0.057 [-0.022, +0.126] | +0.035 [-0.017, +0.088] | +0.115 [-0.115, +0.323] |
| LLM judge | +0.022 [-0.008, +0.061] | +0.048 [+0.003, +0.098] | +0.023 [-0.007, +0.062] | -0.135 [-0.406, +0.146] |
| debate | +0.042 [+0.007, +0.079] | +0.063 [+0.004, +0.119] | +0.044 [+0.006, +0.081] | +0.115 [-0.219, +0.479] |
| oracle (test S) | +0.099 [+0.070, +0.135] | +0.087 [+0.021, +0.144] | +0.086 [+0.059, +0.114] | +0.115 [-0.198, +0.458] |

Mean per-topic Spearman correlation of reward with evaluation (planted):

| reward | held-out questions | shifted questions | held-out reader | expert rubric (1-10) |
|---|---|---|---|---|
| latent S_τ | +0.40 [+0.15, +0.62] | +0.27 [+0.10, +0.44] | +0.31 [+0.05, +0.59] | +0.03 [-0.11, +0.18] |
| LLM judge | -0.03 [-0.26, +0.23] | +0.07 [-0.12, +0.29] | -0.05 [-0.25, +0.16] | +0.22 [+0.04, +0.40] |
| debate | +0.13 [-0.14, +0.38] | +0.10 [-0.05, +0.23] | +0.12 [-0.16, +0.38] | +0.03 [-0.15, +0.18] |
| oracle (test S) | +1.00 [+1.00, +1.00] | +0.55 [+0.33, +0.72] | +0.74 [+0.63, +0.82] | +0.06 [-0.15, +0.25] |

## testbed: gain of best-of-12 over a random report

| reward | held-out questions | shifted questions | held-out reader | expert rubric (1-10) |
|---|---|---|---|---|
| latent S_τ | +0.065 [+0.042, +0.098] | +0.131 [+0.080, +0.181] | +0.063 [+0.024, +0.103] | +0.043 [-0.220, +0.343] |
| LLM judge | -0.032 [-0.074, +0.006] | +0.049 [+0.009, +0.089] | -0.058 [-0.118, -0.005] | -0.139 [-0.516, +0.217] |
| debate | +0.003 [-0.026, +0.029] | +0.030 [-0.028, +0.089] | +0.003 [-0.030, +0.034] | -0.048 [-0.371, +0.272] |
| oracle (test S) | +0.079 [+0.052, +0.111] | +0.131 [+0.080, +0.181] | +0.068 [+0.037, +0.099] | +0.134 [-0.242, +0.538] |

Mean per-topic Spearman correlation of reward with evaluation (testbed):

| reward | held-out questions | shifted questions | held-out reader | expert rubric (1-10) |
|---|---|---|---|---|
| latent S_τ | +0.59 [+0.35, +0.75] | +0.14 [+0.09, +0.20] | +0.42 [+0.21, +0.60] | +0.10 [-0.06, +0.24] |
| LLM judge | -0.04 [-0.26, +0.17] | -0.05 [-0.15, +0.05] | -0.08 [-0.24, +0.07] | -0.00 [-0.15, +0.12] |
| debate | +0.21 [+0.04, +0.35] | -0.27 [-0.37, -0.18] | +0.03 [-0.13, +0.18] | +0.21 [+0.05, +0.37] |
| oracle (test S) | +1.00 [+1.00, +1.00] | +0.35 [+0.30, +0.39] | +0.45 [+0.28, +0.61] | +0.05 [-0.11, +0.19] |

## Planted worlds: share of the gap from the mean report to the true theory closed by best-of-12

| topic           |   mean report |   best report |   true theory |   bayes |   no theory |   latent |   judge |   debate |   oracle |
|:----------------|--------------:|--------------:|--------------:|--------:|------------:|---------:|--------:|---------:|---------:|
| planted/world_0 |        -0.786 |        -0.714 |        -0.487 |  -0.487 |      -1.513 |   -0.371 |   0.085 |    0.167 |    0.24  |
| planted/world_1 |        -0.715 |        -0.613 |        -0.514 |  -0.495 |      -1.066 |    0.335 |  -0.084 |   -0.084 |    0.508 |
| planted/world_2 |        -0.706 |        -0.557 |        -0.416 |  -0.385 |      -1.44  |    0.512 |   0.512 |    0.512 |    0.512 |
| planted/world_3 |        -0.584 |        -0.521 |        -0.451 |  -0.411 |      -0.988 |    0.044 |  -0.209 |   -0.223 |    0.47  |
| planted/world_4 |        -0.696 |        -0.637 |        -0.527 |  -0.507 |      -1.601 |   -0.063 |   0.059 |    0.058 |    0.352 |
| planted/world_5 |        -0.679 |        -0.613 |        -0.52  |  -0.518 |      -1.35  |   -0.066 |  -0.053 |    0.415 |    0.415 |
| planted/world_6 |        -0.668 |        -0.59  |        -0.499 |  -0.482 |      -1.42  |    0.433 |   0.025 |    0.433 |    0.462 |
| planted/world_7 |        -0.728 |        -0.524 |        -0.396 |  -0.387 |      -1.478 |    0.614 |   0.119 |    0.119 |    0.614 |

Mean share closed: latent S_τ +0.18, LLM judge +0.06, debate +0.17, oracle (test S) +0.45
