# Essays as latents: first results

Reader: {'model': 'openai/gpt-4.1-mini', 'eps': 0.01, 'debias': True, 'reason': True}. Judge: anthropic/claude-sonnet-4-5 (cost $0.00).

## Held-out value of reports (test items, mean log score per item, nats)

| family             | kind       |   n |   test_log |   test_acc |    words |   gain_vs_empty |
|:-------------------|:-----------|----:|-----------:|-----------:|---------:|----------------:|
| agents/auction     | empty      |   1 |     -1.671 |      0.188 |    0     |           0     |
| agents/auction     | examples   |   1 |     -0.673 |      0.75  | 4489     |           0.997 |
| agents/auction     | informed   |   6 |     -0.572 |      0.708 |  548.333 |           1.098 |
| agents/auction     | mixture    |   4 |     -0.66  |      0.547 |  375.75  |           1.011 |
| agents/auction     | uninformed |   5 |     -1.219 |      0.325 |  438     |           0.451 |
| agents/persona     | empty      |   1 |     -1.037 |      0.438 |    0     |           0     |
| agents/persona     | examples   |   1 |     -0.846 |      0.75  | 2250     |           0.19  |
| agents/persona     | informed   |   7 |     -1.049 |      0.518 |  436.714 |          -0.012 |
| agents/persona     | mixture    |   6 |     -0.973 |      0.396 |  299.833 |           0.063 |
| agents/persona     | uninformed |   6 |     -1.122 |      0.25  |  429.167 |          -0.086 |
| agents/sycophancy  | empty      |   1 |     -0.216 |      0.875 |    0     |           0     |
| agents/sycophancy  | examples   |   1 |     -0.346 |      0.812 | 2180     |          -0.13  |
| agents/sycophancy  | informed   |   7 |     -0.271 |      0.902 |  399.143 |          -0.055 |
| agents/sycophancy  | mixture    |   6 |     -0.268 |      0.917 |  351.167 |          -0.053 |
| agents/sycophancy  | uninformed |   6 |     -0.296 |      0.906 |  411     |          -0.08  |
| decisions/newcomb  | empty      |   1 |     -0.315 |      0.812 |    0     |           0     |
| decisions/newcomb  | examples   |   1 |     -0.124 |      0.875 | 3219     |           0.191 |
| decisions/newcomb  | informed   |   9 |     -0.221 |      0.931 |  496.778 |           0.094 |
| decisions/newcomb  | mixture    |   4 |     -0.28  |      0.875 |  397     |           0.034 |
| decisions/newcomb  | uninformed |   6 |     -0.267 |      0.854 |  410     |           0.047 |
| numbers/prime_race | empty      |   1 |     -0.739 |      0.688 |    0     |           0     |
| numbers/prime_race | examples   |   1 |     -0.543 |      0.812 |  691     |           0.196 |
| numbers/prime_race | informed   |   7 |     -0.494 |      0.741 |  485.714 |           0.245 |
| numbers/prime_race | mixture    |   6 |     -0.598 |      0.688 |  397.333 |           0.141 |
| numbers/prime_race | uninformed |   5 |     -0.479 |      0.7   |  433.8   |           0.26  |
| rlhack/fake_goal   | empty      |   1 |     -0.469 |      0.812 |    0     |           0     |
| rlhack/fake_goal   | examples   |   1 |     -0.783 |      0.562 | 3054     |          -0.314 |
| rlhack/fake_goal   | informed   |   9 |     -0.538 |      0.757 |  455.444 |          -0.069 |
| rlhack/fake_goal   | mixture    |   6 |     -0.549 |      0.719 |  374.667 |          -0.079 |
| rlhack/fake_goal   | uninformed |   6 |     -0.534 |      0.781 |  436     |          -0.065 |
| rlhack/sensor      | empty      |   1 |     -0.557 |      0.75  |    0     |           0     |
| rlhack/sensor      | examples   |   1 |     -0.681 |      0.75  | 3435     |          -0.124 |
| rlhack/sensor      | informed   |   9 |     -0.593 |      0.715 |  483     |          -0.036 |
| rlhack/sensor      | mixture    |   6 |     -0.513 |      0.729 |  395.667 |           0.044 |
| rlhack/sensor      | uninformed |   6 |     -0.555 |      0.729 |  386     |           0.002 |

## Gain over the no-essay reader, with 95% bootstrap CIs over test items (nats per item)

| family             | kind       |   gain |     lo |     hi |   items |
|:-------------------|:-----------|-------:|-------:|-------:|--------:|
| agents/auction     | examples   |  0.997 |  0.136 |  1.706 |      16 |
| agents/auction     | informed   |  1.098 |  0.627 |  1.496 |      16 |
| agents/auction     | mixture    |  1.011 |  0.637 |  1.35  |      16 |
| agents/auction     | uninformed |  0.451 |  0.227 |  0.667 |      16 |
| agents/persona     | examples   |  0.19  | -0.675 |  0.914 |      16 |
| agents/persona     | informed   | -0.012 | -0.571 |  0.553 |      16 |
| agents/persona     | mixture    |  0.063 | -0.275 |  0.406 |      16 |
| agents/persona     | uninformed | -0.086 | -0.353 |  0.184 |      16 |
| agents/sycophancy  | examples   | -0.13  | -0.429 |  0.11  |      16 |
| agents/sycophancy  | informed   | -0.055 | -0.201 |  0.082 |      16 |
| agents/sycophancy  | mixture    | -0.053 | -0.171 |  0.038 |      16 |
| agents/sycophancy  | uninformed | -0.08  | -0.144 | -0.034 |      16 |
| decisions/newcomb  | examples   |  0.191 | -0.023 |  0.479 |      16 |
| decisions/newcomb  | informed   |  0.094 | -0.172 |  0.41  |      16 |
| decisions/newcomb  | mixture    |  0.034 | -0.033 |  0.103 |      16 |
| decisions/newcomb  | uninformed |  0.047 | -0.128 |  0.274 |      16 |
| numbers/prime_race | examples   |  0.196 | -0.154 |  0.614 |      16 |
| numbers/prime_race | informed   |  0.245 | -0.141 |  0.637 |      16 |
| numbers/prime_race | mixture    |  0.141 | -0.2   |  0.524 |      16 |
| numbers/prime_race | uninformed |  0.26  | -0.11  |  0.645 |      16 |
| rlhack/fake_goal   | examples   | -0.314 | -0.759 |  0.09  |      16 |
| rlhack/fake_goal   | informed   | -0.069 | -0.243 |  0.089 |      16 |
| rlhack/fake_goal   | mixture    | -0.079 | -0.24  |  0.082 |      16 |
| rlhack/fake_goal   | uninformed | -0.065 | -0.19  |  0.076 |      16 |
| rlhack/sensor      | examples   | -0.124 | -0.573 |  0.293 |      16 |
| rlhack/sensor      | informed   | -0.036 | -0.189 |  0.111 |      16 |
| rlhack/sensor      | mixture    |  0.044 | -0.126 |  0.225 |      16 |
| rlhack/sensor      | uninformed |  0.002 | -0.134 |  0.137 |      16 |

## Mixtures: prior vs updated on dev outcomes

| family             | writer                      |   test_log |   test_log_updated_tau1 |   test_log_updated_tau0.9 |   I_joint_bits |   post_max_weight |
|:-------------------|:----------------------------|-----------:|------------------------:|--------------------------:|---------------:|------------------:|
| rlhack/sensor      | openai/gpt-4.1-mini         |     -0.423 |                  -0.375 |                    -0.379 |          1.335 |             0.504 |
| rlhack/sensor      | openai/gpt-4.1-mini         |     -0.495 |                  -0.43  |                    -0.429 |          1.234 |             0.976 |
| rlhack/sensor      | openai/gpt-5-mini           |     -0.542 |                  -0.336 |                    -0.337 |          1.109 |             0.994 |
| rlhack/sensor      | openai/gpt-5-mini           |     -0.644 |                  -0.633 |                    -0.631 |          0.785 |             0.947 |
| rlhack/sensor      | anthropic/claude-sonnet-4-5 |     -0.516 |                  -0.559 |                    -0.555 |          1.093 |             0.757 |
| rlhack/sensor      | anthropic/claude-sonnet-4-5 |     -0.458 |                  -0.451 |                    -0.451 |          0.922 |             0.674 |
| rlhack/fake_goal   | openai/gpt-4.1-mini         |     -0.732 |                  -0.703 |                    -0.704 |          1.115 |             0.473 |
| rlhack/fake_goal   | openai/gpt-4.1-mini         |     -0.493 |                  -0.467 |                    -0.468 |          1.026 |             0.981 |
| rlhack/fake_goal   | openai/gpt-5-mini           |     -0.561 |                  -0.478 |                    -0.475 |          1.236 |             0.671 |
| rlhack/fake_goal   | openai/gpt-5-mini           |     -0.372 |                  -0.395 |                    -0.391 |          1.118 |             0.663 |
| rlhack/fake_goal   | anthropic/claude-sonnet-4-5 |     -0.691 |                  -0.671 |                    -0.673 |          1.05  |             0.476 |
| rlhack/fake_goal   | anthropic/claude-sonnet-4-5 |     -0.443 |                  -0.485 |                    -0.484 |          0.818 |             0.813 |
| agents/auction     | openai/gpt-4.1-mini         |     -0.598 |                  -0.511 |                    -0.511 |          1.363 |             0.572 |
| agents/auction     | openai/gpt-4.1-mini         |     -0.735 |                  -0.606 |                    -0.609 |          1.288 |             0.875 |
| agents/auction     | anthropic/claude-sonnet-4-5 |     -0.778 |                  -0.66  |                    -0.658 |          1.403 |             0.99  |
| agents/auction     | anthropic/claude-sonnet-4-5 |     -0.528 |                  -0.373 |                    -0.373 |          1.315 |             1     |
| agents/sycophancy  | openai/gpt-4.1-mini         |     -0.364 |                  -0.28  |                    -0.282 |          1.061 |             0.966 |
| agents/sycophancy  | openai/gpt-4.1-mini         |     -0.381 |                  -0.266 |                    -0.267 |          0.525 |             0.989 |
| agents/sycophancy  | openai/gpt-5-mini           |     -0.285 |                  -0.283 |                    -0.282 |          0.611 |             0.894 |
| agents/sycophancy  | openai/gpt-5-mini           |     -0.217 |                  -0.152 |                    -0.152 |          0.835 |             1     |
| agents/sycophancy  | anthropic/claude-sonnet-4-5 |     -0.196 |                  -0.146 |                    -0.149 |          0.472 |             0.513 |
| agents/sycophancy  | anthropic/claude-sonnet-4-5 |     -0.166 |                  -0.151 |                    -0.15  |          0.587 |             0.937 |
| agents/persona     | openai/gpt-4.1-mini         |     -1.205 |                  -1.363 |                    -1.341 |          1.141 |             0.877 |
| agents/persona     | openai/gpt-4.1-mini         |     -1.278 |                  -0.915 |                    -0.93  |          1.222 |             0.839 |
| agents/persona     | openai/gpt-5-mini           |     -0.81  |                  -0.579 |                    -0.579 |          1.302 |             1     |
| agents/persona     | openai/gpt-5-mini           |     -1.03  |                  -1.109 |                    -1.102 |          1.234 |             0.932 |
| agents/persona     | anthropic/claude-sonnet-4-5 |     -0.904 |                  -1.133 |                    -1.112 |          1.387 |             0.932 |
| agents/persona     | anthropic/claude-sonnet-4-5 |     -0.613 |                  -0.769 |                    -0.75  |          1.184 |             0.849 |
| numbers/prime_race | openai/gpt-4.1-mini         |     -0.535 |                  -0.52  |                    -0.52  |          0.77  |             0.992 |
| numbers/prime_race | openai/gpt-4.1-mini         |     -0.563 |                  -0.538 |                    -0.54  |          0.492 |             0.916 |
| numbers/prime_race | openai/gpt-5-mini           |     -0.802 |                  -0.744 |                    -0.742 |          0.743 |             0.894 |
| numbers/prime_race | openai/gpt-5-mini           |     -0.551 |                  -0.495 |                    -0.496 |          0.733 |             0.948 |
| numbers/prime_race | anthropic/claude-sonnet-4-5 |     -0.566 |                  -0.514 |                    -0.515 |          1.087 |             0.999 |
| numbers/prime_race | anthropic/claude-sonnet-4-5 |     -0.572 |                  -0.561 |                    -0.561 |          0.899 |             1     |
| decisions/newcomb  | openai/gpt-4.1-mini         |     -0.253 |                  -0.261 |                    -0.259 |          0.705 |             0.49  |
| decisions/newcomb  | openai/gpt-4.1-mini         |     -0.271 |                  -0.252 |                    -0.253 |          1.002 |             0.65  |
| decisions/newcomb  | anthropic/claude-sonnet-4-5 |     -0.264 |                  -0.338 |                    -0.33  |          0.893 |             0.977 |
| decisions/newcomb  | anthropic/claude-sonnet-4-5 |     -0.333 |                  -0.095 |                    -0.095 |          0.738 |             1     |

## Does the training signal track held-out value? (informed essays)

| family             |   spearman(judge, test) |   spearman(dev S, test) |
|:-------------------|------------------------:|------------------------:|
| agents/auction     |                  -0.029 |                  -0.371 |
| agents/persona     |                   0.867 |                   0.929 |
| agents/sycophancy  |                   0.299 |                   0.536 |
| decisions/newcomb  |                   0.317 |                   0.966 |
| numbers/prime_race |                   0.318 |                   0     |
| rlhack/fake_goal   |                  -0.47  |                  -0.533 |
| rlhack/sensor      |                   0.187 |                  -0.35  |

## Best-of-all selection (k = all candidates)

| family             |   k |   by_dev_S |   by_judge |   random |
|:-------------------|----:|-----------:|-----------:|---------:|
| agents/auction     |   6 |     -0.923 |     -0.923 |   -0.572 |
| agents/sycophancy  |   7 |     -0.068 |     -0.173 |   -0.271 |
| agents/persona     |   7 |     -0.852 |     -0.852 |   -1.049 |
| numbers/prime_race |   7 |     -0.513 |     -0.477 |   -0.494 |
| decisions/newcomb  |   9 |     -0.01  |     -0.01  |   -0.221 |
| rlhack/fake_goal   |   9 |     -0.675 |     -0.675 |   -0.538 |
| rlhack/sensor      |   9 |     -0.605 |     -0.427 |   -0.593 |
