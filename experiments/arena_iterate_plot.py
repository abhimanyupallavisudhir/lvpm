"""Goodhart curves for the in-context expert-iteration run: per chain, the reward of each round's new reports
(z-scored on round 0) against their held-out score (gain over the round-0 mean), with 95% CIs over topics.

    PYTHONPATH=. python experiments/arena_iterate_plot.py --dir results/arena-iterate
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLOURS = {"latent": "#2a78d6", "judge": "#eb6834", "debate": "#1baf7a"}
NAMES = {"latent": "latent S_τ", "judge": "LLM judge", "debate": "debate"}


def ci(x):
    x = np.asarray(x, dtype=float)
    m = np.random.default_rng(0).choice(x, (4000, len(x))).mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def main(d: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z = pd.read_csv(d / "iterate_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (grp, title) in zip(axes, [("planted", "Planted worlds (8)"), ("testbed", "Testbed families (11)")]):
        ax.axhline(0, color="#b5b4ae", lw=1, ls=(0, (3, 3)))
        ends = []
        for chain in COLOURS:
            s = Z[(Z.group == grp) & (Z.chain == chain)]
            pts = [(ci(g.new_reward_z)[0], *ci(g.new_eval)) for _, g in s.groupby("round")]
            x, y, lo, hi = map(np.array, zip(*pts))
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=COLOURS[chain], lw=2, marker="o", ms=6,
                        mec="#fcfcfb", mew=1.5, elinewidth=1, capsize=0, label=NAMES[chain])
            ends.append([y[-1], x[-1], NAMES[chain].split(" ")[0]])
        ax.annotate("round 0", (0, 0), xytext=(4, 8), textcoords="offset points", fontsize=7, color="#52514e")
        lo_, hi_ = ax.get_ylim()
        ends.sort()
        for a in range(1, len(ends)):
            ends[a][0] = max(ends[a][0], ends[a - 1][0] + 0.07 * (hi_ - lo_))
        for y, x, name in ends:
            ax.annotate(name, (x, y), xytext=(8, 0), textcoords="offset points", fontsize=8, color="#52514e",
                        va="center")
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.set_xlabel("reward of new reports (z, vs round 0); points are rounds 0-3", fontsize=9, color="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b5b4ae")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
    axes[0].set_ylabel("held-out score of new reports\n(nats/question vs round 0)", fontsize=9, color="#52514e")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(d / "goodhart.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/arena-iterate")
    main(Path(ap.parse_args().dir))
