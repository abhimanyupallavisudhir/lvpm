"""Planted worlds for RL: prompts, reward questions and true conditionals for any world index.

Training uses worlds 8..(8+N); worlds 0..7 are the held-out evaluation worlds of the testbed's ``planted/*``
families. Everything here is pure Python (no model calls), so it runs inside the training container.
"""

from __future__ import annotations

import random

from cvtestbed.domains import planted as P

REPORT_INSTRUCTIONS = (
    "Write 3 competing hypotheses about what determines the test results in this world - genuinely different "
    "theories, each at most 200 words and each usable on its own to predict new crystals - and your probability that "
    "each is the best description. Reply with only a JSON object: {\"hypotheses\": [{\"text\": \"...\", "
    "\"weight\": p}, ...]}.")
SYSTEM = ("You are a scientist writing a theory of a domain. Another model will read your theory and use it - and "
          "nothing else from you - to predict new cases. Your theory is scored by how well that reader predicts the "
          "true outcomes, using a proper scoring rule: being confidently wrong is costly.")


def prompt(k: int) -> list[dict]:
    user = (f"THE DOMAIN:\n{P.CARD}\n\nLAB NOTEBOOK (fully tested crystals; 1 = passed):\n{P.notebook_text(k)}\n\n"
            f"{REPORT_INSTRUCTIONS}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def questions(k: int, n: int, *, seed: str = "reward", shifted: bool | None = False) -> list[dict]:
    """``n`` questions about new crystals of world ``k``: text, P(yes) under the truth, and a sampled outcome."""
    w = P.world(k)
    rng = random.Random(f"rl-questions|{k}|{seed}")
    out = []
    while len(out) < n:
        sh = rng.random() < 0.25 if shifted is None else shifted
        c = P.sample_crystal(w, rng, shifted=sh)
        test = rng.choice(list(P.TESTS))
        others = [t for t in P.TESTS if t != test]
        observed = {t: c["results"][t] for t in rng.sample(others, rng.choice([0, 1, 1, 2]))}
        out.append({"text": f"{P.describe(c, observed)} Will it {P.TESTS[test]}?",
                    "p_yes": P.bayes_p(w, c, observed, test), "yes": bool(c["results"][test]), "shifted": sh})
    return out
