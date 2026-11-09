"""Synthetic worlds with known latents, for testing essay scores against ground truth (E1).

A *setting* is a fictional place with three hidden binary factors and 30 yes/no questions about the coming
year. Each question depends on exactly one factor, with probabilities fixed by design:
P(yes | factor true) and P(yes | factor false). A *world* is a setting plus a truth assignment to its factors
(all 8 are used). An *author* gets natural-language signals about the factors, with stated reliabilities;
the author's Bayesian posterior over the factors - and hence the information quality of anything they write -
is known exactly.

The entities are invented, so a reader has no outside knowledge of them: whatever it predicts beyond common
sense, it gets from the essay.
"""

import itertools
import json
import random
from dataclasses import dataclass

import numpy as np

from cvtestbed.llm import default_llm

SETTINGS = {
    "valdoria": {
        "brief": ("Valdoria is a mid-sized, largely agricultural republic with its own currency (the valdor), an "
                  "independent central bank, and a parliament governed by a three-party coalition. Wheat and "
                  "barley from the northern farm belt are its main exports. All questions concern the next "
                  "twelve months."),
        "factors": [
            ("hawkish", "Valdoria's central bank has decided to prioritise fighting inflation over supporting "
                        "growth (a hawkish turn)."),
            ("drought", "The northern farm belt is entering a severe, multi-season drought."),
            ("coalition", "The governing coalition is on the verge of collapse."),
        ],
    },
    "corvane": {
        "brief": ("Corvane Therapeutics is a mid-sized, publicly listed biotech company. Its value rests mainly on "
                  "its lead drug, Velitrex, which is in a phase 3 trial; it also has a small portfolio of early "
                  "programs. All questions concern the next twelve months."),
        "factors": [
            ("trial", "Velitrex's phase 3 trial is succeeding: unpublished interim data show a clear benefit."),
            ("cash", "Corvane is running out of cash and will need emergency funding within the year."),
            ("ceo", "Corvane's CEO has privately decided to leave the company."),
        ],
    },
}
STRONG, WEAK = (0.85, 0.20), (0.65, 0.40)  # P(yes | factor true), P(yes | factor false) for a "+" question
N_PER_FACTOR = 10

QGEN = """{brief}

Hidden fact (never mention it or hint at it in a question): "{statement}"
Other hidden facts, independent of this one, which your questions should depend on as little as possible:
{others}

Write {n} yes/no forecasting questions about concrete, observable events in the next twelve months, each
resolvable unambiguously. For questions 1-{half}, "Yes" must be clearly MORE likely if the hidden fact is true;
for questions {half1}-{n}, "Yes" must be clearly LESS likely if it is true. Vary the topics. Reply with only a
JSON list of {n} strings."""


@dataclass(frozen=True)
class Q:
    """A yes/no question in the testbed's shape (``text``, ``options``, ``answer``) plus its true model."""

    id: str
    text: str
    factor: int
    p_true: float   # P(yes | factor true)
    p_false: float  # P(yes | factor false)
    options: tuple = ("Yes", "No")
    answer: int = 0  # set per world when outcomes are drawn

    def p_yes(self, z):
        return self.p_true if z[self.factor] else self.p_false


async def questions(setting, model):
    """The setting's 30 questions (generated once by ``model``; cached)."""
    spec = SETTINGS[setting]
    llm = default_llm()
    out = []
    for k, (name, statement) in enumerate(spec["factors"]):
        others = "\n".join(f"- {s}" for j, (_, s) in enumerate(spec["factors"]) if j != k)
        half = N_PER_FACTOR // 2
        r = await llm.chat(model, QGEN.format(brief=spec["brief"], statement=statement, others=others,
                                              n=N_PER_FACTOR, half=half, half1=half + 1),
                           max_tokens=2000, temperature=0.0, key=f"qgen|{setting}|{name}")
        texts = json.loads(r.text[r.text.index("["):r.text.rindex("]") + 1])[:N_PER_FACTOR]
        for i, t in enumerate(texts):
            hi, lo = STRONG if i % half < 3 else WEAK
            pt, pf = (hi, lo) if i < half else (lo, hi)
            out.append(Q(f"{setting}/{name}/{i}", t.strip(), k, pt, pf))
    return out


def worlds(setting):
    """All 8 truth assignments of the setting's factors."""
    return [tuple(z) for z in itertools.product([1, 0], repeat=len(SETTINGS[setting]["factors"]))]


def draw(qs, z, seed):
    """One realization of the outcomes in world ``z``: questions with ``answer`` set (0 = yes)."""
    rng = random.Random(f"draw|{seed}|{z}")
    return [Q(q.id, q.text, q.factor, q.p_true, q.p_false, q.options,
              0 if rng.random() < q.p_yes(z) else 1) for q in qs]


# ------------------------------------------------------------------ authors
PROFILES = {  # per factor: (reliability, signal correct?) or None for no signal
    "uninformed": [None, None, None],
    "weak": [(0.7, True)] * 3,
    "strong": [(0.9, True)] * 3,
    "misled": [(0.9, False)] * 3,
    "partial": [(0.95, True), None, None],
}


def signals(setting, z, profile):
    """(text the author is given, the author's posterior P(factor true) for each factor)."""
    lines, post = [], []
    for k, ((_, statement), sig) in enumerate(zip(SETTINGS[setting]["factors"], PROFILES[profile])):
        if sig is None:
            post.append(0.5)
            continue
        rel, correct = sig
        says_true = bool(z[k]) == correct
        claim = statement if says_true else f"It is NOT the case that: {statement[0].lower()}{statement[1:]}"
        lines.append(f"- A source who is right about such things {round(rel * 10)} times out of 10 told you: "
                     f"{claim}")
        post.append(rel if says_true else 1 - rel)
    text = "\n".join(lines) if lines else "You have no private information beyond the background."
    return text, np.array(post)


def predictive(qs, post):
    """P(yes) for each question given independent factor probabilities ``post``."""
    return np.array([post[q.factor] * q.p_true + (1 - post[q.factor]) * q.p_false for q in qs])


def expected_log_score(qs, p, z):
    """Mean over questions of E[log p(y)] when y ~ the true model in world z (nats/question)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    t = np.array([q.p_yes(z) for q in qs])
    return float(np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def belief_expected_log_score(qs, p, post):
    """Mean expected log score of predictions ``p`` when the world is drawn from the author's posterior."""
    tot = 0.0
    for z in itertools.product([1, 0], repeat=len(post)):
        w = np.prod([pk if zk else 1 - pk for pk, zk in zip(post, z)])
        tot += w * expected_log_score(qs, p, z)
    return tot
