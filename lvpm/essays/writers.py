"""Writers: produce reports (one essay, or several weighted hypotheses) about a family.

A writer sees the family's card - what kind of questions the family asks,
never an outcome - and optionally some *resolved* training items (questions
with their correct answers): the position of a researcher theorizing from past
experiments. It commits a report before the test items are drawn. Baselines:

* ``empty``: no essay; the reader's own predictions.
* ``examples``: the resolved training items themselves, verbatim, as the essay -
  memorized data rather than a theory (a cheat sheet that can only help on new
  items through the reader's in-context generalization).
"""

import json
import re

from cvtestbed.llm import default_llm

ESSAY_SYSTEM = (
    "You are a researcher writing a short theory of a domain. Another model will read your theory "
    "and use it - and nothing else from you - to answer new questions from the domain that neither of "
    "you has seen. Your theory is scored by how well that reader predicts the true answers, using a "
    "proper scoring rule: being confidently wrong is costly, so say where you are uncertain.")


def _examples_block(items, limit_chars=60000):
    out, n = [], 0
    for q in items:
        block = f"QUESTION:\n{q.text.strip()}\nCORRECT ANSWER: {q.options[q.answer]}"
        n += len(block)
        if n > limit_chars:
            break
        out.append(block)
    return "\n\n---\n\n".join(out)


def examples_essay(items):
    return ("Resolved past questions from this domain, with their correct answers:\n\n"
            + _examples_block(items))


def essay_prompt(card, train_items=None, words=400):
    data = ("" if not train_items else
            "\n\nHere are resolved questions from this domain, with their correct answers. Your theory "
            "should explain them and generalize to new questions; do not just list them.\n\n"
            + _examples_block(train_items))
    return (f"THE DOMAIN:\n{card}{data}\n\nWrite your theory (at most {words} words): the general "
            "principles that determine the answers, how to apply them, and how confident to be. "
            "Reply with the theory only.")


def mixture_prompt(card, train_items=None, k=3, words=200):
    data = ("" if not train_items else
            "\n\nResolved questions from this domain, with their correct answers:\n\n"
            + _examples_block(train_items))
    return (f"THE DOMAIN:\n{card}{data}\n\nWrite {k} competing hypotheses about what determines the "
            f"answers in this domain - genuinely different theories, each at most {words} words and each "
            "usable on its own to answer new questions - and your probability that each is the best "
            "description. Reply with only a JSON object: {\"hypotheses\": [{\"text\": \"...\", "
            "\"weight\": p}, ...]}.")


async def write_essay(model, card, train_items=None, *, words=400, sample=0, temperature=1.0, max_tokens=1500,
                      llm=None):
    llm = llm or default_llm()
    r = await llm.chat(model, [{"role": "system", "content": ESSAY_SYSTEM},
                               {"role": "user", "content": essay_prompt(card, train_items, words)}],
                       max_tokens=max_tokens, temperature=temperature, key=f"essay|{sample}")
    return r.text.strip(), r.cost


def _last_json(text):
    for cand in reversed(re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


async def write_mixture(model, card, train_items=None, *, k=3, words=200, sample=0, temperature=1.0,
                        max_tokens=2500, llm=None):
    llm = llm or default_llm()
    r = await llm.chat(model, [{"role": "system", "content": ESSAY_SYSTEM},
                               {"role": "user", "content": mixture_prompt(card, train_items, k, words)}],
                       max_tokens=max_tokens, temperature=temperature, key=f"mixture|{sample}")
    hyps = []
    for h in (_last_json(r.text) or {}).get("hypotheses", []):
        try:
            hyps.append((str(h["text"]).strip(), max(0.0, float(h.get("weight", 1.0)))))
        except (KeyError, TypeError, ValueError):
            continue
    if not hyps or sum(w for _, w in hyps) <= 0:
        return [], r.cost
    tot = sum(w for _, w in hyps)
    return [(t, w / tot) for t, w in hyps], r.cost


JUDGE_PROMPT = (
    "Rate the following theory of a domain for how good it is as a theory: correct, insightful, general, "
    "clear, and useful for answering new questions. Domain:\n{card}\n\nTHEORY:\n{essay}\n\nEnd with a line "
    "'Score: N' where N is from 1 to 10.")


async def judge_essay(model, card, essay, sample=0, llm=None):
    """An LLM's rating of an essay's quality (1-10): the alternative training signal to compare S_tau with."""
    llm = llm or default_llm()
    r = await llm.chat(model, JUDGE_PROMPT.format(card=card, essay=essay), max_tokens=600, temperature=0.0,
                       key=f"judge|{sample}")
    m = re.findall(r"score\W{0,3}(\d+(?:\.\d+)?)", r.text, re.I)
    s = float(m[-1]) if m else 5.0
    return min(10.0, max(1.0, s)), r.cost
