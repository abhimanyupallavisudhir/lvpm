"""The frozen reader: what a hypothesis (essay) predicts about a task.

The reader is a fixed language model with a fixed prompt. Given a hypothesis h
and a two-option task, it answers as if h were true and states its probability
for each option. That probability is
p_R(y | h): the meaning of h, relative to this reader. The writer of h never
sees the tasks it will be scored on, so it can move these predictions only
through what h says about the domain.

Three choices keep p_R well defined:

* Position bias is removed by asking with the options in both orders and
  averaging (``debias=True``).
* Probabilities are clipped to [eps, 1 - eps], so a single confident miss
  costs at most log(1/eps) - the agreed likelihood of a settlement.
* With ``reason=True`` the reader first works the question through at
  temperature 0 (a theory that must be *applied* - a formula, a procedure -
  means nothing to a reader that cannot apply it), and then states the
  probability that each answer is correct. The reasoning budget must be large enough to finish: a reader cut
  off mid-calculation guesses, which penalizes exactly the precise theories that need working out (a 700-token
  budget truncated 46% of readings of the true planted theories). ``truncated`` counts readings that hit the
  budget; report it with every result. After reasoning, next-token
  log-probabilities are nearly always 0 or 1, which would turn every hypothesis
  into a deterministic one; a stated probability lets a theory that says "this
  is a toss-up" or "70% likely" predict softly.

Calls go through the testbed's cached LLM client (``cvtestbed.llm``), so they
are cached on disk and their cost is recorded (``Reader.cost``).
"""

import asyncio
import re

import numpy as np

from cvtestbed.llm import LLM

SYSTEM = (
    "You answer questions by applying a theory someone else wrote about the domain. "
    "Assume the theory is correct and answer as it implies. Where it does not settle "
    "the question, use your best judgment in the spirit of the theory.")
SYSTEM_BARE = "You answer questions as accurately as you can."
PROB = ('Reply with a JSON object {"A": p, "B": 1 - p} giving your probability that each answer is correct, '
        "and nothing else.")


def task_text(q, swap=False):
    a, b = q.options
    first, second = (b, a) if swap else (a, b)
    return f"{q.text.strip()}\n\n(A) {first}\n(B) {second}"


def parse_prob(text):
    m = re.search(r'"A"\s*:\s*([0-9.]+)', text or "")
    try:
        return float(m.group(1)) if m else 0.5
    except ValueError:
        return 0.5


class Reader:
    """A frozen reader: ``model`` answering as if a hypothesis were true (see the module docstring)."""

    def __init__(self, model="openai/gpt-4.1-mini", eps=0.01, debias=True, reason=True, max_reasoning=4000,
                 llm=None):
        self.model_name, self.eps, self.debias = model, eps, debias
        self.reason, self.max_reasoning = reason, max_reasoning
        self.llm = llm or LLM()
        self.cost, self.calls = 0.0, 0
        self.readings, self.truncated = 0, 0

    def describe(self):
        return {"model": self.model_name, "eps": self.eps, "debias": self.debias, "reason": self.reason}

    async def _chat(self, msgs, max_tokens, key):
        r = await self.llm.chat(self.model_name, msgs, max_tokens=max_tokens, temperature=0.0, key=key)
        self.cost += r.cost
        self.calls += 1
        if key.startswith("think|"):
            self.readings += 1
            self.truncated += int(r.tokens_out >= max_tokens - 2)
        return r.text

    @property
    def truncation_rate(self):
        return self.truncated / max(1, self.readings)

    async def _p_first(self, hypothesis, q, swap):
        """P(the option shown first is correct)."""
        body = task_text(q, swap)
        if hypothesis:
            sys_ = SYSTEM
            ask = f"THEORY:\n{hypothesis.strip()}\n\nQUESTION:\n{body}\n\nWhich answer is correct according to the theory?"
        else:
            sys_ = SYSTEM_BARE
            ask = f"QUESTION:\n{body}\n\nWhich answer is correct?"
        key = f"{q.id}|{swap}"  # the cache key also covers the messages, so each hypothesis is its own draw
        msgs = [{"role": "system", "content": sys_}]
        if self.reason:
            msgs.append({"role": "user", "content": ask + " Work it out step by step, briefly."})
            think = await self._chat(msgs, self.max_reasoning, "think|" + key)
            msgs += [{"role": "assistant", "content": think},
                     {"role": "user", "content": "Now give the probability that each answer is correct"
                      + (" according to the theory" if hypothesis else "") + ". " + PROB}]
        else:
            msgs.append({"role": "user", "content": ask + " " + PROB})
        return parse_prob(await self._chat(msgs, 40, "prob|" + key))

    async def prob(self, hypothesis, q):
        """p_R(option A of question ``q`` is correct | hypothesis), clipped."""
        p = await self._p_first(hypothesis, q, False)
        if self.debias:
            p = 0.5 * (p + 1.0 - await self._p_first(hypothesis, q, True))
        return float(np.clip(p, self.eps, 1 - self.eps))

    async def matrix(self, hypotheses, questions):
        """[K, n] matrix P[k, i] = p_R(option A of question i | hypothesis k)."""
        cells = [self.prob(h, q) for h in hypotheses for q in questions]
        vals = await asyncio.gather(*cells)
        return np.array(vals, dtype=float).reshape(len(hypotheses), len(questions))


def outcomes(questions):
    """y_i = 1 if option A of question i is the correct one."""
    return np.array([1.0 if q.answer == 0 else 0.0 for q in questions])


WORD_RE = re.compile(r"\w+")


def words(text):
    return len(WORD_RE.findall(text or ""))
