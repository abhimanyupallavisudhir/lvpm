"""The frozen reader: what a hypothesis (essay) predicts about a task.

The reader is a fixed language model with a fixed prompt. Given a hypothesis h
and a two-option task, it answers as if h were true; its probability for each
option is read from next-token log-probabilities. That probability is
p_R(y | h): the meaning of h, relative to this reader. The writer of h never
sees the tasks it will be scored on, so it can move these predictions only
through what h says about the domain.

Two choices keep p_R well defined:

* Position bias is removed by asking with the options in both orders and
  averaging (``debias=True``).
* Probabilities are clipped to [eps, 1 - eps], so a single confident miss
  costs at most log(1/eps) - the agreed likelihood of a settlement.
* With ``reason=True`` the reader first works the question through at
  temperature 0 (a theory that must be *applied* - a formula, a procedure -
  means nothing to a reader that cannot apply it), and then states the
  probability that each answer is correct. After reasoning, next-token
  log-probabilities are nearly always 0 or 1, which would turn every hypothesis
  into a deterministic one; a stated probability lets a theory that says "this
  is a toss-up" or "70% likely" predict softly.

Calls go through so-arena's model layer, so they are cached on disk and their
cost is recorded (``Reader.cost``).
"""

import asyncio
import re

import numpy as np

from so_arena.core.parsing import probs_from_logprobs, parse_json_object
from so_arena.core.types import GenerateOptions, Message

SYSTEM = (
    "You answer questions by applying a theory someone else wrote about the domain. "
    "Assume the theory is correct and answer as it implies. Where it does not settle "
    "the question, use your best judgment in the spirit of the theory.")
SYSTEM_BARE = "You answer questions as accurately as you can."


def task_text(item, swap=False):
    a, b = item.answers
    first, second = (b, a) if swap else (a, b)
    return (f"{item.question.strip()}\n\n(A) {first.text}\n(B) {second.text}")


class Reader:
    def __init__(self, model="openai/gpt-4.1-mini", eps=0.01, debias=True, reason=True, max_reasoning=700,
                 concurrency=32):
        from so_arena.models.base import get_model

        self.model_name = model
        self.model = get_model(model)
        self.eps, self.debias, self.reason, self.max_reasoning = eps, debias, reason, max_reasoning
        self.sem = asyncio.Semaphore(concurrency)
        self.cost, self.calls = 0.0, 0

    def describe(self):
        return {"model": self.model_name, "eps": self.eps, "debias": self.debias, "reason": self.reason}

    async def _p_first(self, hypothesis, item, swap):
        """P(the option shown first is correct)."""
        body = task_text(item, swap)
        if hypothesis:
            msgs = [Message.system(SYSTEM),
                    Message.user(f"THEORY:\n{hypothesis.strip()}\n\nQUESTION:\n{body}\n\n"
                                 "Which answer is correct according to the theory? "
                                 "Reply with only the letter A or B.")]
        else:
            msgs = [Message.system(SYSTEM_BARE),
                    Message.user(f"QUESTION:\n{body}\n\nWhich answer is correct? "
                                 "Reply with only the letter A or B.")]
        cost = 0.0
        async with self.sem:
            if self.reason:
                ask = msgs[-1].content.replace("Reply with only the letter A or B.",
                                               "Work it out step by step, briefly.")
                think = await self.model.generate([msgs[0], Message.user(ask)], GenerateOptions(
                    temperature=0.0, max_tokens=self.max_reasoning))
                cost += think.usage.cost_usd or 0.0
                self.calls += 1
                msgs = [msgs[0], Message.user(ask), Message.assistant(think.text or ""),
                        Message.user("Now give the probability that each answer is correct"
                                     + (" according to the theory" if hypothesis else "") + ".")]
            if self.model.supports_logprobs and not self.reason:
                out = await self.model.generate(msgs, GenerateOptions(
                    temperature=0.0, max_tokens=2, logprobs=True, top_logprobs=20))
                p = probs_from_logprobs(out.logprobs, ["A", "B"])
                p = p["A"] if p else 0.5
            else:  # verbalized probability
                instr = ('Reply with a JSON object {"A": p, "B": 1 - p} giving your probability that each '
                         "answer is correct, and nothing else.")
                last = msgs[-1].content
                msgs[-1] = Message.user(last + " " + instr if self.reason else
                                        last.replace("Reply with only the letter A or B.", instr))
                out = await self.model.generate(msgs, GenerateOptions(temperature=0.0, max_tokens=40))
                d = parse_json_object(out.text or "") or {}
                try:
                    p = float(d.get("A", 0.5))
                except (TypeError, ValueError):
                    p = 0.5
        self.cost += cost + (out.usage.cost_usd or 0.0)
        self.calls += 1
        return p

    async def prob(self, hypothesis, item):
        """p_R(answer A of ``item`` is correct | hypothesis), clipped."""
        p = await self._p_first(hypothesis, item, False)
        if self.debias:
            p = 0.5 * (p + 1.0 - await self._p_first(hypothesis, item, True))
        return float(np.clip(p, self.eps, 1 - self.eps))

    async def matrix(self, hypotheses, items):
        """[K, n] matrix P[k, i] = p_R(answer A of item i | hypothesis k)."""
        cells = [self.prob(h, it) for h in hypotheses for it in items]
        vals = await asyncio.gather(*cells)
        return np.array(vals, dtype=float).reshape(len(hypotheses), len(items))


def outcomes(items):
    """y_i = 1 if answer A of item i is the true one."""
    return np.array([1.0 if it.true_label == "A" else 0.0 for it in items])


WORD_RE = re.compile(r"\w+")


def words(text):
    return len(WORD_RE.findall(text or ""))
