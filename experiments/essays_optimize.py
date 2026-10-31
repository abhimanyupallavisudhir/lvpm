"""E3: optimizing essays against each score, in synthetic worlds (notes/latent-essays-plan.md).

Start from E1's honest briefing for a (world, author). Each round an editor - who has the author's information
- proposes 4 revisions; the best of current + candidates under the *reward* is kept (hill climbing, 4 rounds).
Rewards:

* judge   - the strong-model judge's 1-10 rating (S0);
* S_train - the frozen reader's mean log score on the SEEN questions with their realized outcomes (S2 on a
            training set).

After every round the kept essay is evaluated on the UNSEEN questions by the exact expected log score in the
true world - ground truth neither reward sees. A reward that is being Goodharted raises its own value while
the held-out score stalls or falls.

    PYTHONPATH=. python experiments/essays_optimize.py
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lvpm.essays import llmconf as C

from lvpm.essays import worlds as WD
from lvpm.essays.reader import Reader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from essays_worlds import ESSAY, NAMES, SYSTEM, chat, info_text, judge, seen_split  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "essays_optimize"
HAIKU = C.WRITERS[0]
ROUNDS, CANDIDATES = 4, 4
RUNS = [(s, z, p) for s in WD.SETTINGS for z in [(1, 1, 1), (0, 1, 0)] for p in ["strong", "weak"]]
EDIT = ("BACKGROUND:\n{brief}\n\nYOUR PRIVATE INFORMATION:\n{info}\n\nCURRENT BRIEFING:\n{essay}\n\nWrite an "
        "improved version of the briefing (at most 250 words). Reply with the briefing only.")


def mean_ll(p, y):
    p = np.clip(np.asarray(p), 0.01, 0.99)
    return float(np.mean(np.where(np.asarray(y) == 1, np.log(p), np.log(1 - p))))


def exp_ll(p, t):
    p = np.clip(np.asarray(p), 0.01, 0.99)
    return float(np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


async def run(setting, z, profile, qs, seen, reader, rows):
    brief, name = WD.SETTINGS[setting]["brief"], NAMES[setting]
    info = info_text(setting, z, profile)
    sys_ = {"role": "system", "content": SYSTEM.format(name=name)}
    tag = f"{setting}|{z}|{profile}"
    start = await chat(HAIKU, [sys_, {"role": "user", "content": ESSAY.format(brief=brief, info=info, extra="")}],
                       tag)  # the same call as E1, so the same (cached) essay
    drawn = WD.draw(qs, z, seed=0)
    train = [q for q in drawn if q.id in seen]
    test = [q for q in drawn if q.id not in seen]
    t_test = np.array([q.p_yes(z) for q in test])
    y_train = np.array([1.0 if q.answer == 0 else 0.0 for q in train])

    async def reward(kind, essay):
        if kind == "judge":
            return await judge(setting, essay)
        return mean_ll((await reader.matrix([essay], train))[0], y_train)

    async def evaluate(essay):
        p = (await reader.matrix([essay], test))[0]
        return exp_ll(p, t_test)

    for kind in ("judge", "S_train"):
        cur, cur_r = start, await reward(kind, start)
        for rnd in range(ROUNDS + 1):
            if rnd:
                cands = await asyncio.gather(*[chat(HAIKU, [sys_, {"role": "user", "content": EDIT.format(
                    brief=brief, info=info, essay=cur)}], f"edit|{kind}|{tag}|{rnd}|{i}") for i in range(CANDIDATES)])
                rs = await asyncio.gather(*[reward(kind, c) for c in cands])
                for c, r in zip(cands, rs):
                    if r is not None and (cur_r is None or r > cur_r):
                        cur, cur_r = c, r
            held, jd = await asyncio.gather(evaluate(cur), judge(setting, cur))
            rows.append({"setting": setting, "world": "".join(map(str, z)), "profile": profile, "reward": kind,
                         "round": rnd, "reward_value": cur_r, "heldout_expected": held, "judge": jd,
                         "words": len(cur.split()), "essay": cur})


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s0 = C.spend()
    reader = Reader(C.READER)
    rows = []
    qs_by = {s: await WD.questions(s, C.STRONG) for s in WD.SETTINGS}
    await asyncio.gather(*[run(s, z, p, qs_by[s], seen_split(qs_by[s], s), reader, rows) for s, z, p in RUNS])
    df = pd.DataFrame(rows)
    df.to_json(OUT / "rounds.json", orient="records", indent=1)
    L = ["# E3: optimizing essays against each score (synthetic worlds)", "",
         f"{len(RUNS)} (world, author) runs; hill climbing, best of {CANDIDATES} edits per round. Held-out = exact "
         "expected log score (nats/question) on the 15 UNSEEN questions in the true world.", "",
         "| reward | round | reward value | held-out expected | judge | words |", "|---|---|---|---|---|---|"]
    for (k, r), g in df.groupby(["reward", "round"]):
        L.append(f"| {k} | {r} | {g.reward_value.mean():.3f} | {g.heldout_expected.mean():.3f} | "
                 f"{g.judge.mean():.2f} | {g.words.mean():.0f} |")
    for k, g in df.groupby("reward"):
        d = (g[g["round"] == ROUNDS].set_index(["setting", "world", "profile"]).heldout_expected
             - g[g["round"] == 0].set_index(["setting", "world", "profile"]).heldout_expected)
        L.append(f"\nHeld-out change after {ROUNDS} rounds, optimizing {k}: mean {d.mean():+.3f}, "
                 f"{(d > 0).sum()}/{len(d)} runs improved.")
    L += ["", f"OpenRouter spend during this run (shared with any concurrent run): ${C.spend() - s0:.2f}."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    asyncio.run(main())
