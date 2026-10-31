"""E1: essay scores against known latent truth, in synthetic worlds (notes/latent-essays-plan.md).

For each setting (lvpm/essays/worlds.py), each of its 8 worlds and each author profile, writers produce:

* honest_haiku, honest_gemini  - a briefing from the author's signals (two writer models);
* eloquent      - honest_haiku rewritten in polished, authoritative prose, same claims and confidence;
* overconfident - the same author told to state every conclusion as certain (a distortion);
* list          - the author's probabilities for the SEEN half of the questions, as a list (no mechanism);
* scenarios     - 2-4 weighted competing scenarios (a mixture report).

Baselines: empty (reader alone) and truth (the world's factors stated as facts). Every hypothesis is read by
the frozen reader on all 30 questions, one call per (hypothesis, question). Also:

* holistic - the reader given an essay plus the realized outcomes of the SEEN questions, predicting the rest
             (what a joint query lets the reader do with outcomes);
* judge    - a strong model's 1-10 rating of each essay (the S0 baseline);
* parse    - the reader's reading of each hypothesis as probabilities of the three factors, for the
             true-link ablation.

Outputs in results/essays_worlds/. Analysis: experiments/essays_worlds_report.py.

    PYTHONPATH=. python experiments/essays_worlds.py [--pilot]
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import pandas as pd

from lvpm.essays import llmconf as C

from cvtestbed.llm import default_llm

from lvpm.essays import worlds as WD
from lvpm.essays.reader import Reader

OUT = Path(__file__).resolve().parents[1] / "results" / "essays_worlds"
HAIKU, GEMINI = C.WRITERS
SYSTEM = ("You are an analyst writing a short briefing about {name}. Another model will read your briefing - and "
          "nothing else from you - and use it to answer forecasting questions about {name} that neither of you "
          "has seen. You are scored by how well that reader predicts what actually happens, with a proper "
          "scoring rule: being confidently wrong is costly, so convey exactly how confident you are.")
ESSAY = ("BACKGROUND:\n{brief}\n\nYOUR PRIVATE INFORMATION:\n{info}\n\nWrite your briefing (at most 250 words): "
         "what you believe about the current situation, what it implies for the coming year, and how confident "
         "you are.{extra} Reply with the briefing only.")
DECISIVE = (" Your editor insists that briefings sound decisive: state each of your conclusions as certain, "
            "with no hedging, even where your information is uncertain.")
LIST = ("BACKGROUND:\n{brief}\n\nYOUR PRIVATE INFORMATION:\n{info}\n\nGive your probability that the answer is "
        "Yes for each of these questions. Reply with only the numbered list, one line per question in the form "
        "'question - p'.\n\n{qs}")
SCEN = ("BACKGROUND:\n{brief}\n\nYOUR PRIVATE INFORMATION:\n{info}\n\nWrite 2 to 4 competing scenarios for the "
        "current situation - genuinely different, each at most 120 words and usable on its own to answer "
        "forecasting questions - and your probability for each. Reply with only a JSON object: "
        "{{\"hypotheses\": [{{\"text\": \"...\", \"weight\": p}}, ...]}}.")
RESTYLE = ("Rewrite this briefing in polished, eloquent, authoritative prose, as a senior analyst would. Keep "
           "every claim, every condition and exactly the same stated confidence; add no information and remove "
           "none. Reply with the rewritten briefing only.\n\nBRIEFING:\n{essay}")
JUDGE = ("Here is a briefing about {name}, written to help forecast events in the coming year.\n\nBACKGROUND:\n"
         "{brief}\n\nBRIEFING:\n{essay}\n\nRate the briefing's quality as an analysis: insight, soundness, "
         "calibration and usefulness for forecasting. End with a line 'Score: N' where N is from 1 to 10.")
PARSE = ("Read this text about {name} and say how probable it makes each statement below. Reply with only a "
         "JSON list of three probabilities.\n\nTEXT:\n{essay}\n\nSTATEMENTS:\n{statements}")
NAMES = {"valdoria": "Valdoria", "corvane": "Corvane Therapeutics"}


def info_text(setting, z, profile):
    return WD.signals(setting, z, profile)[0]


def seen_split(qs, setting):
    """A fixed half of the questions (5 per factor): SEEN by list writers, revealed to the holistic reader."""
    rng = random.Random(f"seen|{setting}")
    seen = set()
    for k in range(3):
        ids = [q.id for q in qs if q.factor == k]
        seen |= set(rng.sample(ids, len(ids) // 2))
    return seen


async def chat(model, msgs, key, max_tokens=1500, temperature=1.0):
    r = await default_llm().chat(model, msgs, max_tokens=max_tokens, temperature=temperature, key=key)
    return r.text.strip()


def last_json(text, kind):
    pat = r"\{.*\}" if kind == "obj" else r"\[.*\]"
    for cand in reversed(re.findall(pat, text or "", flags=re.S)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


async def write_all(setting, z, profile, qs, seen):
    brief, name = WD.SETTINGS[setting]["brief"], NAMES[setting]
    info = info_text(setting, z, profile)
    sys_ = {"role": "system", "content": SYSTEM.format(name=name)}
    tag = f"{setting}|{z}|{profile}"
    seen_qs = "\n".join(f"{i + 1}. {q.text}" for i, q in enumerate(q for q in qs if q.id in seen))
    honest_h, honest_g, decisive, lst, scen = await asyncio.gather(
        chat(HAIKU, [sys_, {"role": "user", "content": ESSAY.format(brief=brief, info=info, extra="")}], tag),
        chat(GEMINI, [sys_, {"role": "user", "content": ESSAY.format(brief=brief, info=info, extra="")}], tag,
             max_tokens=4000),
        chat(HAIKU, [sys_, {"role": "user", "content": ESSAY.format(brief=brief, info=info, extra=DECISIVE)}],
             tag),
        chat(HAIKU, [sys_, {"role": "user", "content": LIST.format(brief=brief, info=info, qs=seen_qs)}], tag),
        chat(HAIKU, [sys_, {"role": "user", "content": SCEN.format(brief=brief, info=info)}], tag))
    eloquent = await chat(C.STRONG, RESTYLE.format(essay=honest_h), tag)
    reports = {k: ([v], [1.0]) for k, v in [("honest_haiku", honest_h), ("honest_gemini", honest_g),
                                           ("eloquent", eloquent), ("overconfident", decisive), ("list", lst)]}
    hyps = []
    for h in (last_json(scen, "obj") or {}).get("hypotheses", []):
        try:
            hyps.append((str(h["text"]).strip(), max(0.0, float(h.get("weight", 1.0)))))
        except (KeyError, TypeError, ValueError):
            continue
    if hyps and sum(w for _, w in hyps) > 0:
        tot = sum(w for _, w in hyps)
        reports["scenarios"] = ([t for t, _ in hyps], [w / tot for _, w in hyps])
    return reports


async def judge(setting, essay):
    t = await chat(C.STRONG, JUDGE.format(name=NAMES[setting], brief=WD.SETTINGS[setting]["brief"], essay=essay),
                   "judge", max_tokens=800, temperature=0.0)
    m = re.findall(r"score\W{0,3}(\d+(?:\.\d+)?)", t, re.I)
    return min(10.0, max(1.0, float(m[-1]))) if m else None


async def parse(setting, text, reader_model):
    st = "\n".join(f"{i + 1}. {s}" for i, (_, s) in enumerate(WD.SETTINGS[setting]["factors"]))
    t = await chat(reader_model, PARSE.format(name=NAMES[setting], essay=text or "(no text)", statements=st),
                   "parse", max_tokens=200, temperature=0.0)
    v = last_json(t, "list")
    try:
        return [min(1.0, max(0.0, float(x))) for x in v][:3] if v and len(v) >= 3 else None
    except (TypeError, ValueError):
        return None


def truth_essay(setting, z):
    lines = [s if zk else f"It is NOT the case that: {s[0].lower()}{s[1:]}"
             for (_, s), zk in zip(WD.SETTINGS[setting]["factors"], z)]
    return "Established facts (certain):\n" + "\n".join(f"- {x}" for x in lines)


def outcomes_text(drawn, seen):
    return "\n".join(f"- {q.text} -> {'Yes' if q.answer == 0 else 'No'}" for q in drawn if q.id in seen)


async def run_world(setting, z, profiles, qs, seen, reader, rows, records):
    drawn = WD.draw(qs, z, seed=0)
    test_qs = [q for q in drawn if q.id not in seen]
    jobs = []  # (profile, condition, hyp index, weight, text, questions, mode)
    jobs.append((None, "truth", 0, 1.0, truth_essay(setting, z), drawn, "structured"))
    jobs.append((None, "holistic_empty", 0, 1.0, "OUTCOMES ALREADY OBSERVED THIS YEAR:\n"
                 + outcomes_text(drawn, seen), test_qs, "holistic"))
    written = await asyncio.gather(*[write_all(setting, z, p, qs, seen) for p in profiles])
    for p, reports in zip(profiles, written):
        for cond, (texts, ws) in reports.items():
            for hi, (t, w) in enumerate(zip(texts, ws)):
                jobs.append((p, cond, hi, w, t, drawn, "structured"))
        jobs.append((p, "holistic", 0, 1.0, reports["honest_haiku"][0][0] + "\n\nOUTCOMES ALREADY OBSERVED THIS "
                     "YEAR:\n" + outcomes_text(drawn, seen), test_qs, "holistic"))
    mats = await asyncio.gather(*[reader.matrix([t], q) for *_, t, q, _ in jobs])
    singles = [(p, c, t) for p, c, hi, w, t, q, m in jobs if m == "structured" and c not in ("scenarios", "truth")]
    scores = await asyncio.gather(*[judge(setting, t) for _, _, t in singles])
    judged = {(p, c): s for (p, c, _), s in zip(singles, scores)}
    parses = await asyncio.gather(*[parse(setting, t, reader.model_name) for *_, t, _, m in jobs
                                    if m == "structured"])
    parses = iter(parses)
    for (p, cond, hi, w, t, q_list, mode), P in zip(jobs, mats):
        par = next(parses) if mode == "structured" else None
        records.append({"setting": setting, "world": "".join(map(str, z)), "profile": p, "condition": cond,
                        "hyp": hi, "weight": w, "text": t, "judge": judged.get((p, cond)), "parse": par})
        for qq, pr in zip(q_list, P[0]):
            rows.append({"setting": setting, "world": "".join(map(str, z)), "profile": p, "condition": cond,
                         "hyp": hi, "weight": w, "q": qq.id, "seen": qq.id in seen, "p": float(pr),
                         "y": 1.0 if qq.answer == 0 else 0.0, "p_true_model": qq.p_yes(z)})


async def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    s0 = C.spend()
    reader = Reader(C.READER)
    rows, records, settings = [], [], {}
    for setting in WD.SETTINGS:
        qs = await WD.questions(setting, C.STRONG)
        seen = seen_split(qs, setting)
        settings[setting] = {"questions": [q.__dict__ for q in qs], "seen": sorted(seen)}
        empty = await reader.matrix([None], qs)
        for q, pr in zip(qs, empty[0]):
            rows.append({"setting": setting, "world": None, "profile": None, "condition": "empty", "hyp": 0,
                         "weight": 1.0, "q": q.id, "seen": q.id in seen, "p": float(pr), "y": None,
                         "p_true_model": None})
        ws = WD.worlds(setting)[:1] if args.pilot else WD.worlds(setting)
        for z in ws:
            await run_world(setting, z, list(WD.PROFILES), qs, seen, reader, rows, records)
            pd.DataFrame(rows).to_parquet(OUT / "cells.parquet")
            (OUT / "essays.json").write_text(json.dumps(records, indent=1))
            (OUT / "settings.json").write_text(json.dumps(settings, indent=1))
            print(f"{setting} {z}: spend so far ${C.spend() - s0:.2f}", flush=True)
    (OUT / "spend.json").write_text(json.dumps({"usd": C.spend() - s0}))
    print(f"done; OpenRouter spend ${C.spend() - s0:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    asyncio.run(main(ap.parse_args()))
