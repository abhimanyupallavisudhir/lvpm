"""E2 data: dated LessWrong essays (Sep 2025 - Feb 2026) and resolved Manifold questions.

The reader (claude-haiku-4.5) has a training cutoff of about July 2025, so it has seen neither these essays nor
the outcomes. Questions are drawn per essay from markets created *after* the essay was posted.

1. data/lw_posts_2025-09_2026-02.json and data/manifold_pool.json are fetched first (see the notes in
   notes/latent-essays-results.md for the exact queries);
2. this script downloads the candidate essays' text, classifies each by topic and by how much it bears on
   real-world events in the following year (a cheap model), and writes data/e2_essays.json.

    PYTHONPATH=. python experiments/essays_history_data.py            # essays
    PYTHONPATH=. python experiments/essays_history_data.py questions  # question topics
"""

import asyncio
import collections
import html
import json
import re
import urllib.request
from pathlib import Path

from lvpm.essays import llmconf as C  # noqa: F401

from cvtestbed.llm import default_llm

DATA = Path(__file__).resolve().parents[1] / "data"
KEEP_TAGS = {"AI Governance", "AI Timelines", "Economics", "Politics", "Forecasting & Prediction", "AI Takeoff",
             "Newsletters", "China", "Anthropic (org)", "AI Evaluations", "OpenAI", "AI Risk",
             "Language Models (LLMs)", "World Optimization"}
TOPICS = ["ai", "us_politics", "geopolitics", "economy"]
CLASSIFY = ("Here is the start of an essay posted on {date}.\n\nTITLE: {title}\n\n{text}\n\nAnswer two questions. "
            "(1) Which topic does it mainly concern: ai (AI capabilities, AI companies, AI policy), us_politics, "
            "geopolitics, economy, or none? (2) How much does it make claims that bear on real-world events in "
            "the following twelve months - developments, trends, likely outcomes - as opposed to theory, "
            "personal advice or fiction? 0 = not at all, 3 = centrally. Reply with only JSON: "
            "{{\"topic\": \"...\", \"forward\": n}}.")
MAX_WORDS = 3000


def gql(query):
    req = urllib.request.Request("https://www.lesswrong.com/graphql", data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=120))["data"]


def body_text(post_id):
    d = gql('{ post(input:{selector:{_id:"%s"}}){ result { contents { html } } } }' % post_id)
    h = ((d["post"]["result"] or {}).get("contents") or {}).get("html") or ""
    t = re.sub(r"<(br|/p|/li|/h\d)>", "\n", h)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def truncate(text, n=MAX_WORDS):
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " [...]"


async def main():
    posts = json.loads((DATA / "lw_posts_2025-09_2026-02.json").read_text())
    cands = [p for p in posts if KEEP_TAGS & {t["name"] for t in (p.get("tags") or [])} and p.get("user")]
    for p in cands:
        p["text"] = body_text(p["_id"])
    llm = default_llm()
    rs = await asyncio.gather(*[llm.chat(C.WRITERS[1], CLASSIFY.format(
        date=p["postedAt"][:10], title=p["title"], text=" ".join(p["text"].split()[:700])), max_tokens=2000,
        temperature=0.0, key="classify") for p in cands])
    out = []
    for p, r in zip(cands, rs):
        m = re.findall(r"\{.*?\}", r.text, flags=re.S)
        try:
            c = json.loads(m[-1])
        except (IndexError, json.JSONDecodeError):
            continue
        if c.get("topic") in TOPICS and int(c.get("forward", 0)) >= 2 and len(p["text"].split()) >= 300:
            out.append({"id": p["_id"], "title": p["title"], "date": p["postedAt"][:10],
                        "author": p["user"]["displayName"], "karma": p["baseScore"], "words": p["wordCount"],
                        "topic": c["topic"], "forward": int(c["forward"]), "text": truncate(p["text"])})
    (DATA / "e2_essays.json").write_text(json.dumps(out, indent=1))
    print(len(cands), "candidates ->", len(out), "essays")
    print(collections.Counter(e["topic"] for e in out))
    print(collections.Counter(e["author"] for e in out).most_common(10))


QCLASS = ("Classify each forecasting question by topic: ai (AI models, AI companies, AI policy, AI capabilities), "
          "us_politics, geopolitics (international affairs, wars, other countries' politics), economy (macro, "
          "markets, crypto, companies outside AI), or other (sports, entertainment, personal, Manifold-meta, "
          "jokes, anything else). Reply with only a JSON list of topics, one per question, in order.\n\n{qs}")


async def classify_questions():
    """Topic of each Manifold question by a cheap model (search terms alone are too noisy)."""
    pool = json.loads((DATA / "manifold_pool.json").read_text())
    llm = default_llm()
    batches = [pool[i:i + 50] for i in range(0, len(pool), 50)]
    rs = await asyncio.gather(*[llm.chat(C.WRITERS[1], QCLASS.format(qs="\n".join(
        f"{j + 1}. {q['question']}" for j, q in enumerate(b))), max_tokens=4000, temperature=0.0, key="qclass")
        for b in batches])
    for b, r in zip(batches, rs):
        m = re.findall(r"\[.*\]", r.text, flags=re.S)
        labels = json.loads(m[-1]) if m else []
        for q, t in zip(b, labels + [None] * len(b)):
            q["topic"] = t if t in TOPICS + ["other"] else None
    (DATA / "manifold_pool.json").write_text(json.dumps(pool))
    print(collections.Counter(q.get("topic") for q in pool))


if __name__ == "__main__":
    import sys
    asyncio.run(classify_questions() if sys.argv[1:] == ["questions"] else main())
