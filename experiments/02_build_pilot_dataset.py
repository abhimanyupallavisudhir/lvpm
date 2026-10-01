"""Build the pilot dataset: topical clusters of resolved Polymarket binary questions.

Forecast date t0 = 2026-07-01 (after the forecaster LLM's June-2026 knowledge cutoff).
A question is eligible if it opened by 2026-06-20 (>= 10 days of pre-t0 price history),
resolved cleanly YES/NO between 2026-07-04 and 2026-10-01 with a stated deadline
after 2026-07-05, is not part of a
mutually-exclusive (negRisk) set, and traded at 2%-98% at t0.  One question per event
(the one priced closest to 50% at t0), so date/threshold ladders don't make the
structure trivially logical.  Clusters are assigned by keyword rules on the event title
alone (never on outcomes), capped at 12 questions by volume.

Writes data/pilot_clusters.json."""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvpm.covariation import DAY, daily_grid  # noqa: E402
from lvpm.data import polymarket as pm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T0 = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
PRE_DAYS = 30

CLUSTERS = {  # first match wins
    "russia_ukraine_battlefield": r"^Will Russia (capture|enter)|Ukraine re-enter",
    "russia_ukraine_diplomacy": r"Russia x Ukraine|Ukraine signs|Putin|Zelensk|Ukraine election|cede territory|Ukraine agrees",
    "iran_israel": r"Iran|Hormuz|Kharg|Israel|Hezbollah|Lebanon|Khamenei|Netanyahu|enrich|Yeshiva",
    "ai_and_tech": r"Claude|GPT|Gemini|OpenAI|Anthropic|Mythos|Grok|SpaceX|Cursor|Cloudflare|Databricks",
    "world_cup": r"World Cup|WC Champions|Penalty Shootout|Messi|Ronaldo",
    "us_politics": r"Trump|SCOTUS|House pass|Hegseth|Comey|Noem|Hunter Biden|DNC|anti-cartel|Cuba|SBF|primary|Primary",
    "world_politics": r"Starmer|Netherlands|AfD|Romanian|Peru|Lula|Tid|Italy Senate|Haiti|NATO Summit|Tapia",
    "bank_q2_earnings": r"Goldman|JP Morgan|Citigroup|Morgan Stanley|Bank of America|Johnson & Johnson",
}


MONTHS = "January February March April May June July August September October November December".split()


def stated_deadline(text: str):
    """Last 'Month D' date mentioned in the question (year 2026 assumed), or None."""
    hits = re.findall(r"(%s) (\d{1,2})" % "|".join(MONTHS), text)
    if not hits:
        return None
    mo, d = hits[-1]
    return int(datetime(2026, MONTHS.index(mo) + 1, int(d), tzinfo=timezone.utc).timestamp())


def main():
    ms = pm.closed_markets("2026-07-04T00:00:00Z", "2026-10-01T00:00:00Z", "2026-06-20T00:00:00Z", 5000)
    ms = [m for m in ms if not m.get("negRisk") and pm.yes_outcome(m) is not None
          and json.loads(m["outcomes"]) == ["Yes", "No"]]
    by_event = {}
    for m in ms:
        ev = (m.get("events") or [{}])[0]
        title = ev.get("title", m["question"]).strip()
        cl = next((c for c, rx in CLUSTERS.items() if re.search(rx, title)), None)
        if cl is None:
            continue
        hist = pm.price_history(json.loads(m["clobTokenIds"])[0])
        if not hist:
            continue
        grid = daily_grid([hist], T0 - PRE_DAYS * DAY, T0)[:, 0]
        p0 = grid[-1]
        if np.isnan(p0) or not 0.02 <= p0 <= 0.98 or np.isnan(grid[-11]):
            continue
        deadline = stated_deadline(m["question"])
        if deadline is not None and deadline < T0 + 5 * DAY:  # already (nearly) decided at t0
            continue
        q = dict(id=m["id"], question=m["question"].strip(), event=title, cluster=cl,
                 description=(m.get("description") or "")[:900], end_date=m.get("endDate"),
                 closed_time=m.get("closedTime"), volume=m.get("volumeNum"), p_t0=float(p0),
                 outcome=pm.yes_outcome(m), history=hist)
        by_event.setdefault(title, []).append(q)
    clusters = {}
    for title, qs in by_event.items():
        q = min(qs, key=lambda q: abs(q["p_t0"] - 0.5))
        clusters.setdefault(q["cluster"], []).append(q)
    out = {}
    for c, qs in clusters.items():
        qs = sorted(qs, key=lambda q: -q["volume"])[:12]
        if len(qs) >= 4:
            out[c] = qs
    meta = dict(t0=T0, t0_iso="2026-07-01T00:00:00Z", pre_days=PRE_DAYS, source="Polymarket gamma+clob APIs",
                built=datetime.now(timezone.utc).isoformat())
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "pilot_clusters.json").write_text(json.dumps(dict(meta=meta, clusters=out), indent=1))
    for c, qs in out.items():
        print(f"\n## {c} ({len(qs)})")
        for q in qs:
            print(f"  {q['p_t0']:.2f} -> {q['outcome']}  {q['question'][:90]}")


if __name__ == "__main__":
    main()
