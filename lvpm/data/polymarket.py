"""Fetch resolved Polymarket binary markets and their price histories (public APIs, cached on disk)."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
import urllib.parse
import urllib.request
from pathlib import Path

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
ISO = "%Y-%m-%dT%H:%M:%SZ"
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"


def _get(url: str, retries: int = 6):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            data = json.load(urllib.request.urlopen(req, timeout=60))
            f.write_text(json.dumps(data))
            return data
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 ** k)


def closed_markets(end_min: str, end_max: str, start_max: str, volume_min: float = 0,
                   window_days: int = 3) -> list[dict]:
    """All closed markets with end date in [end_min, end_max] that opened before start_max.

    The API caps offsets (~2700), so we page within short end-date windows."""
    lo, hi = (datetime.fromisoformat(d.replace("Z", "+00:00")) for d in (end_min, end_max))
    out = {}
    while lo < hi:
        nxt = min(lo + timedelta(days=window_days), hi)
        off = 0
        while True:
            q = urllib.parse.urlencode(dict(closed="true", limit=100, offset=off,
                                            end_date_min=lo.strftime(ISO), end_date_max=nxt.strftime(ISO),
                                            start_date_max=start_max, volume_num_min=volume_min), safe=":")
            batch = _get(f"{GAMMA}/markets?{q}")
            out.update((m["id"], m) for m in batch)
            off += len(batch)
            if len(batch) < 100:
                break
        lo = nxt
    return list(out.values())


def price_history(token_id: str, fidelity_min: int = 720) -> list[tuple[int, float]]:
    """[(unix_ts, price)] over the market's whole life (resolved markets need fidelity >= 720 min)."""
    q = urllib.parse.urlencode(dict(market=token_id, interval="max", fidelity=fidelity_min))
    return [(h["t"], h["p"]) for h in _get(f"{CLOB}/prices-history?{q}").get("history", [])]


def yes_outcome(m: dict) -> int | None:
    """1/0 if the market resolved cleanly to its first outcome / second outcome, else None."""
    if m.get("umaResolutionStatus") not in (None, "resolved"):
        return None
    prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
    if len(prices) != 2 or sorted(prices) != [0.0, 1.0]:
        return None
    return int(prices[0] == 1.0)
