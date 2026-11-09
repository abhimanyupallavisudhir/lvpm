"""Model access for the essay experiments: OpenRouter keys, prices, cache, real spend.

Import this before creating any client. It
- maps OPENROUTER_API_KEY to the project's OpenRouter key if unset;
- puts the LLM cache outside the repo (CVT_CACHE), so reruns are free;
- adds OpenRouter prices to the testbed's table, so per-run cost estimates are right;
- reads the key's real usage from OpenRouter (``spend()``), the ground truth for costs.

OpenAI models are refused on these keys (an account-level block), so the
experiments use Anthropic and Google models through OpenRouter.
"""

import json
import os
import urllib.request
from pathlib import Path

os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY_ASTRA", ""))
os.environ.setdefault("CVT_CACHE", str(Path(__file__).resolve().parents[3] / ".cvt-cache"))

from cvtestbed.llm import prices  # noqa: E402

READER = "openrouter/anthropic/claude-haiku-4.5"
WRITERS = ["openrouter/anthropic/claude-haiku-4.5", "openrouter/google/gemini-2.5-flash"]
STRONG = "openrouter/anthropic/claude-sonnet-4.5"  # paraphraser and judge

prices().update({  # USD per million tokens (input, output), OpenRouter list prices
    "openrouter/anthropic/claude-haiku-4.5": (1.00, 5.00),
    "openrouter/anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "openrouter/google/gemini-2.5-flash": (0.30, 2.50),
})


def spend():
    """Cumulative USD used by the OpenRouter key, as reported by OpenRouter."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return float(json.load(r)["data"]["usage"])
