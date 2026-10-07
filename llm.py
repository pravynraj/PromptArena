"""LLM provider layer.

Providers (auto-selected, override with LLM_PROVIDER):
  - anthropic : needs ANTHROPIC_API_KEY
  - ollama    : local models, needs a running Ollama server
  - mock      : offline fallback so the whole pipeline works without a key
"""
import os
import re
import time
from typing import Any, Dict

import httpx

# USD per 1M tokens (input, output). Used for cost estimates only.
PRICING = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (3.00, 15.00),
}


def _provider() -> str:
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    return round((in_tok * in_price + out_tok * out_price) / 1_000_000, 6)


def call_llm(prompt: str, model: str = "claude-haiku-4-5", max_tokens: int = 512) -> Dict[str, Any]:
    provider = _provider()
    start = time.perf_counter()
    error = None
    text = ""
    in_tok = out_tok = 0

    try:
        if provider == "anthropic":
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            text = "".join(b.get("text", "") for b in data["content"])
            in_tok = data["usage"]["input_tokens"]
            out_tok = data["usage"]["output_tokens"]

        elif provider == "ollama":
            r = httpx.post(
                os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "")
            in_tok = data.get("prompt_eval_count", _estimate_tokens(prompt))
            out_tok = data.get("eval_count", _estimate_tokens(text))

        else:  # mock
            text = _mock_response(prompt)
            in_tok, out_tok = _estimate_tokens(prompt), _estimate_tokens(text)
            time.sleep(0.05)
    except Exception as exc:  # noqa: BLE001 - surface any provider failure in the run record
        error = str(exc)[:300]

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return {
        "text": text,
        "provider": provider,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_ms": latency_ms,
        "cost_usd": _cost(model, in_tok, out_tok),
        "error": error,
    }


def _mock_response(prompt: str) -> str:
    """Tiny rule-based stand-in so demos run offline. Not a real model."""
    lower = prompt.lower()
    wants_json = "valid json" in lower
    if "positive or negative" in lower or "sentiment" in lower:
        last = lower.split("input:")[-1]
        neg = any(w in last for w in ["bad", "terrible", "hate", "worst", "slow", "broken"])
        label = "negative" if neg else "positive"
    else:
        m = re.findall(r"input:\s*(.+)", prompt, re.I)
        label = (m[-1].strip() if m else "ok")[:60]
    if wants_json:
        return '{"answer": "%s"}' % label
    if "step by step" in lower:
        return f"Step 1: read the input.\nStep 2: decide.\nAnswer: {label}"
    return label
