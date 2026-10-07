"""Scoring of model outputs against expected answers (0.0 - 1.0)."""
import difflib
import json
import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def score(output: str, expected: str, method: str = "contains") -> float:
    out, exp = _norm(output), _norm(expected)
    if not exp:
        return 0.0
    if method == "exact":
        return 1.0 if out == exp else 0.0
    if method == "similarity":
        return round(difflib.SequenceMatcher(None, out, exp).ratio(), 3)
    if method == "json_valid":
        try:
            json.loads(output)
            return 1.0
        except (json.JSONDecodeError, TypeError):
            return 0.0
    # default: contains
    return 1.0 if exp in out else 0.0
