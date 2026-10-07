"""Prompt engineering techniques.

Each technique wraps a user's task/instruction around a test input so the same
task can be evaluated under different prompting strategies.
"""
from typing import Dict, List, Optional

TECHNIQUES = ["zero_shot", "few_shot", "chain_of_thought", "role_based", "structured_json"]


def build_prompt(
    technique: str,
    instruction: str,
    user_input: str,
    examples: Optional[List[Dict[str, str]]] = None,
    role: str = "an expert assistant",
) -> str:
    """Build the final prompt string for a given technique."""
    examples = examples or []

    if technique == "zero_shot":
        return f"{instruction}\n\nInput: {user_input}\nAnswer:"

    if technique == "few_shot":
        shots = "\n\n".join(f"Input: {e['input']}\nAnswer: {e['output']}" for e in examples)
        return f"{instruction}\n\nExamples:\n{shots}\n\nInput: {user_input}\nAnswer:"

    if technique == "chain_of_thought":
        return (
            f"{instruction}\n\nInput: {user_input}\n\n"
            "Think through the problem step by step, then give the final answer "
            "on a new line starting with 'Answer:'."
        )

    if technique == "role_based":
        return (
            f"You are {role}. Follow the instruction precisely and be concise.\n\n"
            f"{instruction}\n\nInput: {user_input}\nAnswer:"
        )

    if technique == "structured_json":
        return (
            f"{instruction}\n\nInput: {user_input}\n\n"
            'Respond ONLY with valid JSON in the form {"answer": "<your answer>"}.'
        )

    raise ValueError(f"Unknown technique: {technique}")


def extract_answer(technique: str, text: str) -> str:
    """Pull the final answer out of a raw model response."""
    import json
    import re

    text = (text or "").strip()
    if technique == "structured_json":
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return str(json.loads(match.group(0)).get("answer", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                return text
    if technique == "chain_of_thought":
        match = re.findall(r"Answer:\s*(.+)", text, re.I)
        if match:
            return match[-1].strip()
    return text
