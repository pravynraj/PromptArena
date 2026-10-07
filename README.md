# PromptLab — Prompt Engineering & LLM Evaluation Platform

Version prompts, run them against an LLM with different prompt engineering techniques, and compare
quality, latency, tokens, and cost to find the best prompt for a task.

## Features
- **Prompt versioning** — every save of a named prompt creates a new version (v1, v2, …).
- **5 prompt engineering techniques** — zero-shot, few-shot, chain-of-thought, role-based, structured JSON output.
- **Experiments** — run selected prompt versions over a test set in one click.
- **Automatic scoring** — contains / exact / similarity / JSON-validity.
- **Metrics per prompt** — average score, latency, token usage, estimated cost, error count; best prompt is highlighted.
- **Pluggable LLM layer** — Anthropic API, local Ollama models, or an offline mock so it runs with no key.

## Stack
FastAPI · SQLite · httpx · vanilla JS dashboard · Docker

## Run locally
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000
```

To use a real model, copy `.env.example` to `.env`, set `ANTHROPIC_API_KEY`, and export it
(or run via `docker compose up --build`). Without a key it uses the offline mock provider,
which is only meant for testing the pipeline.

## API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/prompts` | Create a prompt version |
| GET | `/api/prompts` | List prompts |
| POST | `/api/experiments` | Run prompts over test cases and score them |
| GET | `/api/experiments/{id}` | Per-prompt summary + every run |

## Example
```json
POST /api/experiments
{
  "prompt_ids": [1, 2, 3],
  "test_cases": [{"input": "Great product", "expected": "positive"}],
  "model": "claude-haiku-4-5",
  "method": "contains"
}
```

## Structure
```
backend/app/   techniques.py  llm.py  evaluator.py  db.py  main.py
frontend/      index.html
```
