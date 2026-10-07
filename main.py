import json
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .db import get_db, init_db
from .evaluator import score
from .llm import call_llm
from .techniques import TECHNIQUES, build_prompt, extract_answer

app = FastAPI(title="PromptLab", description="Prompt engineering & LLM evaluation platform")
init_db()

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")


class PromptIn(BaseModel):
    name: str
    technique: str
    instruction: str
    role: str = "an expert assistant"
    examples: List[Dict[str, str]] = Field(default_factory=list)


class TestCase(BaseModel):
    input: str
    expected: str


class ExperimentIn(BaseModel):
    prompt_ids: List[int]
    test_cases: List[TestCase]
    model: str = "claude-haiku-4-5"
    method: str = "contains"  # contains | exact | similarity | json_valid


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health():
    from .llm import _provider
    return {"status": "ok", "provider": _provider(), "techniques": TECHNIQUES}


@app.post("/api/prompts")
def create_prompt(p: PromptIn):
    if p.technique not in TECHNIQUES:
        raise HTTPException(400, f"technique must be one of {TECHNIQUES}")
    if p.technique == "few_shot" and not p.examples:
        raise HTTPException(400, "few_shot requires examples")
    with get_db() as db:
        row = db.execute("SELECT MAX(version) v FROM prompts WHERE name=?", (p.name,)).fetchone()
        version = (row["v"] or 0) + 1
        cur = db.execute(
            "INSERT INTO prompts(name,version,technique,instruction,role,examples) VALUES(?,?,?,?,?,?)",
            (p.name, version, p.technique, p.instruction, p.role, json.dumps(p.examples)),
        )
        return {"id": cur.lastrowid, "name": p.name, "version": version}


@app.get("/api/prompts")
def list_prompts():
    with get_db() as db:
        rows = db.execute("SELECT * FROM prompts ORDER BY name, version").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/experiments")
def run_experiment(e: ExperimentIn):
    with get_db() as db:
        prompts = {}
        for pid in e.prompt_ids:
            row = db.execute("SELECT * FROM prompts WHERE id=?", (pid,)).fetchone()
            if not row:
                raise HTTPException(404, f"prompt {pid} not found")
            prompts[pid] = row
        exp_id = db.execute("INSERT INTO experiments(model,method) VALUES(?,?)", (e.model, e.method)).lastrowid

        for pid, p in prompts.items():
            for tc in e.test_cases:
                final = build_prompt(p["technique"], p["instruction"], tc.input,
                                     json.loads(p["examples"]), p["role"])
                res = call_llm(final, e.model)
                answer = extract_answer(p["technique"], res["text"])
                s = 0.0 if res["error"] else score(answer, tc.expected, e.method)
                db.execute(
                    """INSERT INTO runs(experiment_id,prompt_id,test_input,expected,final_prompt,output,
                       answer,score,input_tokens,output_tokens,latency_ms,cost_usd,error)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (exp_id, pid, tc.input, tc.expected, final, res["text"], answer, s,
                     res["input_tokens"], res["output_tokens"], res["latency_ms"], res["cost_usd"], res["error"]),
                )
    return get_experiment(exp_id)


@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: int):
    with get_db() as db:
        exp = db.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
        if not exp:
            raise HTTPException(404, "experiment not found")
        summary = db.execute(
            """SELECT p.id prompt_id, p.name, p.version, p.technique,
                      COUNT(*) runs, ROUND(AVG(r.score),3) avg_score,
                      ROUND(AVG(r.latency_ms),1) avg_latency_ms,
                      SUM(r.input_tokens+r.output_tokens) total_tokens,
                      ROUND(SUM(r.cost_usd),6) total_cost_usd,
                      SUM(r.error IS NOT NULL) errors
               FROM runs r JOIN prompts p ON p.id=r.prompt_id
               WHERE r.experiment_id=? GROUP BY p.id ORDER BY avg_score DESC, total_tokens ASC""",
            (exp_id,),
        ).fetchall()
        runs = db.execute("SELECT * FROM runs WHERE experiment_id=?", (exp_id,)).fetchall()
    summary = [dict(s) for s in summary]
    return {
        "experiment": dict(exp),
        "summary": summary,
        "best_prompt": summary[0] if summary else None,
        "runs": [dict(r) for r in runs],
    }


@app.get("/api/experiments")
def list_experiments():
    with get_db() as db:
        rows = db.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]
