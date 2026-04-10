"""
server/app.py — FastAPI server exposing the OpenEnv HTTP API.

Endpoints:
  GET  /            → health check
  GET  /tasks       → list all tasks (alias: /state)
  POST /reset       → reset environment, return initial observation
  POST /grade/{id}  → step environment with agent action, return scored result
"""

import json
import logging
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .environment import ContentModEnv, _clamp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("content_mod_server")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Content Moderation RL Environment",
    description="OpenEnv-compatible moderation benchmark for RL agent evaluation.",
    version="1.1.0",
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def health():
    return {
        "status": "running",
        "benchmark": "content_mod_pro",
        "version": "1.1.0",
        "tasks": 3,
    }


@app.get("/tasks", tags=["tasks"])
@app.get("/state", tags=["tasks"])
def get_tasks():
    """Return task metadata for all tasks (OpenEnv /state endpoint)."""
    env = ContentModEnv()
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "grader": True,
            "description": f"Task {t['id']}: {t['name']} ({t['difficulty']})",
            "score_range": [0.01, 0.99],
        }
        for t in env.tasks
    ]


@app.post("/reset", tags=["env"])
async def reset():
    """Reset the environment and return the initial observation."""
    env = ContentModEnv()
    obs = env.reset()
    return obs.model_dump()


@app.post("/grade/{task_id}", tags=["env"])
async def grade(task_id: int, request: Request):
    """
    Step the environment with the given action and return a scored result.

    Accepts:
      { "action": { ... } }   — wrapped form
      { ... }                  — flat form (action fields at top level)
    """
    if task_id < 1 or task_id > 3:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Support both wrapped {"action": {...}} and flat {"is_spam": ...} forms
    action = body.get("action", body)
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except Exception:
            action = {}

    env = ContentModEnv()
    env.current_idx = task_id - 1   # 0-indexed

    try:
        obs, reward, done, info = env.step(action)
    except Exception as exc:
        logger.error(f"Step error for task {task_id}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=200,
            content={
                "task_id": task_id,
                "score": 0.50,  # neutral fallback, still in (0,1)
                "done": True,
                "error": str(exc)[:200],
            },
        )

    # Secondary hard clamp (belt-and-suspenders)
    final_score = _clamp(reward)

    logger.info(
        f"Task {task_id} ({env.tasks[task_id-1]['name']}) | "
        f"action={action} | raw_reward={reward:.4f} | final_score={final_score:.4f}"
    )

    return {
        "task_id": task_id,
        "score": final_score,
        "observation": obs.model_dump(),
        "done": bool(done),
        "info": info,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
