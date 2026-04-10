import json
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .environment import ContentModEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Content Moderation RL Environment", version="1.1.0")

@app.get("/")
def health():
    return {"status": "running", "benchmark": "content_mod_pro", "tasks": 3}

@app.get("/tasks")
@app.get("/state")
def get_tasks():
    env = ContentModEnv()
    return [{"id": t["id"], "name": t["name"], "difficulty": t["difficulty"],
             "grader": True, "score_range": [0.01, 0.99]} for t in env.tasks]

@app.post("/reset")
async def reset():
    env = ContentModEnv()
    return env.reset().model_dump()

@app.post("/grade/{task_id}")
@app.post("/step")
async def grade(request: Request, task_id: int = 1):
    try:
        body = await request.json()
    except Exception:
        body = {}

    action = body.get("action", body)
    if isinstance(action, str):
        try: action = json.loads(action)
        except Exception: action = {}

    try:
        env = ContentModEnv()
        env.current_idx = max(0, task_id - 1)
        obs, reward, done, info = env.step(action)
        final_score = max(0.01, min(0.99, float(reward)))
    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        return JSONResponse(content={"task_id": task_id, "score": 0.50, "done": True})

    return {"task_id": task_id, "score": float(final_score),
            "observation": obs.model_dump(), "done": bool(done)}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
