import uvicorn
from fastapi import FastAPI, Request
from .environment import ContentModEnv

app = FastAPI()
env = ContentModEnv()

@app.get("/")
def health():
    return {"status": "Running", "benchmark": "content_mod_pro"}

@app.post("/reset")
async def reset():
    obs = env.reset()
    return obs.dict() if hasattr(obs, 'dict') else obs

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    # Support both wrapped and unwrapped action formats
    action = data.get("action", data)
    
    obs, reward, done, info = env.step(action)
    
    return {
        "observation": obs.dict() if hasattr(obs, 'dict') else obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()