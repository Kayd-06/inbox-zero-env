# app.py
# ============================================================
# InboxZeroEnv — OpenEnv REST API
# FastAPI wrapper for automated hackathon evaluation.
# ============================================================

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from env.email_env import InboxZeroEnv
from env.models import Action, Observation, Reward

app = FastAPI(title="InboxZeroEnv — v3.0.0", version="3.0.0")

# Global environment instance
# In a production RL setting, this might be per-session, 
# but for the OpenEnv single-agent evaluation, a simple global works.
_env: Optional[InboxZeroEnv] = None

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepResponse(BaseModel):
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "name": "InboxZeroEnv",
        "version": "3.0.0",
        "status": "ready",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/reset", response_model=Observation)
async def reset(request: Optional[ResetRequest] = Body(default=None)):
    global _env
    # Handle the case where 'request' is None (empty POST body)
    task_id = request.task_id if (request and request.task_id) else "easy"
    try:
        # Re-initialize the environment for the requested task
        _env = InboxZeroEnv(task_name=task_id)
        obs = _env.reset()
        print(f"--- START TASK: {task_id.upper()} ---")
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step(action: Action = Body(...)):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        obs, reward, done, info = _env.step(action)
        
        if done:
            print(f"--- END TASK: {_env.task.name.upper()} ---")
            
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return _env.state()

@app.get("/tasks")
async def tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "default": "easy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
