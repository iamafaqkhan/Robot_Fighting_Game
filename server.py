"""
Local development server (optional).

Mirrors the Vercel POST /api/predict endpoint for testing before deploy.
Production uses Vercel serverless — run: vercel dev  OR  python server.py

    python server.py
    # open http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rl.inference import MODEL_PATH, get_agent, predict_action

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Robot Boxing (local dev)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GameState(BaseModel):
    player_x: float = 0.25
    ai_x: float = 0.75
    player_health: float = 100.0
    ai_health: float = 100.0
    player_stamina: float = 100.0
    ai_stamina: float = 100.0
    player_action: str = "step_left"


@app.on_event("startup")
def warmup_model() -> None:
    get_agent()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/app.js")
async def app_js() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "app.js")


@app.get("/api/predict")
async def health() -> dict:
    get_agent()
    return {"status": "ok", "model": MODEL_PATH.name, "loaded": True}


@app.post("/api/predict")
async def predict(state: GameState) -> dict:
    action = predict_action(state.model_dump())
    return {"action": action}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
