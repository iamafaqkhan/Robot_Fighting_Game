"""
Vercel Python entrypoint (root app.py).

Vercel's runtime detects this file automatically and routes /api/predict here.
Static assets (index.html, app.js) are served separately by the CDN.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rl.inference import MODEL_PATH, get_agent, predict_action

app = FastAPI(title="Robot Boxing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/predict")
async def health() -> dict:
    """Health / warm-up — loads model once per container."""
    get_agent()
    return {"status": "ok", "model": MODEL_PATH.name, "loaded": True}


@app.post("/api/predict")
async def predict(payload: dict) -> dict:
    """Greedy DQN inference from live game state JSON."""
    try:
        return {"action": predict_action(payload)}
    except Exception as exc:  # noqa: BLE001
        return {"action": "step_left", "error": str(exc)}
