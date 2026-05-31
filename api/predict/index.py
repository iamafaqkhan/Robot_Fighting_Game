"""
Vercel serverless function — /api/predict

FastAPI `app` entrypoint (required filename: api/predict/index.py).
POST JSON game state → { "action": "jab" }.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root: api/predict/index.py → api/predict → api → root
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from rl.inference import MODEL_PATH, get_agent, predict_action  # noqa: E402

app = FastAPI(title="Robot Boxing Predict")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health() -> dict:
    """Warm-up / health check (also used by the frontend on load)."""
    get_agent()
    return {"status": "ok", "model": MODEL_PATH.name, "loaded": True}


@app.post("/")
async def predict(payload: dict) -> dict:
    """Greedy DQN inference from live canvas state."""
    try:
        action = predict_action(payload)
        return {"action": action}
    except Exception as exc:  # noqa: BLE001
        return {"action": "step_left", "error": str(exc)}
