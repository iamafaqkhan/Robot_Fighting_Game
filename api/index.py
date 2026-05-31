"""
Vercel serverless entrypoint — api/index.py

Vercel only auto-detects files named index.py, app.py, etc. inside /api.
This function is served at /api; vercel.json rewrites /api/predict → here.

Export: app = FastAPI()
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
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
    """GET /api/predict (via rewrite) — warm-up and health check."""
    get_agent()
    return {"status": "ok", "model": MODEL_PATH.name, "loaded": True}


@app.post("/")
async def predict(payload: dict) -> dict:
    """POST /api/predict (via rewrite) — DQN inference."""
    try:
        return {"action": predict_action(payload)}
    except Exception as exc:  # noqa: BLE001
        return {"action": "step_left", "error": str(exc)}
