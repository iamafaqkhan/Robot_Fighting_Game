"""
Vercel serverless entrypoint — api/index.py

Routes accept both "/" and "/api/predict" because Vercel may pass either path.
Uses NumPy inference (robot_boxer.npz) — no PyTorch cold start.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from rl.inference import NPZ_PATH, get_agent, predict_action  # noqa: E402

app = FastAPI(title="Robot Boxing Predict")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _health() -> dict:
    get_agent()
    return {"status": "ok", "model": NPZ_PATH.name, "loaded": True}


async def _predict(payload: dict) -> dict:
    try:
        return {"action": predict_action(payload)}
    except Exception as exc:  # noqa: BLE001
        return {"action": "step_left", "error": str(exc)}


# Register both paths — Vercel rewrite may deliver /api/predict or /
for _path in ("/", "/api/predict"):
    app.add_api_route(_path, _health, methods=["GET"])
    app.add_api_route(_path, _predict, methods=["POST"])
