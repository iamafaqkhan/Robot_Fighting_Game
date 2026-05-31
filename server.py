"""
Layer 2 — Real-time inference server (FastAPI + Socket.IO).

Loads pre-trained DQN weights on startup and answers `game_tick` events from the
browser with greedy `ai_response` actions.

Run:
    uvicorn server:socket_app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import socketio
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from rl.dqn import DQNAgent, STATE_DIM
from rl.robot_boxing_env import ACTION_NAMES, NUM_ACTIONS, RING_MAX, RING_MIN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_boxing")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "robot_boxer.pt"

ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}

# --- Socket.IO (ASGI) ---
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

fastapi_app = FastAPI(title="Robot Boxing RL Server", version="1.0.0")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent: DQNAgent | None = None


@fastapi_app.on_event("startup")
async def load_model() -> None:
    """Load DQN weights into evaluation mode when the server starts."""
    global agent
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Run: python train_boxer.py"
        )

    agent = DQNAgent()
    agent.load(str(MODEL_PATH), eval_mode=True)
    agent.policy_net.eval()
    logger.info("Loaded model from %s (device=%s)", MODEL_PATH, agent.device)


def build_state_vector(payload: dict) -> np.ndarray:
    """
    Convert frontend game_tick payload into the 6-D normalized observation
    used during Gymnasium training.
    """
    player_x = float(payload.get("player_x", 0.25))
    ai_x = float(payload.get("ai_x", 0.75))
    player_health = float(payload.get("player_health", 100.0))
    ai_health = float(payload.get("ai_health", 100.0))
    player_stamina = float(payload.get("player_stamina", 100.0))
    ai_stamina = float(payload.get("ai_stamina", 100.0))

    raw_action = payload.get("player_action", "step_left")
    if isinstance(raw_action, str):
        player_action_idx = ACTION_TO_INDEX.get(raw_action, 0)
    else:
        player_action_idx = int(np.clip(int(raw_action), 0, NUM_ACTIONS - 1))

    span = RING_MAX - RING_MIN
    distance = abs(player_x - ai_x) / span if span > 0 else 0.0

    return np.array(
        [
            float(np.clip(distance, 0.0, 1.0)),
            float(np.clip(player_health / 100.0, 0.0, 1.0)),
            float(np.clip(ai_health / 100.0, 0.0, 1.0)),
            float(np.clip(player_stamina / 100.0, 0.0, 1.0)),
            float(np.clip(ai_stamina / 100.0, 0.0, 1.0)),
            float(player_action_idx / (NUM_ACTIONS - 1)),
        ],
        dtype=np.float32,
    )


def infer_action(state: np.ndarray) -> str:
    assert agent is not None
    with torch.no_grad():
        tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(
            0
        )
        q_values = agent.policy_net(tensor)
        action_idx = int(q_values.argmax(dim=1).item())
    return ACTION_NAMES[action_idx]


@fastapi_app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html")


@fastapi_app.get("/app.js")
async def serve_app_js() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "app.js")


@fastapi_app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": agent is not None and MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
    }


@sio.event
async def connect(sid: str, environ: dict) -> None:
    logger.info("Client connected: %s", sid)
    await sio.emit("server_ready", {"message": "Robot Boxing AI online"}, room=sid)


@sio.event
async def disconnect(sid: str) -> None:
    logger.info("Client disconnected: %s", sid)


@sio.event
async def game_tick(sid: str, data: dict) -> None:
    """
    Receive live UI state, run DQN inference, return optimal action string.
    """
    if agent is None:
        await sio.emit("ai_response", {"action": "block", "error": "model_not_loaded"}, room=sid)
        return

    try:
        state = build_state_vector(data or {})
        action = infer_action(state)
        await sio.emit("ai_response", {"action": action}, room=sid)
    except Exception as exc:  # noqa: BLE001 — keep socket alive
        logger.exception("Inference failed for %s", sid)
        await sio.emit(
            "ai_response",
            {"action": "step_left", "error": str(exc)},
            room=sid,
        )


# ASGI app mounted by uvicorn: server:socket_app
socket_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)
