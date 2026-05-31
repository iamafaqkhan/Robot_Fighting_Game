"""
Shared DQN inference utilities for local dev and Vercel serverless.

Model weights are loaded once per process (container) and reused across requests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rl.dqn import DQNAgent
from rl.robot_boxing_env import ACTION_NAMES, NUM_ACTIONS, RING_MAX, RING_MIN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "robot_boxer.pt"

ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}

# Global cache — survives warm serverless invocations
_agent: DQNAgent | None = None


def get_agent() -> DQNAgent:
    """Load PyTorch weights once per container lifecycle (CPU-only for serverless)."""
    global _agent
    if _agent is not None:
        return _agent

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Run: python train_boxer.py"
        )

    _agent = DQNAgent(device="cpu")
    _agent.load(str(MODEL_PATH), eval_mode=True)
    _agent.policy_net.eval()
    return _agent


def build_state_vector(payload: dict) -> np.ndarray:
    """Map frontend JSON game state to the 6-D normalized observation vector."""
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


def predict_action(payload: dict) -> str:
    """Run greedy DQN inference and return an action name string."""
    agent = get_agent()
    state = build_state_vector(payload)

    with torch.no_grad():
        tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_values = agent.policy_net(tensor)
        action_idx = int(q_values.argmax(dim=1).item())

    return ACTION_NAMES[action_idx]
