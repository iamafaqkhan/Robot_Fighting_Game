"""
Shared DQN inference for local dev and Vercel serverless.

Uses lightweight NumPy forward pass (robot_boxer.npz) on Vercel — no PyTorch
import at cold start. Falls back to PyTorch (.pt) for local training workflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl.robot_boxing_env import ACTION_NAMES, NUM_ACTIONS, RING_MAX, RING_MIN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = PROJECT_ROOT / "robot_boxer.npz"
MODEL_PATH = PROJECT_ROOT / "robot_boxer.pt"

ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}

_weights: np.lib.npyio.NpzFile | None = None
_torch_agent = None


def export_numpy_weights(pt_path: Path | None = None, npz_path: Path | None = None) -> Path:
    """Export PyTorch policy weights to compressed NumPy archive for serverless."""
    import torch

    from rl.dqn import DQNAgent

    pt_path = pt_path or MODEL_PATH
    npz_path = npz_path or NPZ_PATH

    agent = DQNAgent(device="cpu")
    agent.load(str(pt_path), eval_mode=True)
    arrays = {k: v.cpu().numpy() for k, v in agent.policy_net.state_dict().items()}
    np.savez_compressed(str(npz_path), **arrays)
    return npz_path


def _load_numpy_weights() -> np.lib.npyio.NpzFile:
    global _weights
    if _weights is not None:
        return _weights

    if not NPZ_PATH.is_file():
        if MODEL_PATH.is_file():
            export_numpy_weights()
        else:
            raise FileNotFoundError(
                f"Missing {NPZ_PATH.name}. Run: python train_boxer.py"
            )

    _weights = np.load(NPZ_PATH, allow_pickle=False)
    return _weights


def _numpy_forward(state: np.ndarray) -> int:
    w = _load_numpy_weights()
    x = state.astype(np.float32)
    x = np.maximum(0.0, x @ w["net.0.weight"].T + w["net.0.bias"])
    x = np.maximum(0.0, x @ w["net.2.weight"].T + w["net.2.bias"])
    x = x @ w["net.4.weight"].T + w["net.4.bias"]
    return int(np.argmax(x))


def _torch_forward(state: np.ndarray) -> int:
    global _torch_agent
    import torch

    from rl.dqn import DQNAgent

    if _torch_agent is None:
        _torch_agent = DQNAgent(device="cpu")
        _torch_agent.load(str(MODEL_PATH), eval_mode=True)
        _torch_agent.policy_net.eval()

    with torch.no_grad():
        tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = _torch_agent.policy_net(tensor)
        return int(q_values.argmax(dim=1).item())


def get_agent():
    """Compatibility hook — loads NumPy weights (fast) for health checks."""
    _load_numpy_weights()
    return True


def build_state_vector(payload: dict) -> np.ndarray:
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
    state = build_state_vector(payload)
    if NPZ_PATH.is_file() or not MODEL_PATH.is_file():
        action_idx = _numpy_forward(state)
    else:
        action_idx = _torch_forward(state)
    return ACTION_NAMES[action_idx]
