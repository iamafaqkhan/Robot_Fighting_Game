"""
Deep Q-Network (DQN) agent with experience replay and epsilon-greedy exploration.

Shared by offline training (train_boxer.py) and live inference (server.py).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.robot_boxing_env import ACTION_NAMES, NUM_ACTIONS

STATE_DIM = 6


class QNetwork(nn.Module):
    """Fully-connected Q-value approximator."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 50_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995  # per episode
    target_update_freq: int = 10  # episodes
    min_buffer_size: int = 500


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN with target network, replay buffer, and epsilon-greedy policy."""

    def __init__(self, config: DQNConfig | None = None, device: str | None = None):
        self.config = config or DQNConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.replay = ReplayBuffer(self.config.buffer_capacity)
        self.epsilon = self.config.epsilon_start

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(tensor)
            return int(q_values.argmax(dim=1).item())

    def action_name(self, action_index: int) -> str:
        return ACTION_NAMES[action_index]

    def predict_action_name(self, state: np.ndarray) -> str:
        """Greedy inference for deployment (no exploration)."""
        return self.action_name(self.select_action(state, explore=False))

    def optimize(self) -> float | None:
        if len(self.replay) < max(self.config.batch_size, self.config.min_buffer_size):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size
        )

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "config": self.config,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str, eval_mode: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        if eval_mode:
            self.policy_net.eval()
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
