"""
Custom Gymnasium environment: RobotBoxing-v1

1D boxing ring where the AI agent (right fighter) learns to box against a
rule-based / randomized human proxy (left fighter) during offline training.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- Ring & combat constants ---
RING_MIN = 0.0
RING_MAX = 1.0
STRIKE_DISTANCE = 0.12
MOVE_DELTA = 0.04
DODGE_DELTA = 0.08
MAX_HEALTH = 100.0
MAX_STAMINA = 100.0
JAB_DAMAGE = 12.0
JAB_STAMINA_COST = 15.0
BLOCK_STAMINA_COST = 8.0
DODGE_STAMINA_COST = 12.0
MOVE_STAMINA_COST = 3.0
STAMINA_RECOVERY = 4.0
MAX_STEPS = 120

# Discrete actions (AI and opponent share the same action vocabulary)
ACTION_STEP_LEFT = 0
ACTION_STEP_RIGHT = 1
ACTION_JAB = 2
ACTION_BLOCK = 3
ACTION_DODGE = 4
NUM_ACTIONS = 5

ACTION_NAMES = ["step_left", "step_right", "jab", "block", "dodge"]


def register_robot_boxing_env() -> None:
    """Register RobotBoxing-v1 with Gymnasium (idempotent)."""
    try:
        gym.spec("RobotBoxing-v1")
    except gym.error.Error:
        gym.register(
            id="RobotBoxing-v1",
            entry_point="rl.robot_boxing_env:RobotBoxingEnv",
            max_episode_steps=MAX_STEPS,
        )


class RobotBoxingEnv(gym.Env):
    """
    State (Box, shape=(6,), float32 in [0, 1]):
        0 distance_between_fighters (normalized)
        1 player_health
        2 ai_health
        3 player_stamina
        4 ai_stamina
        5 player_action_state (last action / NUM_ACTIONS)

    Action (Discrete, 5): step_left, step_right, jab, block, dodge — for the AI.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.player_x = 0.25
        self.ai_x = 0.75
        self.player_health = MAX_HEALTH
        self.ai_health = MAX_HEALTH
        self.player_stamina = MAX_STAMINA
        self.ai_stamina = MAX_STAMINA
        self.player_last_action = ACTION_STEP_LEFT
        self.ai_last_action = ACTION_STEP_RIGHT
        self.steps = 0

        # Flags set during step() for reward accounting
        self._player_jabbed = False
        self._ai_jabbed = False
        self._ai_blocked_player = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.player_x = 0.25
        self.ai_x = 0.75
        self.player_health = MAX_HEALTH
        self.ai_health = MAX_HEALTH
        self.player_stamina = MAX_STAMINA
        self.ai_stamina = MAX_STAMINA
        self.player_last_action = ACTION_STEP_LEFT
        self.ai_last_action = ACTION_STEP_RIGHT
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        self._player_jabbed = False
        self._ai_jabbed = False
        self._ai_blocked_player = False

        player_action = self._sample_opponent_action()
        self.player_last_action = player_action
        self.ai_last_action = int(action)

        reward = self._resolve_turn(int(action), player_action)
        self._recover_stamina()

        terminated = self.player_health <= 0 or self.ai_health <= 0
        truncated = self.steps >= MAX_STEPS

        info = {
            "player_action": ACTION_NAMES[player_action],
            "ai_action": ACTION_NAMES[int(action)],
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        distance = abs(self.player_x - self.ai_x) / (RING_MAX - RING_MIN)
        return np.array(
            [
                np.clip(distance, 0.0, 1.0),
                self.player_health / MAX_HEALTH,
                self.ai_health / MAX_HEALTH,
                self.player_stamina / MAX_STAMINA,
                self.ai_stamina / MAX_STAMINA,
                self.player_last_action / (NUM_ACTIONS - 1),
            ],
            dtype=np.float32,
        )

    # --- Combat resolution ---
    def _in_strike_range(self, attacker_x: float, defender_x: float) -> bool:
        return abs(attacker_x - defender_x) <= STRIKE_DISTANCE

    def _resolve_turn(self, ai_action: int, player_action: int) -> float:
        """Apply both fighters' actions and return shaped reward for the AI."""
        reward = 0.0

        # Movement (applied before strikes)
        self._apply_movement(is_ai=False, action=player_action)
        self._apply_movement(is_ai=True, action=ai_action)

        player_blocking = player_action == ACTION_BLOCK
        ai_blocking = ai_action == ACTION_BLOCK
        player_dodging = player_action == ACTION_DODGE
        ai_dodging = ai_action == ACTION_DODGE

        # Player jab
        if player_action == ACTION_JAB and self.player_stamina >= JAB_STAMINA_COST:
            self.player_stamina -= JAB_STAMINA_COST
            self._player_jabbed = True
            if self._in_strike_range(self.player_x, self.ai_x):
                if ai_blocking:
                    pass  # fully blocked
                elif ai_dodging and random.random() < 0.6:
                    pass  # dodged
                else:
                    self.ai_health = max(0.0, self.ai_health - JAB_DAMAGE)
                    reward -= 5.0  # AI got hit
            else:
                reward += 0.0  # player missed — no AI penalty

        # AI jab
        if ai_action == ACTION_JAB and self.ai_stamina >= JAB_STAMINA_COST:
            self.ai_stamina -= JAB_STAMINA_COST
            self._ai_jabbed = True
            if self._in_strike_range(self.ai_x, self.player_x):
                if player_blocking:
                    reward -= 1.0  # whiff against block counts as miss
                elif player_dodging and random.random() < 0.6:
                    reward -= 1.0  # miss
                else:
                    self.player_health = max(0.0, self.player_health - JAB_DAMAGE)
                    reward += 10.0  # landed punch
            else:
                reward -= 1.0  # missed punch

        # Successful AI block against incoming player jab
        if (
            ai_blocking
            and self._player_jabbed
            and self._in_strike_range(self.player_x, self.ai_x)
        ):
            self._ai_blocked_player = True
            reward += 3.0

        # Small time penalty encourages decisive play
        reward -= 0.05
        return reward

    def _apply_movement(self, is_ai: bool, action: int) -> None:
        x = self.ai_x if is_ai else self.player_x
        stamina = self.ai_stamina if is_ai else self.player_stamina

        if action == ACTION_STEP_LEFT:
            if stamina >= MOVE_STAMINA_COST:
                if is_ai:
                    self.ai_stamina -= MOVE_STAMINA_COST
                else:
                    self.player_stamina -= MOVE_STAMINA_COST
                x = max(RING_MIN, x - MOVE_DELTA)
        elif action == ACTION_STEP_RIGHT:
            if stamina >= MOVE_STAMINA_COST:
                if is_ai:
                    self.ai_stamina -= MOVE_STAMINA_COST
                else:
                    self.player_stamina -= MOVE_STAMINA_COST
                x = min(RING_MAX, x + MOVE_DELTA)
        elif action == ACTION_DODGE:
            if stamina >= DODGE_STAMINA_COST:
                if is_ai:
                    self.ai_stamina -= DODGE_STAMINA_COST
                else:
                    self.player_stamina -= DODGE_STAMINA_COST
                # Dodge jumps away from opponent
                if is_ai:
                    x = min(RING_MAX, x + DODGE_DELTA)
                else:
                    x = max(RING_MIN, x - DODGE_DELTA)
        elif action == ACTION_BLOCK:
            if stamina >= BLOCK_STAMINA_COST:
                if is_ai:
                    self.ai_stamina -= BLOCK_STAMINA_COST
                else:
                    self.player_stamina -= BLOCK_STAMINA_COST

        if is_ai:
            self.ai_x = x
        else:
            self.player_x = x

    def _recover_stamina(self) -> None:
        self.player_stamina = min(
            MAX_STAMINA, self.player_stamina + STAMINA_RECOVERY
        )
        self.ai_stamina = min(MAX_STAMINA, self.ai_stamina + STAMINA_RECOVERY)

    def _sample_opponent_action(self) -> int:
        """
        Rule-based / randomized scripting opponent for offline pre-training.
        Biased toward engaging when in range, retreating when low health.
        """
        distance = abs(self.player_x - self.ai_x)
        low_health = self.player_health < 35
        in_range = distance <= STRIKE_DISTANCE

        roll = random.random()
        if low_health and roll < 0.45:
            return random.choice([ACTION_BLOCK, ACTION_DODGE, ACTION_STEP_LEFT])
        if in_range and roll < 0.55:
            return random.choice([ACTION_JAB, ACTION_JAB, ACTION_BLOCK])
        if distance > STRIKE_DISTANCE + 0.05 and roll < 0.6:
            return ACTION_STEP_RIGHT if self.player_x < self.ai_x else ACTION_STEP_LEFT
        return random.randint(0, NUM_ACTIONS - 1)


# Auto-register when module is imported
register_robot_boxing_env()
