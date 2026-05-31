#!/usr/bin/env python3
"""
Layer 1 — Offline DQN pre-training for Robot Boxing AI.

Trains against a rule-based / randomized opponent for 2,000 episodes, then saves:
  - robot_boxer.pt          (PyTorch weights for server inference)
  - training_performance.png (Reward vs Episode plot for the project report)

Usage (from project root):
    pip install -r requirements.txt
    python train_boxer.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from rl.dqn import DQNAgent, DQNConfig
from rl.robot_boxing_env import RobotBoxingEnv, register_robot_boxing_env

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "robot_boxer.pt"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "training_performance.png"
DEFAULT_EPISODES = 2000


def train(
    episodes: int = DEFAULT_EPISODES,
    model_path: Path = DEFAULT_MODEL_PATH,
    plot_path: Path = DEFAULT_PLOT_PATH,
) -> list[float]:
    register_robot_boxing_env()
    env = gym.make("RobotBoxing-v1")

    config = DQNConfig(
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=10,
    )
    agent = DQNAgent(config=config)

    episode_rewards: list[float] = []
    steps_per_episode: list[int] = []

    print(f"Training DQN on RobotBoxing-v1 for {episodes} episodes...")
    print(f"Device: {agent.device}")
    start = time.time()

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay.push(state, action, reward, next_state, done)
            agent.optimize()

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        if episode % config.target_update_freq == 0:
            agent.sync_target_network()

        episode_rewards.append(total_reward)
        steps_per_episode.append(steps)

        if episode % 100 == 0 or episode == 1:
            window = episode_rewards[-100:]
            avg = np.mean(window)
            print(
                f"Episode {episode:4d}/{episodes} | "
                f"Reward: {total_reward:7.2f} | "
                f"Avg(100): {avg:7.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Steps: {steps}"
            )

    elapsed = time.time() - start
    env.close()

    agent.save(str(model_path))
    print(f"\nSaved model weights -> {model_path}")
    print(f"Training finished in {elapsed:.1f}s")

    _save_reward_plot(episode_rewards, plot_path)
    print(f"Saved training graph -> {plot_path}")

    return episode_rewards


def _save_reward_plot(episode_rewards: list[float], plot_path: Path) -> None:
    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards, dtype=np.float32)

    # Smoothed curve for readability in the report
    window = 50
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smooth_x = episodes[window - 1 :]
    else:
        smoothed = rewards
        smooth_x = episodes

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.25, color="#4a90d9", label="Episode reward")
    ax.plot(smooth_x, smoothed, color="#c0392b", linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Robot Boxing DQN — Reward vs Episode")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Robot Boxing DQN agent")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    args = parser.parse_args()

    train(episodes=args.episodes, model_path=args.model, plot_path=args.plot)


if __name__ == "__main__":
    main()
