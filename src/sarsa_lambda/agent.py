from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .env_compat import reset_env, step_env
from .features import StateActionFeatureVectorWithTile


@dataclass
class TrainingResult:
    weights: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray


def action_values(
    weights: np.ndarray,
    features: StateActionFeatureVectorWithTile,
    state: np.ndarray,
    done: bool,
) -> np.ndarray:
    """Return Q(s, a) for every action."""
    values = np.zeros(features.num_actions, dtype=float)
    if done:
        return values

    for action in range(features.num_actions):
        active = features.active_indices(state, done, action)
        values[action] = float(weights[active].sum())
    return values


def epsilon_greedy_action(
    weights: np.ndarray,
    features: StateActionFeatureVectorWithTile,
    state: np.ndarray,
    done: bool,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Choose an epsilon-greedy action with random tie-breaking."""
    if rng.random() < epsilon:
        return int(rng.integers(features.num_actions))

    q_values = action_values(weights, features, state, done)
    best_actions = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best_actions))


def train_sarsa_lambda(
    env,
    gamma: float,
    lam: float,
    alpha: float,
    features: StateActionFeatureVectorWithTile,
    num_episodes: int,
    epsilon: float = 0.0,
    seed: Optional[int] = None,
    max_steps_per_episode: Optional[int] = None,
) -> TrainingResult:
    """Train a linear True Online Sarsa(lambda) agent.

    The implementation follows the True Online Sarsa(lambda) update with Dutch
    traces from van Seijen and Sutton (2014).
    """
    rng = np.random.default_rng(seed)
    weights = np.zeros(features.feature_vector_len(), dtype=float)
    returns = np.zeros(num_episodes, dtype=float)
    lengths = np.zeros(num_episodes, dtype=int)

    for episode in range(num_episodes):
        episode_seed = None if seed is None else seed + episode
        state, _ = reset_env(env, seed=episode_seed)
        done = False
        action = epsilon_greedy_action(
            weights, features, state, done, epsilon, rng
        )
        x = features(state, done, action)
        z = np.zeros_like(weights)
        q_old = 0.0
        total_reward = 0.0
        step_count = 0

        while not done:
            next_state, reward, done, _, _, _ = step_env(env, action)
            next_action = epsilon_greedy_action(
                weights, features, next_state, done, epsilon, rng
            )
            x_next = features(next_state, done, next_action)

            q = float(np.dot(weights, x))
            q_next = float(np.dot(weights, x_next))
            delta = reward + gamma * q_next - q

            z = gamma * lam * z + (1.0 - alpha * gamma * float(np.dot(z, x))) * x
            weights += alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x

            q_old = q_next
            x = x_next
            action = next_action
            total_reward += reward
            step_count += 1

            if max_steps_per_episode is not None and step_count >= max_steps_per_episode:
                break

        returns[episode] = total_reward
        lengths[episode] = step_count

    return TrainingResult(
        weights=weights,
        episode_returns=returns,
        episode_lengths=lengths,
    )
