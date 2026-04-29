from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def import_gym_backend():
    """Import Gymnasium when available, otherwise fall back to Gym."""
    try:
        import gymnasium as gym

        return gym, "gymnasium"
    except ImportError:
        import gym

        return gym, "gym"


def make_env(env_id: str, seed: Optional[int] = None, render_mode: Optional[str] = None):
    from .fhn_env import FHN_ENV_ID, FitzHughNagumoStabilizeEnv

    if env_id != FHN_ENV_ID:
        raise ValueError(f"Only {FHN_ENV_ID} is supported by this project.")

    env = FitzHughNagumoStabilizeEnv(render_mode=render_mode)
    seed_spaces(env, seed)
    return env


def seed_spaces(env, seed: Optional[int]) -> None:
    if seed is None:
        return
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        if hasattr(space, "seed"):
            space.seed(seed)


def reset_env(env, seed: Optional[int] = None) -> Tuple[np.ndarray, dict[str, Any]]:
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()

    if isinstance(result, tuple) and len(result) == 2:
        state, info = result
    else:
        state, info = result, {}
    return np.asarray(state, dtype=float), info


def step_env(env, action: int):
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return (
            np.asarray(state, dtype=float),
            float(reward),
            done,
            bool(terminated),
            bool(truncated),
            info,
        )

    state, reward, done, info = result
    return np.asarray(state, dtype=float), float(reward), bool(done), bool(done), False, info


def render_rgb_frame(env):
    try:
        frame = env.render()
    except TypeError:
        frame = env.render(mode="rgb_array")

    if isinstance(frame, list):
        frame = frame[-1] if frame else None
    if frame is None:
        return None
    return np.asarray(frame)


def observation_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(env.observation_space.low, dtype=float)
    high = np.asarray(env.observation_space.high, dtype=float)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError(
            "Tile coding requires finite observation_space.low/high bounds."
        )
    return low, high
