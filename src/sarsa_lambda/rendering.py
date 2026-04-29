from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .agent import action_values
from .env_compat import make_env, render_rgb_frame, reset_env, step_env
from .features import StateActionFeatureVectorWithTile


def greedy_action(
    weights: np.ndarray,
    features: StateActionFeatureVectorWithTile,
    state: np.ndarray,
    done: bool,
) -> int:
    q_values = action_values(weights, features, state, done)
    return int(np.argmax(q_values))


def render_policy(
    env_id: str,
    weights: np.ndarray,
    features: StateActionFeatureVectorWithTile,
    output_path: Path,
    seed: Optional[int] = None,
    max_steps: int = 1000,
    fps: int = 30,
) -> float:
    env = make_env(env_id, seed=seed, render_mode="rgb_array")
    frames = []
    total_reward = 0.0

    try:
        state, _ = reset_env(env, seed=seed)
        done = False
        first_frame = render_rgb_frame(env)
        if first_frame is not None:
            frames.append(first_frame)

        for _ in range(max_steps):
            action = greedy_action(weights, features, state, done)
            state, reward, done, _, _, _ = step_env(env, action)
            total_reward += reward

            frame = render_rgb_frame(env)
            if frame is not None:
                frames.append(frame)

            if done:
                break
    finally:
        env.close()

    if not frames:
        raise RuntimeError("Environment did not produce RGB frames for rendering.")

    save_frames(frames, output_path, fps=fps)
    return total_reward


def save_frames(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        from PIL import Image

        pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(1, int(round(1000 / fps))),
            loop=0,
        )
    elif suffix in {".mp4", ".mov", ".avi"}:
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        raise ValueError("Output path must end in .gif, .mp4, .mov, or .avi.")
