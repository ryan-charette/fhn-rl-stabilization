from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from .env_compat import make_env, reset_env, step_env
from .features import StateActionFeatureVectorWithTile
from .rendering import greedy_action


def save_fhn_rollout(
    env_id: str,
    weights: np.ndarray,
    features: StateActionFeatureVectorWithTile,
    output_dir: Path,
    seed: Optional[int] = None,
    max_steps: int = 300,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_fhn_rollout(env_id, weights, features, seed=seed, max_steps=max_steps)
    write_rollout_csv(output_dir / "stabilization_rollout.csv", rows)
    plot_fhn_rollout(rows, output_dir / "stabilization_rollout.png")


def collect_fhn_rollout(
    env_id: str,
    weights: np.ndarray,
    features: StateActionFeatureVectorWithTile,
    seed: Optional[int],
    max_steps: int,
) -> list[dict[str, float | int | bool]]:
    env = make_env(env_id, seed=seed)
    rows: list[dict[str, float | int | bool]] = []

    try:
        state, info = reset_env(env, seed=seed)
        done = False
        rows.append(info_to_row(step=0, reward=0.0, info=info))

        for step in range(1, max_steps + 1):
            action = greedy_action(weights, features, state, done)
            state, reward, done, _, _, info = step_env(env, action)
            rows.append(info_to_row(step=step, reward=reward, info=info))
            if done:
                break
    finally:
        env.close()

    return rows


def info_to_row(
    step: int, reward: float, info: dict
) -> dict[str, float | int | bool]:
    return {
        "step": step,
        "time": float(info.get("time", step)),
        "voltage": float(info.get("voltage", np.nan)),
        "recovery": float(info.get("recovery", np.nan)),
        "target_voltage": float(info.get("target_voltage", np.nan)),
        "target_recovery": float(info.get("target_recovery", np.nan)),
        "control_current": float(info.get("control_current", np.nan)),
        "action": int(info.get("action", -1)),
        "reward": float(reward),
        "distance": float(info.get("distance", np.nan)),
        "inside_success_region": bool(info.get("inside_success_region", False)),
    }


def write_rollout_csv(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    fieldnames = [
        "step",
        "time",
        "voltage",
        "recovery",
        "target_voltage",
        "target_recovery",
        "control_current",
        "action",
        "reward",
        "distance",
        "inside_success_region",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_fhn_rollout(rows: list[dict[str, float | int | bool]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray([row["time"] for row in rows], dtype=float)
    voltage = np.asarray([row["voltage"] for row in rows], dtype=float)
    recovery = np.asarray([row["recovery"] for row in rows], dtype=float)
    current = np.asarray([row["control_current"] for row in rows], dtype=float)
    distance = np.asarray([row["distance"] for row in rows], dtype=float)
    target_voltage = float(rows[0]["target_voltage"])
    target_recovery = float(rows[0]["target_recovery"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(time, voltage, color="#111827", linewidth=2.0, label="voltage")
    axes[0].axhline(target_voltage, color="#0f766e", linewidth=1.4, label="target")
    axes[0].set_ylabel("voltage v")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(time, recovery, color="#a95d35", linewidth=2.0, label="recovery")
    axes[1].axhline(target_recovery, color="#0f766e", linewidth=1.4, label="target")
    axes[1].set_ylabel("recovery w")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].step(time, current, where="post", color="#7c3aed", linewidth=1.8, label="current")
    axes[2].plot(time, distance, color="#64748b", linewidth=1.4, label="scaled distance")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("control / distance")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle("Learned FitzHugh-Nagumo stabilization rollout")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
