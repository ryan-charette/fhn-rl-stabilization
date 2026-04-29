from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .agent import train_sarsa_lambda
from .env_compat import import_gym_backend, make_env, observation_bounds
from .features import StateActionFeatureVectorWithTile
from .fhn_env import FHN_ENV_ID
from .plotting import plot_comparison, plot_training_curve
from .rendering import render_policy


@dataclass
class ExperimentConfig:
    env_id: str = FHN_ENV_ID
    gamma: float = 1.0
    lam: float = 0.9
    alpha: float = 0.0125
    epsilon: float = 0.05
    num_episodes: int = 300
    num_tilings: int = 8
    tile_width: tuple[float, ...] = (0.2, 0.02)
    seed: int = 0
    max_steps_per_episode: Optional[int] = None


def timestamped_run_dir(base_dir: Path, env_id: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_env_id = env_id.replace("/", "_")
    return base_dir / f"{stamp}_{safe_env_id}_seed{seed}"


def build_features(env, config: ExperimentConfig) -> StateActionFeatureVectorWithTile:
    state_low, state_high = observation_bounds(env)
    return StateActionFeatureVectorWithTile(
        state_low=state_low,
        state_high=state_high,
        num_actions=env.action_space.n,
        num_tilings=config.num_tilings,
        tile_width=np.asarray(config.tile_width, dtype=float),
    )


def save_config(config: ExperimentConfig, output_dir: Path) -> None:
    gym_backend, backend_name = import_gym_backend()
    payload = asdict(config)
    payload["gym_backend"] = backend_name
    payload["gym_backend_version"] = getattr(gym_backend, "__version__", "unknown")
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_returns(
    output_dir: Path, episode_returns: np.ndarray, episode_lengths: np.ndarray
) -> None:
    with (output_dir / "returns.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length"])
        for episode, (episode_return, episode_length) in enumerate(
            zip(episode_returns, episode_lengths)
        ):
            writer.writerow([episode, float(episode_return), int(episode_length)])

    np.savez_compressed(
        output_dir / "returns.npz",
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )


def save_weights(
    output_dir: Path,
    weights: np.ndarray,
    features: StateActionFeatureVectorWithTile,
) -> None:
    np.savez_compressed(
        output_dir / "weights.npz",
        weights=weights,
        state_low=features.state_low,
        state_high=features.state_high,
        num_actions=np.asarray(features.num_actions),
        num_tilings=np.asarray(features.num_tilings),
        tile_width=features.tile_width,
    )


def run_training(config: ExperimentConfig, output_dir: Path, render_gif: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(config.env_id, seed=config.seed)

    try:
        features = build_features(env, config)
        result = train_sarsa_lambda(
            env=env,
            gamma=config.gamma,
            lam=config.lam,
            alpha=config.alpha,
            features=features,
            num_episodes=config.num_episodes,
            epsilon=config.epsilon,
            seed=config.seed,
            max_steps_per_episode=config.max_steps_per_episode,
        )
    finally:
        env.close()

    save_config(config, output_dir)
    save_returns(output_dir, result.episode_returns, result.episode_lengths)
    save_weights(output_dir, result.weights, features)
    plot_training_curve(
        result.episode_returns,
        output_dir / "training_curve.png",
        title=f"{config.env_id} training curve",
    )

    if render_gif:
        render_policy(
            env_id=config.env_id,
            weights=result.weights,
            features=features,
            output_path=output_dir / "policy.gif",
            seed=config.seed,
        )

    return result


def compare_hyperparameters(
    base_config: ExperimentConfig,
    output_dir: Path,
    seeds: Iterable[int],
    lambdas: Iterable[float],
    alphas: Iterable[float],
    tilings: Iterable[int],
    tile_widths: Iterable[tuple[float, ...]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    curves: dict[str, dict[str, list[np.ndarray]]] = {
        "lambda": {},
        "alpha": {},
        "num_tilings": {},
        "tile_width": {},
    }

    sweeps = [
        ("lambda", "lam", lambdas),
        ("alpha", "alpha", alphas),
        ("num_tilings", "num_tilings", tilings),
        ("tile_width", "tile_width", tile_widths),
    ]

    for sweep_name, attr_name, values in sweeps:
        for value in values:
            value_key = value_to_key(value)
            curves[sweep_name].setdefault(value_key, [])

            for seed in seeds:
                config = ExperimentConfig(**asdict(base_config))
                setattr(config, attr_name, value)
                config.seed = int(seed)

                env = make_env(config.env_id, seed=config.seed)
                try:
                    features = build_features(env, config)
                    result = train_sarsa_lambda(
                        env=env,
                        gamma=config.gamma,
                        lam=config.lam,
                        alpha=config.alpha,
                        features=features,
                        num_episodes=config.num_episodes,
                        epsilon=config.epsilon,
                        seed=config.seed,
                        max_steps_per_episode=config.max_steps_per_episode,
                    )
                finally:
                    env.close()

                curves[sweep_name][value_key].append(result.episode_returns)
                for episode, episode_return in enumerate(result.episode_returns):
                    rows.append(
                        {
                            "sweep": sweep_name,
                            "value": value_key,
                            "seed": int(seed),
                            "episode": episode,
                            "return": float(episode_return),
                            "length": int(result.episode_lengths[episode]),
                        }
                    )

    write_comparison_csv(output_dir / "comparison_results.csv", rows)
    write_summary_json(output_dir / "comparison_summary.json", curves)

    for sweep_name, sweep_curves in curves.items():
        plot_comparison(
            sweep_curves,
            output_dir / f"comparison_{sweep_name}.png",
            title=f"{base_config.env_id}: {sweep_name} comparison",
        )


def value_to_key(value) -> str:
    if isinstance(value, tuple):
        return ",".join(f"{v:g}" for v in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def write_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sweep", "value", "seed", "episode", "return", "length"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(
    path: Path, curves: dict[str, dict[str, list[np.ndarray]]], final_window: int = 25
) -> None:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for sweep_name, sweep_curves in curves.items():
        summary[sweep_name] = {}
        for value_key, runs in sweep_curves.items():
            stacked = np.vstack(runs)
            window = stacked[:, -min(final_window, stacked.shape[1]) :]
            summary[sweep_name][value_key] = {
                "final_window_mean": float(window.mean()),
                "final_window_std": float(window.std()),
                "best_episode_mean": float(stacked.mean(axis=0).max()),
            }

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
