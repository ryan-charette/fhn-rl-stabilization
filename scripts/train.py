from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sarsa_lambda.experiments import (  # noqa: E402
    ExperimentConfig,
    run_training,
    timestamped_run_dir,
)
from sarsa_lambda.features import StateActionFeatureVectorWithTile  # noqa: E402
from sarsa_lambda.fhn_analysis import save_fhn_rollout  # noqa: E402
from sarsa_lambda.fhn_env import FHN_ENV_ID  # noqa: E402


def parse_tile_width(raw: str) -> tuple[float, ...]:
    return tuple(float(part) for part in raw.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a True Online Sarsa(lambda) controller for the FitzHugh-Nagumo stabilization task."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--tilings", type=int, default=8)
    parser.add_argument("--tile-width", type=parse_tile_width, default=(0.25, 0.15))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--render-gif", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_features(weights_path: Path) -> StateActionFeatureVectorWithTile:
    data = np.load(weights_path)
    return StateActionFeatureVectorWithTile(
        state_low=data["state_low"],
        state_high=data["state_high"],
        num_actions=int(data["num_actions"]),
        num_tilings=int(data["num_tilings"]),
        tile_width=data["tile_width"],
    )


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        env_id=FHN_ENV_ID,
        gamma=args.gamma,
        lam=args.lam,
        alpha=args.alpha,
        epsilon=args.epsilon,
        num_episodes=args.episodes,
        num_tilings=args.tilings,
        tile_width=args.tile_width,
        seed=args.seed,
        max_steps_per_episode=args.max_steps,
    )
    output_dir = timestamped_run_dir(args.output_dir, config.env_id, config.seed)
    result = run_training(config, output_dir, render_gif=args.render_gif)
    features = load_features(output_dir / "weights.npz")
    weights = np.load(output_dir / "weights.npz")["weights"]
    save_fhn_rollout(
        env_id=FHN_ENV_ID,
        weights=weights,
        features=features,
        output_dir=output_dir,
        seed=args.seed,
        max_steps=args.rollout_steps,
    )
    print(f"Saved FitzHugh-Nagumo results to {output_dir}")
    print(f"Final return: {result.episode_returns[-1]:.1f}")


if __name__ == "__main__":
    main()
