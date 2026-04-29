from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sarsa_lambda.features import StateActionFeatureVectorWithTile  # noqa: E402
from sarsa_lambda.fhn_env import FHN_ENV_ID  # noqa: E402
from sarsa_lambda.rendering import render_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a saved greedy policy.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def env_from_config(weights_path: Path) -> str:
    config_path = weights_path.parent / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)["env_id"]
    return FHN_ENV_ID


def main() -> None:
    args = parse_args()
    data = np.load(args.weights)
    features = StateActionFeatureVectorWithTile(
        state_low=data["state_low"],
        state_high=data["state_high"],
        num_actions=int(data["num_actions"]),
        num_tilings=int(data["num_tilings"]),
        tile_width=data["tile_width"],
    )
    output_path = args.output or args.weights.with_name("policy.gif")
    env_id = env_from_config(args.weights)
    total_reward = render_policy(
        env_id=env_id,
        weights=data["weights"],
        features=features,
        output_path=output_path,
        seed=args.seed,
        max_steps=args.max_steps,
        fps=args.fps,
    )
    print(f"Saved policy media to {output_path}")
    print(f"Rendered return: {total_reward:.1f}")


if __name__ == "__main__":
    main()
