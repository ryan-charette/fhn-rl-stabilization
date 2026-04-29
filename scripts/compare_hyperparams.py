from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sarsa_lambda.experiments import (  # noqa: E402
    ExperimentConfig,
    compare_hyperparameters,
)
from sarsa_lambda.fhn_env import FHN_ENV_ID  # noqa: E402


def parse_tile_width(raw: str) -> tuple[float, ...]:
    return tuple(float(part) for part in raw.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tile-coding hyperparameters on the FitzHugh-Nagumo stabilization task."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--tilings-base", type=int, default=8)
    parser.add_argument("--tile-width-base", type=parse_tile_width, default=(0.25, 0.15))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--tilings", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument(
        "--tile-widths",
        type=parse_tile_width,
        nargs="+",
        default=[(0.2, 0.10), (0.25, 0.15), (0.35, 0.20)],
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = ExperimentConfig(
        env_id=FHN_ENV_ID,
        gamma=args.gamma,
        lam=args.lam,
        alpha=args.alpha,
        epsilon=args.epsilon,
        num_episodes=args.episodes,
        num_tilings=args.tilings_base,
        tile_width=args.tile_width_base,
        seed=args.seeds[0],
        max_steps_per_episode=args.max_steps,
    )
    output_dir = (
        args.output_dir
        / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{FHN_ENV_ID}_comparison"
    )
    compare_hyperparameters(
        base_config=base_config,
        output_dir=output_dir,
        seeds=args.seeds,
        lambdas=args.lambdas,
        alphas=args.alphas,
        tilings=args.tilings,
        tile_widths=args.tile_widths,
    )
    print(f"Saved FitzHugh-Nagumo comparison results to {output_dir}")


if __name__ == "__main__":
    main()
