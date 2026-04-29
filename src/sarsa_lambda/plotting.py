from __future__ import annotations

from pathlib import Path

import numpy as np


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curve(
    episode_returns: np.ndarray,
    output_path: Path,
    title: str,
    smoothing_window: int = 20,
) -> None:
    plt = _matplotlib()
    episodes = np.arange(len(episode_returns))
    smoothed = moving_average(episode_returns, smoothing_window)
    smoothed_episodes = episodes[-len(smoothed) :]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, episode_returns, color="#9aa3af", linewidth=1, alpha=0.55, label="return")
    ax.plot(
        smoothed_episodes,
        smoothed,
        color="#0f766e",
        linewidth=2.4,
        label=f"{smoothing_window}-episode average",
    )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_comparison(
    curves: dict[str, list[np.ndarray]],
    output_path: Path,
    title: str,
    smoothing_window: int = 10,
) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, runs in curves.items():
        stacked = np.vstack(runs)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        smooth_mean = moving_average(mean, smoothing_window)
        smooth_std = moving_average(std, smoothing_window)
        episodes = np.arange(len(mean))[-len(smooth_mean) :]

        ax.plot(episodes, smooth_mean, linewidth=2, label=label)
        ax.fill_between(
            episodes,
            smooth_mean - smooth_std,
            smooth_mean + smooth_std,
            alpha=0.16,
        )

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.25)
    ax.legend(title="value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
