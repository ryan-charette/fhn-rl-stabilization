from __future__ import annotations

import numpy as np


class StateActionFeatureVectorWithTile:
    def __init__(
        self,
        state_low: np.ndarray,
        state_high: np.ndarray,
        num_actions: int,
        num_tilings: int,
        tile_width: np.ndarray,
    ):
        """
        state_low: possible minimum value for each state dimension
        state_high: possible maximum value for each state dimension
        num_actions: number of possible actions
        num_tilings: number of offset tilings
        tile_width: tile width for each state dimension
        """
        self.state_low = np.asarray(state_low, dtype=float)
        self.state_high = np.asarray(state_high, dtype=float)
        self.num_actions = int(num_actions)
        self.num_tilings = int(num_tilings)
        self.tile_width = np.asarray(tile_width, dtype=float)

        if self.state_low.shape != self.state_high.shape:
            raise ValueError("state_low and state_high must have the same shape.")
        if self.tile_width.shape != self.state_low.shape:
            raise ValueError("tile_width must match the state dimension.")
        if np.any(self.tile_width <= 0):
            raise ValueError("tile_width values must be positive.")
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if self.num_tilings <= 0:
            raise ValueError("num_tilings must be positive.")

        span = self.state_high - self.state_low
        self.num_tiles_per_dim = np.ceil(span / self.tile_width).astype(int) + 1
        self.num_tiles_total = int(np.prod(self.num_tiles_per_dim))

    def feature_vector_len(self) -> int:
        """Return d = num_actions * num_tilings * num_tiles_total."""
        return int(self.num_actions * self.num_tilings * self.num_tiles_total)

    def active_indices(self, state, done: bool, action: int) -> np.ndarray:
        """Return the active feature indices for a state-action pair."""
        if done:
            return np.array([], dtype=int)
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"action must be in [0, {self.num_actions}).")

        state = np.asarray(state, dtype=float)
        indices = np.zeros(self.num_tilings, dtype=int)

        for tiling in range(self.num_tilings):
            offset = self.state_low - (tiling / self.num_tilings) * self.tile_width
            tile_indices = np.floor((state - offset) / self.tile_width).astype(int)
            tile_indices = np.clip(tile_indices, 0, self.num_tiles_per_dim - 1)

            linear_index = 0
            factor = 1
            for dim, idx in enumerate(tile_indices):
                linear_index += int(idx) * factor
                factor *= int(self.num_tiles_per_dim[dim])

            indices[tiling] = (
                action * (self.num_tilings * self.num_tiles_total)
                + tiling * self.num_tiles_total
                + linear_index
            )

        return indices

    def __call__(self, state, done: bool, action: int) -> np.ndarray:
        """
        Implement x: S+ x A -> [0, 1]^d.

        If done is True, return the all-zero vector.
        """
        x = np.zeros(self.feature_vector_len(), dtype=float)
        indices = self.active_indices(state, done, action)
        x[indices] = 1.0
        return x
