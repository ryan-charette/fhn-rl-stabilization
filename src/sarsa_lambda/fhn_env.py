from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


FHN_ENV_ID = "FitzHughNagumoStabilize-v0"


@dataclass(frozen=True)
class FitzHughNagumoParams:
    a: float = 0.7
    b: float = 0.8
    tau: float = 12.5
    background_current: float = 0.0


class FitzHughNagumoStabilizeEnv(gym.Env):
    """Control a simplified excitable heart-cell voltage model.

    The state is `(v, w)`, where `v` is membrane voltage and `w` is a slow
    recovery variable. The action is a bounded stimulation current. The reward
    encourages stabilizing the cell near its quiescent equilibrium while using
    little control effort.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        params: FitzHughNagumoParams = FitzHughNagumoParams(),
        dt: float = 0.05,
        integration_steps: int = 4,
        episode_steps: int = 300,
        action_currents: tuple[float, ...] = (-0.8, -0.4, 0.0, 0.4, 0.8),
        state_low: tuple[float, float] = (-2.5, -1.5),
        state_high: tuple[float, float] = (2.5, 2.0),
        initial_radius: tuple[float, float] = (1.15, 0.65),
        success_radius: tuple[float, float] = (0.12, 0.10),
        success_hold_steps: int = 20,
    ):
        self.render_mode = render_mode
        self.params = params
        self.dt = float(dt)
        self.integration_steps = int(integration_steps)
        self.episode_steps = int(episode_steps)
        self.action_currents = np.asarray(action_currents, dtype=np.float32)
        self.state_low = np.asarray(state_low, dtype=np.float32)
        self.state_high = np.asarray(state_high, dtype=np.float32)
        self.initial_radius = np.asarray(initial_radius, dtype=float)
        self.success_radius = np.asarray(success_radius, dtype=float)
        self.success_hold_steps = int(success_hold_steps)

        if self.integration_steps <= 0:
            raise ValueError("integration_steps must be positive.")
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive.")
        if len(self.action_currents) < 2:
            raise ValueError("At least two stimulation actions are required.")

        self.action_space = spaces.Discrete(len(self.action_currents))
        self.observation_space = spaces.Box(
            low=self.state_low,
            high=self.state_high,
            shape=(2,),
            dtype=np.float32,
        )
        self.target_state = self._find_equilibrium()
        self._rng = np.random.default_rng()
        self.state = self.target_state.copy()
        self.step_count = 0
        self.success_count = 0
        self.last_action = int(np.argmin(np.abs(self.action_currents)))
        self.last_current = 0.0
        self.history: deque[dict[str, float]] = deque(maxlen=450)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            super().reset(seed=seed)
        except TypeError:
            pass
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        if "state" in options:
            state = np.asarray(options["state"], dtype=float)
        else:
            perturbation = self._rng.uniform(-self.initial_radius, self.initial_radius)
            state = self.target_state + perturbation

        self.state = np.clip(state, self.state_low, self.state_high).astype(float)
        self.step_count = 0
        self.success_count = 0
        self.last_action = int(np.argmin(np.abs(self.action_currents)))
        self.last_current = float(self.action_currents[self.last_action])
        self.history.clear()
        self._record_history(reward=0.0, distance=self._target_distance())
        return self._observation(), self._info(reward=0.0)

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"action must be in [0, {self.action_space.n}).")

        self.last_action = action
        self.last_current = float(self.action_currents[action])
        for _ in range(self.integration_steps):
            self.state = self._rk4_step(self.state, self.last_current)

        self.step_count += 1
        out_of_bounds = bool(
            np.any(self.state < self.state_low) or np.any(self.state > self.state_high)
        )
        self.state = np.clip(self.state, self.state_low, self.state_high)

        reward = self._reward(self.last_current)
        distance = self._target_distance()
        if self._inside_success_region():
            self.success_count += 1
        else:
            self.success_count = 0

        terminated = out_of_bounds or self.success_count >= self.success_hold_steps
        truncated = self.step_count >= self.episode_steps
        self._record_history(reward=reward, distance=distance)
        return self._observation(), reward, terminated, truncated, self._info(reward)

    def render(self):
        if self.render_mode not in {None, "rgb_array", "human"}:
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")
        return self._render_frame()

    def close(self):
        return None

    def _derivatives(self, state: np.ndarray, control_current: float) -> np.ndarray:
        v, w = state
        total_current = self.params.background_current + control_current
        dv = v - (v**3) / 3.0 - w + total_current
        dw = (v + self.params.a - self.params.b * w) / self.params.tau
        return np.asarray([dv, dw], dtype=float)

    def _rk4_step(self, state: np.ndarray, control_current: float) -> np.ndarray:
        h = self.dt
        k1 = self._derivatives(state, control_current)
        k2 = self._derivatives(state + 0.5 * h * k1, control_current)
        k3 = self._derivatives(state + 0.5 * h * k2, control_current)
        k4 = self._derivatives(state + h * k3, control_current)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _find_equilibrium(self) -> np.ndarray:
        xs = np.linspace(self.state_low[0], self.state_high[0], 5000)
        w_nullcline = (xs + self.params.a) / self.params.b
        residual = xs - (xs**3) / 3.0 - w_nullcline + self.params.background_current
        idx = int(np.argmin(np.abs(residual)))
        v_target = float(xs[idx])
        w_target = float((v_target + self.params.a) / self.params.b)
        return np.asarray([v_target, w_target], dtype=float)

    def _reward(self, control_current: float) -> float:
        error = self.state - self.target_state
        voltage_penalty = (error[0] / 0.65) ** 2
        recovery_penalty = (error[1] / 0.45) ** 2
        effort_penalty = (control_current / float(np.max(np.abs(self.action_currents)))) ** 2
        reward = 1.0 - 2.2 * voltage_penalty - 0.7 * recovery_penalty - 0.04 * effort_penalty
        if self._inside_success_region():
            reward += 1.0
        return float(reward)

    def _target_distance(self) -> float:
        scaled = (self.state - self.target_state) / np.asarray([0.65, 0.45])
        return float(np.linalg.norm(scaled))

    def _inside_success_region(self) -> bool:
        return bool(np.all(np.abs(self.state - self.target_state) <= self.success_radius))

    def _observation(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def _info(self, reward: float) -> dict[str, float | int | bool]:
        return {
            "voltage": float(self.state[0]),
            "recovery": float(self.state[1]),
            "target_voltage": float(self.target_state[0]),
            "target_recovery": float(self.target_state[1]),
            "control_current": self.last_current,
            "action": self.last_action,
            "reward": float(reward),
            "distance": self._target_distance(),
            "inside_success_region": self._inside_success_region(),
            "time": self.step_count * self.dt * self.integration_steps,
        }

    def _record_history(self, reward: float, distance: float) -> None:
        self.history.append(
            {
                "time": self.step_count * self.dt * self.integration_steps,
                "voltage": float(self.state[0]),
                "recovery": float(self.state[1]),
                "control_current": self.last_current,
                "reward": float(reward),
                "distance": float(distance),
            }
        )

    def _render_frame(self) -> np.ndarray:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        history = list(self.history)
        times = np.asarray([point["time"] for point in history])
        voltages = np.asarray([point["voltage"] for point in history])
        recoveries = np.asarray([point["recovery"] for point in history])
        currents = np.asarray([point["control_current"] for point in history])
        distances = np.asarray([point["distance"] for point in history])

        fig = Figure(figsize=(7.2, 4.4), dpi=100)
        canvas = FigureCanvasAgg(fig)
        grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0])
        phase_ax = fig.add_subplot(grid[:, 0])
        voltage_ax = fig.add_subplot(grid[0, 1])
        current_ax = fig.add_subplot(grid[1, 1])

        v_grid = np.linspace(self.state_low[0], self.state_high[0], 300)
        v_nullcline = (
            v_grid - (v_grid**3) / 3.0 + self.params.background_current
        )
        w_nullcline = (v_grid + self.params.a) / self.params.b
        phase_ax.plot(v_grid, v_nullcline, color="#2f6f8f", linewidth=1.2, label="dv/dt=0")
        phase_ax.plot(v_grid, w_nullcline, color="#a95d35", linewidth=1.2, label="dw/dt=0")
        phase_ax.plot(voltages, recoveries, color="#111827", linewidth=2.0)
        phase_ax.scatter(
            [self.target_state[0]],
            [self.target_state[1]],
            s=52,
            color="#0f766e",
            zorder=3,
            label="target",
        )
        phase_ax.scatter([self.state[0]], [self.state[1]], s=42, color="#b91c1c", zorder=4)
        phase_ax.set_xlim(float(self.state_low[0]), float(self.state_high[0]))
        phase_ax.set_ylim(float(self.state_low[1]), float(self.state_high[1]))
        phase_ax.set_xlabel("voltage v")
        phase_ax.set_ylabel("recovery w")
        phase_ax.set_title("FitzHugh-Nagumo phase plane")
        phase_ax.grid(True, alpha=0.25)
        phase_ax.legend(loc="upper right", fontsize=7)

        voltage_ax.plot(times, voltages, color="#111827", linewidth=1.8)
        voltage_ax.axhline(self.target_state[0], color="#0f766e", linewidth=1.2)
        voltage_ax.set_ylabel("v")
        voltage_ax.set_title("Voltage stabilization")
        voltage_ax.grid(True, alpha=0.25)

        current_ax.step(times, currents, where="post", color="#7c3aed", linewidth=1.6)
        if len(distances):
            current_ax.plot(times, distances, color="#64748b", linewidth=1.0, alpha=0.75)
        current_ax.set_xlabel("time")
        current_ax.set_ylabel("current / distance")
        current_ax.grid(True, alpha=0.25)

        fig.tight_layout()
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
