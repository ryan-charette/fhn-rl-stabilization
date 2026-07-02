import numpy as np

from sarsa_lambda.fhn_env import FitzHughNagumoStabilizeEnv


def test_fhn_environment_reset_and_step_are_finite() -> None:
    env = FitzHughNagumoStabilizeEnv(
        episode_steps=2,
        integration_steps=1,
        success_hold_steps=10,
    )
    observation, info = env.reset(seed=123, options={"state": env.target_state})

    assert observation.shape == (2,)
    assert observation.dtype == np.float32
    assert info["inside_success_region"] is True

    neutral_action = int(np.argmin(np.abs(env.action_currents)))
    next_observation, reward, terminated, truncated, step_info = env.step(neutral_action)

    assert next_observation.shape == (2,)
    assert np.isfinite(next_observation).all()
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert step_info["action"] == neutral_action
