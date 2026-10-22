import numpy as np
import torch

from robot_world_model.env import state_from_obs, make_env
from robot_world_model.model import DynamicsMLP
from robot_world_model.policies import heuristic_action


def test_state_velocity_is_absolute():
    raw = np.zeros(25, dtype=np.float32)
    raw[14:17] = [0.01, 0.02, 0.03]
    raw[20:23] = [0.04, 0.05, 0.06]
    state = state_from_obs({"observation": raw})
    np.testing.assert_allclose(state[6:9], [0.05, 0.07, 0.09])
    np.testing.assert_allclose(state[9:12], [0.04, 0.05, 0.06])


def test_model_shape_and_residual():
    model = DynamicsMLP(hidden=16)
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    state = torch.rand(3, 12)
    action = torch.rand(3, 4)
    torch.testing.assert_close(model(state, action), state)


def test_fetchpush_contract_and_step():
    env = make_env()
    try:
        obs, _ = env.reset(seed=1)
        assert obs["observation"].shape == (25,)
        assert obs["desired_goal"].shape == (3,)
        assert env.action_space.shape == (4,)
        action = heuristic_action(state_from_obs(obs), obs["desired_goal"])
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(state_from_obs(next_obs)).all()
        assert reward in (-1.0, 0.0)
        assert not terminated and not truncated
        assert "is_success" in info
    finally:
        env.close()
