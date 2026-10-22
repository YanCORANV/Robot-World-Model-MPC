import numpy as np
import torch

from robot_world_model.diagnostics import regime_masks
from robot_world_model.improved_mpc import ContactAwareMPC
from robot_world_model.policies import heuristic_action


def test_heuristic_holds_a_reached_goal():
    state = np.zeros(12, dtype=np.float32)
    state[3:6] = [1.3, 0.7, 0.42]
    goal = np.array([1.33, 0.7, 0.42], dtype=np.float32)
    np.testing.assert_array_equal(heuristic_action(state, goal), np.zeros(4))


def test_regime_masks_are_disjoint_and_complete():
    state = np.zeros((3, 12), dtype=np.float32)
    next_state = state.copy()
    state[0, :3], state[0, 3:6] = [0, 0, 0], [1, 0, 0]
    state[1, :3], state[1, 3:6] = [0, 0, 0], [0.04, 0, 0]
    state[2, :3], state[2, 3:6] = [0, 0, 0], [0.04, 0, 0]
    next_state[:] = state
    next_state[2, 3] += 0.01
    data = {"state": state, "next_state": next_state,
            "pushing": np.array([False, False, True])}
    masks = regime_masks(data, np.arange(3))
    assert [masks[name].tolist() for name in masks] == [
        [True, False, False], [False, True, False], [False, False, True]]


def test_contact_aware_planner_uses_model_only_in_push_phase():
    class CountingModel(torch.nn.Module):
        calls = 0

        def forward(self, state, action):
            self.calls += 1
            return state

    model = CountingModel()
    controller = ContactAwareMPC(model, horizon=2, candidates=8,
                                 iterations=1, seed=1)
    state = np.zeros(12, dtype=np.float32)
    state[3:6] = [1.3, 0.7, 0.42]
    goal = np.array([1.5, 0.7, 0.42], dtype=np.float32)
    state[:3] = [1.0, 0.7, 0.47]
    controller.act(state, goal)
    assert controller.last_phase == "approach" and model.calls == 0
    state[:3] = [1.245, 0.7, 0.435]
    controller.act(state, goal)
    assert controller.last_phase == "model_push" and model.calls > 0
