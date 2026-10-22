import numpy as np
import torch

from robot_world_model.improved_mpc import (ContactAwareMPC, FixedPushController,
                                            ZeroChangeDynamics, approach_action)
from robot_world_model.phase3 import paired_stats, wilson


def test_shared_geometric_stage_and_distinct_push_actions():
    state = np.zeros(12, dtype=np.float32)
    state[3:6] = [1.3, .7, .42]
    goal = np.array([1.5, .7, .42], dtype=np.float32)
    state[:3] = [1.0, .7, .47]
    action, phase, _, _ = approach_action(state, goal)
    fixed = FixedPushController()
    cem = ContactAwareMPC(ZeroChangeDynamics(), horizon=2, candidates=8, iterations=1)
    np.testing.assert_allclose(fixed.act(state, goal)[0], action)
    np.testing.assert_allclose(cem.act(state, goal)[0], action)
    assert fixed.last_phase == cem.last_phase == phase == "approach"
    state[:3] = [1.245, .7, .435]
    assert approach_action(state, goal)[0] is None
    np.testing.assert_allclose(fixed.act(state, goal)[0][:2], [.8, 0], atol=1e-6)
    assert cem.act(state, goal)[0].shape == (4,)


def test_zero_change_dynamics_and_paired_statistics():
    state = torch.randn(2,12)
    action = torch.randn(2,4)
    torch.testing.assert_close(ZeroChangeDynamics()(state,action),state)
    a = [{"seed": i, "final_success": i < 3, "final_distance_m": .02 if i < 3 else .2} for i in range(4)]
    b = [{"seed": i, "final_success": i < 1, "final_distance_m": .02 if i < 1 else .2} for i in range(4)]
    result = paired_stats(a,b)
    assert result["a_only_success"] == 2 and result["b_only_success"] == 0
    assert result["success_rate_difference"] == .5
    assert result["exact_mcnemar_two_sided_p"] == .5
    assert wilson(0, 10)[0] == 0
