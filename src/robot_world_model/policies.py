"""Simple hand designed push policy, used for coverage and as a baseline."""

import numpy as np


def heuristic_action(state, goal, noise=0.0, rng=None):
    gripper = np.asarray(state[:3])
    obj = np.asarray(state[3:6])
    toward = np.asarray(goal[:2]) - obj[:2]
    norm = np.linalg.norm(toward)
    direction = toward / max(norm, 1e-6)
    # Approach from behind the block, then push toward the target.
    behind = obj[:2] - 0.055 * direction
    xy_error = behind - gripper[:2]
    target_z = obj[2] + 0.012
    if np.linalg.norm(xy_error) > 0.025:
        target = np.array([behind[0], behind[1], max(target_z + 0.035, gripper[2])])
    elif abs(gripper[2] - target_z) > 0.015:
        target = np.array([behind[0], behind[1], target_z])
    else:
        target = np.array([obj[0] + 0.09 * direction[0],
                           obj[1] + 0.09 * direction[1], target_z])
    action = np.zeros(4, dtype=np.float32)
    # Fetch translates roughly 5 cm for unit Cartesian action.
    action[:3] = np.clip((target - gripper) / 0.05, -1, 1)
    if noise and rng is not None:
        action[:3] += rng.normal(0, noise, 3)
    return np.clip(action, -1, 1).astype(np.float32)
