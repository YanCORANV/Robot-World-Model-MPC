"""Geometric approach followed by model-only CEM push planning."""

import time
import numpy as np
import torch


def approach_action(state, goal):
    """Shared observed-state geometry for every staged push ablation.

    Returns (action, phase, direction, push_height); action is None at push time.
    No simulator future or learned prediction is used here.
    """
    grip, obj = np.asarray(state[:3]), np.asarray(state[3:6])
    displacement = np.asarray(goal[:2]) - obj[:2]
    distance = np.linalg.norm(displacement)
    action = np.zeros(4, dtype=np.float32)
    if np.linalg.norm(np.asarray(goal) - obj) < 0.045:
        return action, "hold", None, None
    direction = displacement / max(distance, 1e-6)
    behind = obj[:2] - 0.055 * direction
    push_height = obj[2] + 0.015
    if np.linalg.norm(grip[:2] - behind) > 0.022:
        target = np.r_[behind, max(push_height + 0.035, grip[2])]
        action[:3] = np.clip((target - grip) / 0.05, -1, 1)
        return action, "approach", direction, push_height
    if abs(grip[2] - push_height) > 0.018:
        target = np.r_[behind, push_height]
        action[:3] = np.clip((target - grip) / 0.05, -1, 1)
        return action, "descend", direction, push_height
    return None, "model_push", direction, push_height


class ZeroChangeDynamics(torch.nn.Module):
    """Literal no-change predictor for all 12 state coordinates."""

    def forward(self, state, action):
        return state


class FixedPushController:
    """Same geometric stage as CEM; fixed 0.8 forward push after contact setup."""

    def __init__(self):
        self.last_phase = "approach"

    def act(self, state, goal):
        started = time.perf_counter()
        action, phase, direction, push_height = approach_action(state, goal)
        self.last_phase = phase
        if action is None:
            action = np.zeros(4, dtype=np.float32)
            action[:2] = 0.8 * direction
            action[2] = np.clip((push_height - state[2]) / 0.05, -0.5, 0.5)
        return action, time.perf_counter() - started


class ContactAwareMPC:
    def __init__(self, model, horizon=10, candidates=64, iterations=3, seed=7):
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.model = model.eval()
        self.horizon = horizon
        self.candidates = candidates
        self.iterations = iterations
        self.elites = max(4, candidates // 6)
        self.rng = np.random.default_rng(seed)
        self.last_phase = "approach"

    def act(self, state, goal):
        started = time.perf_counter()
        action, phase, direction, push_height = approach_action(state, goal)
        self.last_phase = phase
        if action is not None:
            return action, time.perf_counter() - started

        grip = np.asarray(state[:3])
        action = np.zeros(4, dtype=np.float32)
        # Restricted action family: forward push, small lateral correction,
        # and height correction. This keeps CEM in its observed data support.
        mean = np.zeros((self.horizon, 3), dtype=np.float32)
        mean[:, 0] = 0.8  # forward magnitude
        mean[:, 2] = np.clip((push_height - grip[2]) / 0.05, -0.5, 0.5)
        std = np.tile(np.array([0.35, 0.20, 0.15], dtype=np.float32), (self.horizon, 1))
        perp = np.array([-direction[1], direction[0]])
        initial = torch.as_tensor(state, dtype=torch.float32)
        desired = torch.as_tensor(goal[:2], dtype=torch.float32)
        with torch.no_grad():
            for _ in range(self.iterations):
                samples = self.rng.normal(mean, std,
                    (self.candidates, self.horizon, 3)).astype(np.float32)
                samples[:, :, 0] = np.clip(samples[:, :, 0], 0, 1)
                samples[:, :, 1] = np.clip(samples[:, :, 1], -0.35, 0.35)
                samples[:, :, 2] = np.clip(samples[:, :, 2], -0.5, 0.5)
                samples[0] = mean
                actions = torch.zeros(self.candidates, self.horizon, 4)
                actions[:, :, :2] = torch.from_numpy(
                    samples[:, :, 0, None] * direction + samples[:, :, 1, None] * perp)
                actions[:, :, 2] = torch.from_numpy(samples[:, :, 2])
                states = initial.repeat(self.candidates, 1)
                cost = torch.zeros(self.candidates)
                for t in range(self.horizon):
                    before = states
                    states = self.model(states, actions[:, t])
                    goal_distance = torch.linalg.norm(states[:, 3:5] - desired, dim=1)
                    separation = torch.linalg.norm(before[:, :3] - before[:, 3:6], dim=1)
                    object_motion = torch.linalg.norm(states[:, 3:5] - before[:, 3:5], dim=1)
                    # A predicted object displacement without nearby gripper
                    # is likely model bias, not a useful push.
                    phantom = (separation > 0.09).float() * object_motion
                    grip_to_object = torch.linalg.norm(states[:, :2] - states[:, 3:5], dim=1)
                    height_error = torch.abs(states[:, 2] - (states[:, 5] + 0.015))
                    cost += 0.3 * goal_distance + 8.0 * phantom
                    cost += 0.25 * torch.relu(grip_to_object - 0.07) + 0.1 * height_error
                    cost += 0.002 * torch.sum(actions[:, t, :3] ** 2, dim=1)
                cost += 4.0 * torch.linalg.norm(states[:, 3:5] - desired, dim=1)
                elite = samples[torch.topk(cost, self.elites, largest=False).indices.numpy()]
                mean = elite.mean(0)
                std = np.maximum(elite.std(0), np.array([0.10, 0.05, 0.05]))
        action[:2] = np.clip(mean[0, 0] * direction + mean[0, 1] * perp, -1, 1)
        action[2] = np.clip(mean[0, 2], -1, 1)
        return action, time.perf_counter() - started
