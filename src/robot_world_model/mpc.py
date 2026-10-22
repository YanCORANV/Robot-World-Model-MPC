"""CEM plans using only model-predicted states and a hand designed goal cost."""

import time
import numpy as np
import torch
from .policies import heuristic_action


class CEMMPC:
    def __init__(self, model, horizon=8, candidates=64, iterations=3,
                 elite_fraction=0.15, seed=7):
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.model = model.eval()
        self.horizon = horizon
        self.candidates = candidates
        self.iterations = iterations
        self.elite_count = max(2, int(candidates * elite_fraction))
        self.rng = np.random.default_rng(seed)

    def act(self, state, goal):
        started = time.perf_counter()
        mean = np.tile(heuristic_action(state, goal)[:3], (self.horizon, 1))
        std = np.full_like(mean, 0.8)
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32))
        goal_t = torch.from_numpy(np.asarray(goal, dtype=np.float32))
        with torch.no_grad():
            for _ in range(self.iterations):
                samples = np.clip(self.rng.normal(mean, std,
                    (self.candidates, self.horizon, 3)), -1, 1).astype(np.float32)
                # Keep one nominal plan for stability in a small candidate batch.
                samples[0] = np.clip(mean, -1, 1)
                actions = torch.zeros(self.candidates, self.horizon, 4)
                actions[:, :, :3] = torch.from_numpy(samples)
                states = state_t.repeat(self.candidates, 1)
                cost = torch.zeros(self.candidates)
                for t in range(self.horizon):
                    states = self.model(states, actions[:, t])
                    object_xy = states[:, 3:5]
                    to_goal = goal_t[:2] - object_xy
                    direction = to_goal / torch.linalg.norm(to_goal, dim=1, keepdim=True).clamp_min(1e-5)
                    behind = object_xy - 0.055 * direction
                    grip_to_behind = torch.linalg.norm(states[:, :2] - behind, dim=1)
                    height_error = torch.abs(states[:, 2] - (states[:, 5] + 0.012))
                    # Human-designed planning cost. All future states come from the MLP.
                    cost += 0.35 * torch.linalg.norm(to_goal, dim=1)
                    cost += 0.08 * grip_to_behind + 0.03 * height_error
                    cost += 0.002 * torch.sum(actions[:, t, :3] ** 2, dim=1)
                cost += 2.0 * torch.linalg.norm(states[:, 3:5] - goal_t[:2], dim=1)
                elites = samples[torch.topk(cost, self.elite_count, largest=False).indices.numpy()]
                mean = elites.mean(axis=0)
                std = np.maximum(elites.std(axis=0), 0.12)
        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(mean[0], -1, 1)
        return action, time.perf_counter() - started
