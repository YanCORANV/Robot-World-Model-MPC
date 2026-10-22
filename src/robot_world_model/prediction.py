"""Held-out one-step and closed-loop multi-step diagnostics."""

import numpy as np
import torch


def evaluate_prediction(model, data, horizon=5, plot_path=None):
    mask = data["split"] == 2
    state = data["state"][mask].astype(np.float32)
    action = data["action"][mask].astype(np.float32)
    target = data["next_state"][mask].astype(np.float32)
    with torch.no_grad():
        predicted = model(torch.from_numpy(state), torch.from_numpy(action)).numpy()
    one_step = {
        "state_rmse_m": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "object_position_rmse_m": float(np.sqrt(np.mean((predicted[:, 3:6] - target[:, 3:6]) ** 2))),
        "zero_delta_state_rmse_m": float(np.sqrt(np.mean((state - target) ** 2))),
        "zero_delta_object_position_rmse_m": float(np.sqrt(np.mean((state[:, 3:6] - target[:, 3:6]) ** 2))),
    }
    # Only contiguous portions of held-out episodes qualify as open-loop rollouts.
    indices = np.flatnonzero(mask)
    valid = []
    for i in range(len(indices) - horizon + 1):
        group = indices[i:i+horizon]
        if (np.all(np.diff(group) == 1)
                and len(set(data["episode_id"][group])) == 1
                and np.all(np.diff(data["timestep"][group]) == 1)):
            valid.append(group)
    errors = []
    for group in valid:
        rollout = torch.from_numpy(data["state"][group[0]].astype(np.float32)[None])
        with torch.no_grad():
            for index in group:
                act = torch.from_numpy(data["action"][index].astype(np.float32)[None])
                rollout = model(rollout, act)
        errors.append((rollout.numpy()[0, 3:6] - data["next_state"][group[-1], 3:6]) ** 2)
    result = {"one_step": one_step,
              "multi_step": {"horizon": horizon, "windows": len(errors),
                             "object_position_rmse_m": float(np.sqrt(np.mean(errors))) if errors else None}}
    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(target[:, 3], target[:, 4], s=8, alpha=0.5, label="ground truth")
        ax.scatter(predicted[:, 3], predicted[:, 4], s=8, alpha=0.5, label="one-step prediction")
        ax.set(xlabel="object x (m)", ylabel="object y (m)", title="Held-out object positions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    return result
