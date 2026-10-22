"""Prediction errors by motion regime; labels are used offline, never by MPC."""

import numpy as np
import torch


REGIMES = ("free_space", "near_contact_static", "actual_pushing")


def regime_masks(data, indices):
    state, next_state = data["state"][indices], data["next_state"][indices]
    separation = np.minimum(np.linalg.norm(state[:, :3] - state[:, 3:6], axis=1),
                            np.linalg.norm(next_state[:, :3] - next_state[:, 3:6], axis=1))
    if "pushing" in data:
        pushing = data["pushing"][indices].astype(bool)
    else:
        moving = np.linalg.norm(next_state[:, 3:5] - state[:, 3:5], axis=1) > 0.002
        pushing = moving & (separation < 0.10)
    near = separation < 0.10
    return {"free_space": ~pushing & ~near,
            "near_contact_static": ~pushing & near,
            "actual_pushing": pushing}


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None


def prediction_report(model, data, split=2, horizon=5):
    indices = np.flatnonzero(data["split"] == split)
    state = data["state"][indices].astype(np.float32)
    action = data["action"][indices].astype(np.float32)
    next_state = data["next_state"][indices].astype(np.float32)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    with torch.no_grad():
        prediction = model(torch.from_numpy(state), torch.from_numpy(action)).numpy()
    masks = regime_masks(data, indices)

    def subset(mask):
        return {"count": int(mask.sum()),
                "model_object_rmse_m": _rmse(prediction[mask, 3:6] - next_state[mask, 3:6]),
                "zero_change_object_rmse_m": _rmse(state[mask, 3:6] - next_state[mask, 3:6]),
                "predicted_object_xy_motion_m": float(np.mean(np.linalg.norm(
                    prediction[mask, 3:5] - state[mask, 3:5], axis=1))) if mask.any() else None,
                "actual_object_xy_motion_m": float(np.mean(np.linalg.norm(
                    next_state[mask, 3:5] - state[mask, 3:5], axis=1))) if mask.any() else None}

    report = {"overall": subset(np.ones(len(indices), dtype=bool)),
              "regimes": {name: subset(mask) for name, mask in masks.items()}}
    # Each window stays inside one held-out episode; model sees recorded actions
    # but never receives true intermediate states after its initial state.
    windows = []
    local_start = []
    for j in range(len(indices) - horizon + 1):
        group = indices[j:j+horizon]
        if (np.all(np.diff(group) == 1) and
            np.all(np.diff(data["timestep"][group]) == 1) and
            np.all(data["episode_id"][group] == data["episode_id"][group[0]])):
            windows.append(group)
            local_start.append(j)
    if windows:
        groups = np.asarray(windows)
        roll = torch.from_numpy(data["state"][groups[:, 0]].astype(np.float32))
        with torch.no_grad():
            for step in range(horizon):
                roll = model(roll, torch.from_numpy(
                    data["action"][groups[:, step]].astype(np.float32)))
        actual = data["next_state"][groups[:, -1], 3:6]
        predicted = roll.numpy()[:, 3:6]
        unchanged = data["state"][groups[:, 0], 3:6]
        start_regimes = {name: mask[np.asarray(local_start)] for name, mask in masks.items()}
        report["multi_step"] = {"horizon": horizon, "windows": len(groups),
            "model_object_rmse_m": _rmse(predicted - actual),
            "zero_change_object_rmse_m": _rmse(unchanged - actual),
            "by_start_regime": {name: {"windows": int(mask.sum()),
                "model_object_rmse_m": _rmse(predicted[mask] - actual[mask]),
                "zero_change_object_rmse_m": _rmse(unchanged[mask] - actual[mask])}
                for name, mask in start_regimes.items()}}
    return report


def plot_regime_comparison(reports, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ("Free space", "Near static", "Pushing")
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, report) in enumerate(reports.items()):
        values = [report["regimes"][regime]["model_object_rmse_m"] for regime in REGIMES]
        ax.bar(x + (i - 0.5) * 0.27, np.asarray(values) * 1000,
               width=0.26, label=name)
    ax.set(xticks=x, xticklabels=labels, ylabel="Object position RMSE (mm)",
           title="Same held-out episodes, different training data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trajectories(trajectories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, trajectory in trajectories.items():
        xy = np.asarray(trajectory["object_xy"])
        ax.plot(xy[:, 0], xy[:, 1], marker=".", markersize=2, label=name)
    first = next(iter(trajectories.values()))
    ax.scatter(first["goal_xy"][0], first["goal_xy"][1], marker="*", s=160,
               c="black", label="goal")
    ax.set(xlabel="Object x (m)", ylabel="Object y (m)",
           title=f"FetchPush trajectories, seed {first['seed']}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
