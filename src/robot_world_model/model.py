"""Normalized residual forward dynamics. Goal is deliberately absent."""

from pathlib import Path
import numpy as np
import torch
from torch import nn

from .env import STATE_DIM, ACTION_DIM


class DynamicsMLP(nn.Module):
    def __init__(self, hidden=128, state_dim=STATE_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(nn.Linear(state_dim + ACTION_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, state_dim))
        self.register_buffer("input_mean", torch.zeros(state_dim + ACTION_DIM))
        self.register_buffer("input_std", torch.ones(state_dim + ACTION_DIM))
        self.register_buffer("delta_mean", torch.zeros(state_dim))
        self.register_buffer("delta_std", torch.ones(state_dim))

    def fit_normalizers(self, state, action, next_state):
        x = torch.as_tensor(np.concatenate([state, action], axis=1), dtype=torch.float32)
        y = torch.as_tensor(next_state - state, dtype=torch.float32)
        self.input_mean.copy_(x.mean(0))
        self.input_std.copy_(x.std(0).clamp_min(1e-3))
        self.delta_mean.copy_(y.mean(0))
        self.delta_std.copy_(y.std(0).clamp_min(1e-4))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        delta = self.net((x - self.input_mean) / self.input_std)
        return state + delta * self.delta_std + self.delta_mean


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = DynamicsMLP(checkpoint["hidden"], checkpoint.get("state_dim", STATE_DIM))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def train(data, path, epochs=35, batch_size=128, seed=7, hidden=128,
          regime_weights=None, sample_weights=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    state_dim = data["state"].shape[1]
    model = DynamicsMLP(hidden, state_dim)
    train_mask = data["split"] == 0
    val_mask = data["split"] == 1
    model.fit_normalizers(data["state"][train_mask], data["action"][train_mask],
                          data["next_state"][train_mask])
    state = torch.from_numpy(data["state"].astype(np.float32))
    action = torch.from_numpy(data["action"].astype(np.float32))
    next_state = torch.from_numpy(data["next_state"].astype(np.float32))
    if sample_weights is not None:
        if len(sample_weights) != len(state):
            raise ValueError("sample_weights must match transitions")
        weights = torch.as_tensor(sample_weights, dtype=torch.float32)
    elif regime_weights is None:
        weights = torch.ones(len(state))
    else:
        if "pushing" not in data:
            raise ValueError("regime weighting requires pushing labels")
        separation = np.minimum(np.linalg.norm(data["state"][:, :3] - data["state"][:, 3:6], axis=1),
                                np.linalg.norm(data["next_state"][:, :3] - data["next_state"][:, 3:6], axis=1))
        near_static = (separation < 0.10) & ~data["pushing"].astype(bool)
        values = np.ones(len(state), dtype=np.float32)
        values[near_static] = regime_weights[0]
        values[data["pushing"].astype(bool)] = regime_weights[1]
        weights = torch.from_numpy(values)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.as_tensor(np.random.permutation(np.flatnonzero(train_mask)))
        for batch in order.split(batch_size):
            predicted = model(state[batch], action[batch])
            per_sample = (((predicted - next_state[batch]) / model.delta_std) ** 2).mean(1)
            loss = (per_sample * weights[batch]).sum() / weights[batch].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(state[val_mask], action[val_mask])
            train_pred = model(state[train_mask], action[train_mask])
            train_loss = float((((train_pred - next_state[train_mask]) / model.delta_std) ** 2).mean())
            val_per_sample = (((val_pred - next_state[val_mask]) / model.delta_std) ** 2).mean(1)
            val_loss = float((val_per_sample * weights[val_mask]).sum() / weights[val_mask].sum())
        history.append({"epoch": epoch + 1, "train_normalized_mse": train_loss,
                        "validation_normalized_mse": val_loss})
        if val_loss < best:
            best = val_loss
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "hidden": hidden,
                        "state_dim": state_dim,
                        "seed": seed, "best_epoch": epoch + 1,
                        "regime_weights": regime_weights,
                        "has_sample_weights": sample_weights is not None}, path)
    return {"best_validation_normalized_mse": best,
            "best_epoch": min(history, key=lambda row: row["validation_normalized_mse"])["epoch"],
            "last_train_normalized_mse": history[-1]["train_normalized_mse"],
            "history": history}
