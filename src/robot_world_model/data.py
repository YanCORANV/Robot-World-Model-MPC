"""Episode-isolated transition collection and loading."""

from pathlib import Path
import numpy as np
from .env import make_env, state_from_obs, goal_from_obs
from .policies import heuristic_action


def collect(path, episodes=18, seed=7, horizon=50):
    if episodes < 9:
        raise ValueError("At least 9 episodes are needed for sampler-stratified train/val/test splits")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(max_episode_steps=horizon)
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("state", "action", "next_state", "goal",
                             "episode_id", "episode_seed", "timestep", "sampler")}
    contacts = 0
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            sampler = ep % 3  # one random episode, two guided episodes
            for t in range(horizon):
                state = state_from_obs(obs)
                goal = goal_from_obs(obs)
                if sampler == 0:
                    action = np.r_[rng.uniform(-1, 1, 3), 0].astype(np.float32)
                else:
                    action = heuristic_action(state, goal, noise=0.10, rng=rng)
                next_obs, _, terminated, truncated, _ = env.step(action)
                next_state = state_from_obs(next_obs)
                contacts += int(np.linalg.norm(state[:3] - state[3:6]) < 0.065)
                for key, value in (("state", state), ("action", action),
                                   ("next_state", next_state), ("goal", goal),
                                   ("episode_id", ep), ("episode_seed", seed + ep),
                                   ("timestep", t), ("sampler", sampler)):
                    rows[key].append(value)
                obs = next_obs
                if terminated or truncated:
                    break
    finally:
        env.close()
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    # Keep whole episodes together and represent both samplers in every split.
    split_rng = np.random.default_rng(seed)
    split = np.full(len(arrays["episode_id"]), 0, dtype=np.int8)
    for sampler in (0, 1):
        episode_ids = np.arange(episodes)
        episode_ids = split_rng.permutation(episode_ids[(episode_ids % 3 == 0) == (sampler == 0)])
        n_val = max(1, round(len(episode_ids) * 0.2))
        n_test = max(1, round(len(episode_ids) * 0.2))
        split[np.isin(arrays["episode_id"], episode_ids[:n_val])] = 1
        split[np.isin(arrays["episode_id"], episode_ids[n_val:n_val+n_test])] = 2
    arrays["split"] = split
    np.savez_compressed(path, **arrays)
    return {"transitions": len(split), "episodes": episodes,
            "train": int(sum(split == 0)), "validation": int(sum(split == 1)),
            "test": int(sum(split == 2)), "near_object_fraction": contacts / len(split)}


def load(path):
    with np.load(path) as stored:
        data = {key: stored[key] for key in stored.files}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not set(data["episode_id"][data["split"] == a]) & set(
            data["episode_id"][data["split"] == b])
    return data
