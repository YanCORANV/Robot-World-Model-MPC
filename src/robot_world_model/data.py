"""Episode-isolated transition collection and loading."""

from pathlib import Path
import numpy as np
import mujoco
from .env import make_env, state_from_obs, goal_from_obs
from .policies import heuristic_action, legacy_heuristic_action


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
                    action = legacy_heuristic_action(state, goal, noise=0.10, rng=rng)
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


def robot_object_contact(env):
    """True when MuJoCo's final substep has a gripper/object contact."""
    model, sim_data = env.unwrapped.model, env.unwrapped.data
    for i in range(sim_data.ncon):
        pair = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, sim_data.contact[i].geom1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, sim_data.contact[i].geom2))
        if "object0" in pair and any(name and name.startswith("robot0:") and
                                     "gripper" in name for name in pair):
            return True
    return False


def _near_push_action(state, goal, rng):
    """Reach the block, then explore short noisy motions at pushing height."""
    separation = np.linalg.norm(state[:3] - state[3:6])
    if separation > 0.075 or abs(state[2] - state[5]) > 0.04:
        return legacy_heuristic_action(state, goal, noise=0.05, rng=rng)
    toward = np.asarray(goal[:2]) - state[3:5]
    toward /= max(np.linalg.norm(toward), 1e-6)
    if rng.random() < 0.25:
        toward = -toward
    action = np.zeros(4, dtype=np.float32)
    action[:2] = np.clip(0.75 * toward + rng.normal(0, 0.25, 2), -1, 1)
    action[2] = np.clip((state[5] + 0.015 - state[2]) / 0.05, -1, 1)
    return action


def collect_rich(path, episodes=120, seed=5000, horizon=50):
    """Four episode-level samplers; all splits contain every sampler."""
    if episodes < 20 or episodes % 4:
        raise ValueError("episodes must be a multiple of four and at least 20")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(max_episode_steps=horizon)
    rng = np.random.default_rng(seed)
    keys = ("state", "action", "next_state", "goal", "episode_id",
            "episode_seed", "timestep", "sampler", "contact", "pushing", "success")
    rows = {key: [] for key in keys}
    episode_ever_success = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            mode = ep % 4  # random, goal-directed, near-contact, aggressive
            ever_success = False
            for t in range(horizon):
                state, goal = state_from_obs(obs), goal_from_obs(obs)
                if mode == 0:
                    action = np.r_[rng.uniform(-1, 1, 3), 0].astype(np.float32)
                elif mode == 1:
                    action = heuristic_action(state, goal, noise=0.03, rng=rng)
                elif mode == 2:
                    action = _near_push_action(state, goal, rng)
                else:
                    action = legacy_heuristic_action(state, goal, noise=0.10, rng=rng)
                next_obs, _, terminated, truncated, info = env.step(action)
                next_state = state_from_obs(next_obs)
                contact = robot_object_contact(env)
                moving = np.linalg.norm(next_state[3:5] - state[3:5]) > 0.002
                separation = min(np.linalg.norm(state[:3] - state[3:6]),
                                 np.linalg.norm(next_state[:3] - next_state[3:6]))
                # Contact is sampled only at the last MuJoCo substep. Object
                # motion near the gripper also catches a push earlier in the step.
                pushing = bool(moving and (contact or separation < 0.10))
                success = bool(info.get("is_success", False))
                ever_success |= success
                for key, value in (("state", state), ("action", action),
                                   ("next_state", next_state), ("goal", goal),
                                   ("episode_id", ep), ("episode_seed", seed + ep),
                                   ("timestep", t), ("sampler", mode),
                                   ("contact", contact), ("pushing", pushing),
                                   ("success", success)):
                    rows[key].append(value)
                obs = next_obs
                if terminated or truncated or (mode == 1 and success):
                    break
            episode_ever_success.append(ever_success)
    finally:
        env.close()
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    split = np.zeros(len(arrays["episode_id"]), dtype=np.int8)
    split_rng = np.random.default_rng(seed)
    for mode in range(4):
        ids = split_rng.permutation(np.arange(mode, episodes, 4))
        n = len(ids)
        split[np.isin(arrays["episode_id"], ids[:n//5])] = 1
        split[np.isin(arrays["episode_id"], ids[n//5:2*(n//5)])] = 2
    arrays["split"] = split
    np.savez_compressed(path, **arrays)
    return {"episodes": episodes, "transitions": len(split),
            "split_counts": {name: int(np.sum(split == i)) for i, name in enumerate(
                ("train", "validation", "test"))},
            "sampler_counts": {name: int(np.sum(arrays["sampler"] == i)) for i, name in
                               enumerate(("random", "goal_directed", "near_contact", "aggressive"))},
            "contact_steps": int(arrays["contact"].sum()),
            "pushing_steps": int(arrays["pushing"].sum()),
            "episodes_ever_success": int(sum(episode_ever_success))}
