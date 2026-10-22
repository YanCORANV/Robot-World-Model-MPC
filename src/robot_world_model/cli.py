"""Small reproducible commands for environment, data, model, and control."""

import argparse
import json
from pathlib import Path
import time
import numpy as np

from .env import make_env, state_from_obs, goal_from_obs, distance, STATE_NAMES
from .policies import legacy_heuristic_action as heuristic_action
from .data import collect, load
from .model import train, load_model
from .prediction import evaluate_prediction
from .mpc import CEMMPC


def emit(value):
    print(json.dumps(value, indent=2))


def save_metrics(path, value):
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2) + "\n")


def probe(gif_path=None):
    env = make_env(render_mode="rgb_array" if gif_path else None)
    try:
        obs, info = env.reset(seed=11)
        initial = state_from_obs(obs)
        action = heuristic_action(initial, goal_from_obs(obs))
        frames = []
        if gif_path:
            frames.append(env.render())
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        first_reward = float(reward)
        first_terminated = bool(terminated)
        first_truncated = bool(truncated)
        first_success = bool(step_info.get("is_success", False))
        first_distance = distance(state_from_obs(next_obs), goal_from_obs(next_obs))
        if gif_path:
            frames.append(env.render())
            for _ in range(9):
                next_obs, reward, terminated, truncated, step_info = env.step(
                    heuristic_action(state_from_obs(next_obs), goal_from_obs(next_obs)))
                frames.append(env.render())
            import imageio.v2 as imageio
            Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(gif_path, frames, fps=10)
        return {"env": env.spec.id, "observation_keys": list(obs),
                "raw_observation_shape": list(obs["observation"].shape),
                "achieved_goal_shape": list(obs["achieved_goal"].shape),
                "desired_goal_shape": list(obs["desired_goal"].shape),
                "action_space": str(env.action_space), "state_names": STATE_NAMES,
                "first_action": action.tolist(), "first_reward": first_reward,
                "terminated": first_terminated, "truncated": first_truncated,
                "is_success": first_success,
                "next_distance_m": first_distance,
                "gif": gif_path}
    finally:
        env.close()


def evaluate(model_path, seeds, steps=50, cem_candidates=64,
             cem_iterations=3, cem_horizon=8, gif_path=None):
    model = load_model(model_path)
    output = {}
    for policy in ("random", "heuristic", "learned_mpc"):
        episodes = []
        for seed in seeds:
            capture = policy == "learned_mpc" and seed == seeds[0] and gif_path
            env = make_env(render_mode="rgb_array" if capture else None,
                           max_episode_steps=steps)
            rng = np.random.default_rng(seed + 10000)
            controller = CEMMPC(model, candidates=cem_candidates,
                                iterations=cem_iterations, horizon=cem_horizon,
                                seed=seed) if policy == "learned_mpc" else None
            times = []
            frames = []
            try:
                obs, _ = env.reset(seed=seed)
                initial_distance = distance(state_from_obs(obs), goal_from_obs(obs))
                executed_steps = 0
                near_object_steps = 0
                object_path_length = 0.0
                if capture:
                    frames.append(env.render())
                for _ in range(steps):
                    state = state_from_obs(obs)
                    goal = goal_from_obs(obs)
                    near_object_steps += int(np.linalg.norm(state[:3] - state[3:6]) < 0.065)
                    if policy == "random":
                        action = np.r_[rng.uniform(-1, 1, 3), 0].astype(np.float32)
                    elif policy == "heuristic":
                        action = heuristic_action(state, goal)
                    else:
                        action, runtime = controller.act(state, goal)
                        times.append(runtime)
                    obs, reward, terminated, truncated, info = env.step(action)
                    object_path_length += float(np.linalg.norm(state[3:6] - state_from_obs(obs)[3:6]))
                    executed_steps += 1
                    if capture:
                        frames.append(env.render())
                    if terminated or truncated:
                        break
                episodes.append({"seed": seed, "steps": executed_steps,
                                 "success": bool(info.get("is_success", False)),
                                 "initial_distance_m": initial_distance,
                                 "final_distance_m": distance(state_from_obs(obs), goal_from_obs(obs)),
                                 "near_object_fraction": near_object_steps / executed_steps,
                                 "object_path_length_m": object_path_length,
                                 "planning_runtime_ms_per_step": float(np.mean(times)*1000) if times else 0.0})
                if capture:
                    import imageio.v2 as imageio
                    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimsave(gif_path, frames[::2], fps=10)
            finally:
                env.close()
        output[policy] = {"success_rate": float(np.mean([r["success"] for r in episodes])),
                          "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in episodes])),
                          "mean_near_object_fraction": float(np.mean([r["near_object_fraction"] for r in episodes])),
                          "mean_planning_runtime_ms_per_step": float(np.mean([
                              r["planning_runtime_ms_per_step"] for r in episodes])),
                          "episodes": episodes}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--gif")
    p = sub.add_parser("collect")
    p.add_argument("--out", default="data/small.npz")
    p.add_argument("--episodes", type=int, default=18)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=50)
    p = sub.add_parser("train")
    p.add_argument("--data", default="data/small.npz")
    p.add_argument("--out", default="results/dynamics.pt")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--metrics", default="results/train.json")
    p = sub.add_parser("predict")
    p.add_argument("--data", default="data/small.npz")
    p.add_argument("--model", default="results/dynamics.pt")
    p.add_argument("--plot", default="results/prediction.png")
    p.add_argument("--metrics", default="results/prediction.json")
    p = sub.add_parser("evaluate")
    p.add_argument("--model", default="results/dynamics.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--gif")
    p.add_argument("--metrics", default="results/evaluation.json")
    args = parser.parse_args()
    if args.command == "probe":
        emit(probe(args.gif))
    elif args.command == "collect":
        emit(collect(args.out, args.episodes, args.seed, args.steps))
    elif args.command == "train":
        result = train(load(args.data), args.out, args.epochs, seed=args.seed)
        save_metrics(args.metrics, result)
        result.pop("history")
        emit(result)
    elif args.command == "predict":
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        result = evaluate_prediction(load_model(args.model), load(args.data), plot_path=args.plot)
        save_metrics(args.metrics, result)
        emit(result)
    elif args.command == "evaluate":
        result = evaluate(args.model, args.seeds, args.steps, args.candidates,
                          args.iterations, args.horizon, args.gif)
        save_metrics(args.metrics, result)
        emit(result)


if __name__ == "__main__":
    main()
