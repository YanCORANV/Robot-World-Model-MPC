"""Reproducible phase-2 data, prediction, and matched-seed policy evaluation."""

import argparse
import json
from pathlib import Path
import numpy as np

from .data import collect_rich, load, robot_object_contact
from .diagnostics import prediction_report, plot_regime_comparison, plot_trajectories
from .env import make_env, state_from_obs, goal_from_obs, distance
from .improved_mpc import ContactAwareMPC
from .model import load_model, train
from .mpc import CEMMPC
from .policies import heuristic_action


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def evaluate_policies(old_model, new_model, seeds, steps=50,
                      trajectory_plot=None, gif_path=None):
    policies = ("random", "heuristic", "original_mpc", "improved_mpc")
    result = {"seeds": list(seeds), "steps_per_episode": steps, "policies": {}}
    trajectories = {}
    for policy in policies:
        episodes = []
        for seed in seeds:
            capture = policy == "improved_mpc" and seed == seeds[0] and gif_path
            env = make_env(render_mode="rgb_array" if capture else None,
                           max_episode_steps=steps)
            rng = np.random.default_rng(seed + 10000)
            controller = (CEMMPC(old_model, seed=seed) if policy == "original_mpc"
                else ContactAwareMPC(new_model, seed=seed) if policy == "improved_mpc"
                else None)
            frames, plan_times, push_times = [], [], []
            near_count = contact_count = push_count = 0
            phases = {}
            try:
                obs, _ = env.reset(seed=seed)
                initial = distance(state_from_obs(obs), goal_from_obs(obs))
                initial_success = initial < 0.05
                ever_success = initial_success
                track = {"seed": seed, "goal_xy": goal_from_obs(obs)[:2].tolist(),
                         "object_xy": [state_from_obs(obs)[3:5].tolist()],
                         "gripper_xy": [state_from_obs(obs)[:2].tolist()]}
                if capture:
                    frames.append(env.render())
                for t in range(steps):
                    state, goal = state_from_obs(obs), goal_from_obs(obs)
                    separation = np.linalg.norm(state[:3] - state[3:6])
                    near_count += int(separation < 0.065)
                    if policy == "random":
                        action = np.r_[rng.uniform(-1, 1, 3), 0].astype(np.float32)
                    elif policy == "heuristic":
                        action = heuristic_action(state, goal)
                    else:
                        action, elapsed = controller.act(state, goal)
                        plan_times.append(elapsed)
                        phase = getattr(controller, "last_phase", "model_plan")
                        phases[phase] = phases.get(phase, 0) + 1
                        if phase == "model_push":
                            push_times.append(elapsed)
                    obs, _, terminated, truncated, info = env.step(action)
                    next_state = state_from_obs(obs)
                    contact_count += int(robot_object_contact(env))
                    motion = np.linalg.norm(next_state[3:5] - state[3:5]) > 0.002
                    push_count += int(motion and separation < 0.10)
                    ever_success |= bool(info.get("is_success", False))
                    if seed == seeds[0]:
                        track["object_xy"].append(next_state[3:5].tolist())
                        track["gripper_xy"].append(next_state[:2].tolist())
                    if capture:
                        frames.append(env.render())
                    if terminated or truncated:
                        break
                count = t + 1
                episodes.append({"seed": seed, "initial_distance_m": initial,
                    "initial_success": initial_success,
                    "final_distance_m": distance(state_from_obs(obs), goal_from_obs(obs)),
                    "final_success": bool(info.get("is_success", False)),
                    "ever_success": bool(ever_success),
                    "near_contact_fraction": near_count / count,
                    "mujoco_end_step_contact_fraction": contact_count / count,
                    "moving_push_fraction": push_count / count,
                    "planning_runtime_ms_per_step": float(np.mean(plan_times) * 1000) if plan_times else 0.0,
                    "model_push_runtime_ms": float(np.mean(push_times) * 1000) if push_times else 0.0,
                    "phase_counts": phases})
                if seed == seeds[0]:
                    trajectories[policy] = track
                if capture:
                    import imageio.v2 as imageio
                    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimsave(gif_path, frames[::2], fps=10)
            finally:
                env.close()
        initially_unsolved = [ep for ep in episodes if not ep["initial_success"]]
        result["policies"][policy] = {
            "final_success_rate": float(np.mean([ep["final_success"] for ep in episodes])),
            "new_success_rate_excluding_initial_success": float(np.mean([
                ep["final_success"] for ep in initially_unsolved])) if initially_unsolved else None,
            "ever_success_rate": float(np.mean([ep["ever_success"] for ep in episodes])),
            "initial_success_count": sum(ep["initial_success"] for ep in episodes),
            "mean_final_distance_m": float(np.mean([ep["final_distance_m"] for ep in episodes])),
            "mean_near_contact_fraction": float(np.mean([ep["near_contact_fraction"] for ep in episodes])),
            "mean_mujoco_end_step_contact_fraction": float(np.mean([
                ep["mujoco_end_step_contact_fraction"] for ep in episodes])),
            "mean_moving_push_fraction": float(np.mean([ep["moving_push_fraction"] for ep in episodes])),
            "mean_planning_runtime_ms_per_step": float(np.mean([
                ep["planning_runtime_ms_per_step"] for ep in episodes])),
            "mean_model_push_runtime_ms": (
                float(sum(ep["model_push_runtime_ms"] * ep["phase_counts"].get("model_push", 0)
                          for ep in episodes) /
                      sum(ep["phase_counts"].get("model_push", 0) for ep in episodes))
                if any(ep["phase_counts"].get("model_push", 0) for ep in episodes) else 0.0),
            "episodes": episodes}
    if trajectory_plot:
        Path(trajectory_plot).parent.mkdir(parents=True, exist_ok=True)
        plot_trajectories(trajectories, trajectory_plot)
    return result


def ablate_controllers(old_model, new_model, seeds, steps=50):
    """Developer-seed attribution: swap model and controller independently."""
    combinations = (("old_controller_old_model", CEMMPC, old_model),
                    ("old_controller_rich_model", CEMMPC, new_model),
                    ("new_controller_old_model", ContactAwareMPC, old_model),
                    ("new_controller_rich_model", ContactAwareMPC, new_model))
    result = {}
    for name, controller_class, model in combinations:
        rows = []
        for seed in seeds:
            env = make_env(max_episode_steps=steps)
            try:
                obs, _ = env.reset(seed=seed)
                controller = controller_class(model, seed=seed)
                for _ in range(steps):
                    action, _ = controller.act(state_from_obs(obs), goal_from_obs(obs))
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                rows.append({"seed": seed, "success": bool(info.get("is_success", False)),
                             "distance_m": distance(state_from_obs(obs), goal_from_obs(obs))})
            finally:
                env.close()
        result[name] = {"successes": sum(row["success"] for row in rows),
                        "episodes": len(rows),
                        "mean_distance_m": float(np.mean([row["distance_m"] for row in rows])),
                        "per_seed": rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--out", default="data/rich.npz")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--seed", type=int, default=5000)
    p = sub.add_parser("train")
    p.add_argument("--data", default="data/rich.npz")
    p.add_argument("--out", default="results/rich.pt")
    p.add_argument("--metrics", default="results/phase2_train.json")
    p.add_argument("--epochs", type=int, default=35)
    p = sub.add_parser("predict")
    p.add_argument("--old-data", default="data/small.npz")
    p.add_argument("--new-data", default="data/rich.npz")
    p.add_argument("--old-model", default="results/dynamics.pt")
    p.add_argument("--new-model", default="results/rich.pt")
    p.add_argument("--out", default="results/phase2_prediction.json")
    p.add_argument("--plot", default="results/phase2_regimes.png")
    p = sub.add_parser("evaluate")
    p.add_argument("--old-model", default="results/dynamics.pt")
    p.add_argument("--new-model", default="results/rich.pt")
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--out", default="results/phase2_evaluation.json")
    p.add_argument("--plot", default="results/phase2_trajectories.png")
    p.add_argument("--gif", default="results/phase2_mpc.gif")
    p = sub.add_parser("ablate")
    p.add_argument("--old-model", default="results/dynamics.pt")
    p.add_argument("--new-model", default="results/rich.pt")
    p.add_argument("--seed-start", type=int, default=200)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--out", default="results/phase2_dev_ablation.json")
    args = parser.parse_args()
    if args.command == "collect":
        result = collect_rich(args.out, args.episodes, args.seed)
    elif args.command == "train":
        result = train(load(args.data), args.out, epochs=args.epochs, seed=7)
        save_json(args.metrics, result)
        result = {k: v for k, v in result.items() if k != "history"}
    elif args.command == "predict":
        old, new = load_model(args.old_model), load_model(args.new_model)
        old_data, new_data = load(args.old_data), load(args.new_data)
        result = {"original_on_original_test": prediction_report(old, old_data),
                  "original_on_rich_test": prediction_report(old, new_data),
                  "rich_on_rich_test": prediction_report(new, new_data),
                  "rich_on_original_test": prediction_report(new, old_data)}
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot_regime_comparison({"Original": result["original_on_rich_test"],
                                "Rich data": result["rich_on_rich_test"]}, args.plot)
        save_json(args.out, result)
    elif args.command == "evaluate":
        old, new = load_model(args.old_model), load_model(args.new_model)
        result = evaluate_policies(old, new, list(range(args.seed_start,
                                   args.seed_start + args.count)),
                                   trajectory_plot=args.plot, gif_path=args.gif)
        save_json(args.out, result)
    else:
        old, new = load_model(args.old_model), load_model(args.new_model)
        result = ablate_controllers(old, new, list(range(args.seed_start,
                                    args.seed_start + args.count)))
        save_json(args.out, result)
    if args.command == "evaluate":
        print(json.dumps({name: {k:v for k,v in values.items() if k != "episodes"}
                          for name,values in result["policies"].items()}, indent=2))
    elif args.command == "predict":
        print(json.dumps({name: {"overall": values["overall"],
                                 "multi_step": values["multi_step"]["model_object_rmse_m"]}
                          for name,values in result.items()}, indent=2))
    elif args.command == "ablate":
        print(json.dumps({name: {k:v for k,v in values.items() if k != "per_seed"}
                          for name,values in result.items()}, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
