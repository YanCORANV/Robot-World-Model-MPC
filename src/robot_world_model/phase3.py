"""Matched model ablation; final seed range is frozen in scripts/phase3.sh."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import load, robot_object_contact
from .diagnostics import prediction_report
from .env import distance, goal_from_obs, make_env, state_from_obs
from .improved_mpc import ContactAwareMPC, FixedPushController, ZeroChangeDynamics
from .model import load_model
from .policies import heuristic_action

POLICIES = ("heuristic", "fixed_push", "zero_change_mpc", "original_model_mpc", "rich_model_mpc")
PLANNER = {"horizon": 10, "candidates": 64, "iterations": 3}
FINAL_SEEDS = tuple(range(20000, 20100))


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [center-radius, center+radius]


def paired_stats(a, b, seed=104):
    """A minus B, with paired bootstrap CI and exact two-sided McNemar p."""
    assert [r["seed"] for r in a] == [r["seed"] for r in b]
    sa = np.asarray([r["final_success"] for r in a], dtype=np.float64)
    sb = np.asarray([r["final_success"] for r in b], dtype=np.float64)
    da = np.asarray([r["final_distance_m"] for r in a])
    db = np.asarray([r["final_distance_m"] for r in b])
    a_only = int(np.sum((sa == 1) & (sb == 0)))
    b_only = int(np.sum((sa == 0) & (sb == 1)))
    discordant = a_only + b_only
    p = min(1.0, 2*sum(math.comb(discordant, k) for k in range(min(a_only,b_only)+1)) / 2**discordant) if discordant else 1.0
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(a), size=(10000, len(a)))
    success_diff = (sa-sb)[ix].mean(1)
    distance_diff = (da-db)[ix].mean(1)
    return {"n": len(a), "a_only_success": a_only, "b_only_success": b_only,
            "success_rate_difference": float((sa-sb).mean()),
            "success_difference_bootstrap_95_ci": np.quantile(success_diff, [.025,.975]).tolist(),
            "exact_mcnemar_two_sided_p": p,
            "mean_distance_difference_m": float((da-db).mean()),
            "distance_difference_bootstrap_95_ci_m": np.quantile(distance_diff, [.025,.975]).tolist()}


def controller_for(policy, seed, old, rich):
    if policy == "fixed_push":
        return FixedPushController()
    if policy == "zero_change_mpc":
        model = ZeroChangeDynamics()
    elif policy == "original_model_mpc":
        model = old
    elif policy == "rich_model_mpc":
        model = rich
    else:
        return None
    return ContactAwareMPC(model, seed=seed, **PLANNER)


def evaluate(old, rich, seeds, steps=50):
    """Five controllers, independent identical resets, no hidden future state."""
    result = {"seeds": list(seeds), "steps_per_episode": steps,
              "planner": PLANNER, "policies": {}, "paired_comparisons": {}}
    for name in POLICIES:
        rows = []
        for seed in seeds:
            env = make_env(max_episode_steps=steps)
            controller = controller_for(name, seed, old, rich)
            try:
                obs, _ = env.reset(seed=seed)
                initial = distance(state_from_obs(obs), goal_from_obs(obs))
                near = contact = moving_push = 0
                plan_seconds = 0.0
                push_plan_seconds = 0.0
                push_steps = 0
                first_push_action = None
                first_push_state = None
                first_push_goal = None
                object_path = [state_from_obs(obs)[3:5].tolist()] if seed == seeds[0] else None
                for step in range(steps):
                    state, goal = state_from_obs(obs), goal_from_obs(obs)
                    separation = np.linalg.norm(state[:3] - state[3:6])
                    near += int(separation < .065)
                    if controller is None:
                        started = time.perf_counter()
                        action = heuristic_action(state, goal)
                        elapsed = time.perf_counter() - started
                        phase = "heuristic"
                    else:
                        action, elapsed = controller.act(state, goal)
                        phase = controller.last_phase
                    plan_seconds += elapsed
                    if phase == "model_push":
                        push_steps += 1
                        push_plan_seconds += elapsed
                        if first_push_action is None:
                            first_push_action = action[:3].tolist()
                            first_push_state = state.tolist()
                            first_push_goal = goal.tolist()
                    obs, _, terminated, truncated, info = env.step(action)
                    next_state = state_from_obs(obs)
                    contact += int(robot_object_contact(env))
                    moving_push += int(separation < .10 and np.linalg.norm(next_state[3:5]-state[3:5]) > .002)
                    if object_path is not None:
                        object_path.append(next_state[3:5].tolist())
                    if terminated or truncated:
                        break
                count = step + 1
                row = {"seed": seed, "initial_distance_m": initial,
                       "initial_success": bool(initial < .05),
                       "final_success": bool(info.get("is_success", False)),
                       "final_distance_m": distance(state_from_obs(obs), goal_from_obs(obs)),
                       "near_contact_fraction": near/count,
                       "end_step_contact_fraction": contact/count,
                       "moving_push_fraction": moving_push/count,
                       "runtime_ms_per_step": 1000*plan_seconds/count,
                       "push_runtime_ms_per_push_step": 1000*push_plan_seconds/push_steps if push_steps else 0.0,
                       "push_steps": push_steps, "first_push_action_xyz": first_push_action,
                       "first_push_state": first_push_state, "first_push_goal": first_push_goal}
                if object_path is not None:
                    row["object_path_xy"] = object_path
                    row["goal_xy"] = goal[:2].tolist()
                rows.append(row)
            finally:
                env.close()
        unsolved = [r for r in rows if not r["initial_success"]]
        successes = sum(r["final_success"] for r in unsolved)
        result["policies"][name] = {
            "overall_successes": sum(r["final_success"] for r in rows),
            "overall_success_rate": float(np.mean([r["final_success"] for r in rows])),
            "initial_successes": len(rows)-len(unsolved),
            "new_successes": successes, "initially_unsolved_episodes": len(unsolved),
            "new_success_rate": successes/len(unsolved),
            "new_success_wilson_95_ci": wilson(successes, len(unsolved)),
            "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in rows])),
            "median_final_distance_m": float(np.median([r["final_distance_m"] for r in rows])),
            "mean_near_contact_fraction": float(np.mean([r["near_contact_fraction"] for r in rows])),
            "mean_end_step_contact_fraction": float(np.mean([r["end_step_contact_fraction"] for r in rows])),
            "mean_moving_push_fraction": float(np.mean([r["moving_push_fraction"] for r in rows])),
            "mean_runtime_ms_per_step": float(np.mean([r["runtime_ms_per_step"] for r in rows])),
            "mean_push_runtime_ms": float(np.mean([r["push_runtime_ms_per_push_step"] for r in rows if r["push_steps"]])) if any(r["push_steps"] for r in rows) else 0.0,
            "episodes": rows}
        print(f"{name}: {successes}/{len(unsolved)} new successes", flush=True)
    assert all([r["initial_distance_m"] for r in result["policies"][name]["episodes"]] ==
               [r["initial_distance_m"] for r in result["policies"][POLICIES[0]]["episodes"]]
               for name in POLICIES[1:])
    rich_rows = [r for r in result["policies"]["rich_model_mpc"]["episodes"] if not r["initial_success"]]
    for name in POLICIES[:-1]:
        other = [r for r in result["policies"][name]["episodes"] if not r["initial_success"]]
        result["paired_comparisons"][f"rich_minus_{name}"] = paired_stats(rich_rows, other)
    # Staged controllers have identical observations until their first push.
    # This audit isolates how changing only the push policy/model changes action.
    reference = result["policies"]["rich_model_mpc"]["episodes"]
    audit = {}
    for name in POLICIES[1:-1]:
        other = result["policies"][name]["episodes"]
        pair = [(a,b) for a,b in zip(reference,other)
                if a["first_push_action_xyz"] is not None and b["first_push_action_xyz"] is not None]
        assert all(np.allclose(a["first_push_state"],b["first_push_state"],atol=1e-7)
                   and np.allclose(a["first_push_goal"],b["first_push_goal"],atol=1e-7)
                   for a,b in pair), "First push states differed; matched action audit invalid"
        diffs = [np.linalg.norm(np.asarray(a["first_push_action_xyz"])-b["first_push_action_xyz"])
                 for a,b in pair]
        audit[name] = {"matched_first_push_states": len(pair),
                       "mean_l2_action_difference_from_rich": float(np.mean(diffs)) if pair else None,
                       "max_l2_action_difference_from_rich": float(np.max(diffs)) if pair else None}
    result["first_push_action_audit"] = audit
    return result


def predict(old, rich, data):
    reports = {}
    for name, model in (("zero_change", ZeroChangeDynamics()),
                        ("original_model", old), ("rich_model", rich)):
        reports[name] = prediction_report(model, data)
    indices = np.flatnonzero(data["split"] == 2)
    contact = data["contact"][indices].astype(bool)
    state = torch.from_numpy(data["state"][indices].astype(np.float32))
    action = torch.from_numpy(data["action"][indices].astype(np.float32))
    actual = data["next_state"][indices, 3:6]
    for name, model in (("zero_change", ZeroChangeDynamics()),
                        ("original_model", old), ("rich_model", rich)):
        with torch.no_grad():
            pred = model(state, action).numpy()[:, 3:6]
        regimes = {}
        for label, mask in (("end_step_contact", contact), ("no_end_step_contact", ~contact)):
            error = pred[mask] - actual[mask]
            regimes[label] = {"count": int(mask.sum()),
                              "one_step_object_rmse_m": float(np.sqrt(np.mean(error**2))) if mask.any() else None,
                              "predicted_xy_motion_m": float(np.linalg.norm(pred[mask,:2]-state.numpy()[mask,3:5],axis=1).mean()) if mask.any() else None,
                              "actual_xy_motion_m": float(np.linalg.norm(actual[mask,:2]-state.numpy()[mask,3:5],axis=1).mean()) if mask.any() else None}
        reports[name]["contact_conditioned"] = regimes
    return {"dataset": "data/rich.npz, split=2, held-out whole episodes",
            "notes": "Pushing means >2 mm object motion and <10 cm gripper separation; contact is MuJoCo collision at final internal substep.",
            "models": reports}


def plot_prediction(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = result["models"]
    names = list(models)
    labels = ["Zero change", "Original MLP", "Rich-data MLP"]
    regimes = ["free_space", "near_contact_static", "actual_pushing", "end_step_contact"]
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    x = np.arange(len(regimes))
    for i, (name,label) in enumerate(zip(names,labels)):
        r = models[name]
        values = [(r["contact_conditioned"][regime] if regime == "end_step_contact" else r["regimes"][regime]) for regime in regimes]
        errors = [(v["one_step_object_rmse_m"] if regime == "end_step_contact" else v["model_object_rmse_m"])*1000 for regime,v in zip(regimes, values)]
        axes[0].bar(x+(i-1)*.25, errors, .24, label=label)
        push = r["regimes"]["actual_pushing"]
        axes[1].scatter(push["actual_object_xy_motion_m"]*1000,
                        push["predicted_object_xy_motion_m"]*1000,
                        s=80, label=label)
    axes[0].set(xticks=x, xticklabels=["Free", "Near static", "Pushing", "End contact"], ylabel="1-step object RMSE (mm)")
    axes[1].plot([0,25],[0,25],"k--",lw=1)
    axes[1].set(xlabel="Actual mean XY motion in pushing (mm)",ylabel="Predicted mean XY motion (mm)",xlim=(0,25),ylim=(0,25))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def plot_object_motion(old, rich, data, path):
    """Per-transition motion scatter, not only regime-average calibration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ix = np.flatnonzero(data["split"] == 2)
    state = data["state"][ix].astype(np.float32)
    action = data["action"][ix].astype(np.float32)
    true = np.linalg.norm(data["next_state"][ix, 3:5]-state[:, 3:5], axis=1)*1000
    pushing = data["pushing"][ix].astype(bool)
    near = np.minimum(np.linalg.norm(state[:, :3]-state[:, 3:6],axis=1),
                      np.linalg.norm(data["next_state"][ix, :3]-data["next_state"][ix, 3:6],axis=1)) < .10
    regimes = (("Free space", ~near & ~pushing, "#4477aa"),
               ("Near static", near & ~pushing, "#ddaa33"),
               ("Active push", pushing, "#cc6677"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (name, model) in zip(axes, (("Zero change",ZeroChangeDynamics()),
                                  ("Original MLP",old),("Rich-data MLP",rich))):
        with torch.no_grad():
            pred = model(torch.from_numpy(state),torch.from_numpy(action)).numpy()
        motion = np.linalg.norm(pred[:, 3:5]-state[:, 3:5],axis=1)*1000
        for label, mask, color in regimes:
            ax.scatter(true[mask],motion[mask],s=9,alpha=.35,c=color,label=f"{label} ({mask.sum()})")
        ax.plot([0,80],[0,80],"k--",lw=1)
        ax.set(title=name,xlabel="Actual XY motion (mm)",xlim=(0,80),ylim=(0,80))
    axes[0].set_ylabel("Predicted XY motion (mm)")
    axes[-1].legend(fontsize=8,loc="upper left")
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def plot_control(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(POLICIES)
    labels = ["Heuristic", "Fixed push", "Zero CEM", "Original CEM", "Rich CEM"]
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    p = result["policies"]
    x = np.arange(len(names))
    rates = [p[n]["new_success_rate"] for n in names]
    lower = [rates[i]-p[n]["new_success_wilson_95_ci"][0] for i,n in enumerate(names)]
    upper = [p[n]["new_success_wilson_95_ci"][1]-rates[i] for i,n in enumerate(names)]
    axes[0].bar(x, rates, color="#4477aa")
    axes[0].errorbar(x,rates,yerr=[lower,upper],fmt="none",color="black",capsize=3)
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,1.08),ylabel="Final success, initially unsolved")
    axes[1].bar(x,[p[n]["mean_final_distance_m"] for n in names],color="#cc6677")
    axes[1].set(xticks=x,xticklabels=labels,ylabel="Mean final object–goal distance (m)")
    for ax in axes: ax.tick_params(axis="x",rotation=28)
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("predict","dev","final"))
    parser.add_argument("--old-model",default="results/dynamics.pt")
    parser.add_argument("--rich-model",default="results/rich.pt")
    parser.add_argument("--data",default="data/rich.npz")
    args = parser.parse_args()
    old, rich = load_model(args.old_model), load_model(args.rich_model)
    if args.command == "predict":
        data = load(args.data)
        report = predict(old, rich, data)
        out = Path("results/phase3_prediction.json")
        out.write_text(json.dumps(report,indent=2)+"\n")
        plot_prediction(report,"results/phase3_prediction.png")
        plot_object_motion(old,rich,data,"results/phase3_motion_scatter.png")
        print(out)
    else:
        seeds = list(range(300,310)) if args.command == "dev" else list(FINAL_SEEDS)
        result = evaluate(old,rich,seeds)
        out = Path(f"results/phase3_{args.command}_evaluation.json")
        out.write_text(json.dumps(result,indent=2)+"\n")
        if args.command == "final":
            plot_control(result,"results/phase3_controllers.png")
        print(out)


if __name__ == "__main__":
    main()
