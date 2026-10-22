# Learning Robot Dynamics for Model-Based Manipulation

A small, CPU-first research implementation of a **state-based** world model for `FetchPush-v4`. A neural network learns one-step robot and object dynamics from MuJoCo trajectories. Cross-Entropy Method (CEM) model predictive control (MPC) searches action sequences under that learned model and replans after every real simulator step. This is a simulation study; no physical robot or sim-to-real result is claimed.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
rwm probe --gif results/probe.gif
rwm collect --episodes 18 --seed 7 --steps 50
rwm train --epochs 35 --seed 7
rwm predict
rwm evaluate --seeds 100 101 102 --gif results/mpc.gif
pytest -q
```

`rwm probe` checks reset, one step, reward, spaces, and optional RGB rendering. If graphics are unavailable, omit `--gif`; physics, training, and planning can still run headlessly. The small reference dataset, checkpoint, plot, metrics, and MPC GIF are committed; new smoke outputs stay local.

For a faster end-to-end check, run `bash scripts/smoke.sh`. It uses 12 episodes, 18 training epochs, two evaluation seeds, and a smaller CEM search.

## Observation and action contract

The environment returns a dictionary: `observation` (25 floats), `achieved_goal` (current object XYZ), and `desired_goal` (target object XYZ). The action is `[gripper_dx, gripper_dy, gripper_dz, gripper_command]`, each in `[-1,1]`. The gripper is locked for FetchPush, so the fourth command is always zero here. Sparse reward is `0` inside 5 cm of the goal and `-1` otherwise. One environment step runs 20 MuJoCo substeps (0.04 s total).

| Raw `observation` indices | Meaning |
| --- | --- |
| 0–2 | Gripper Cartesian XYZ position in world coordinates (m) |
| 3–5 | Object Cartesian XYZ position in world coordinates (m) |
| 6–8 | Object position relative to gripper (m) |
| 9–10 | Right and left gripper finger joint displacements (m) |
| 11–13 | Object XYZ Euler orientation (rad) |
| 14–16 | Object linear velocity relative to gripper, scaled by step duration |
| 17–19 | Object angular velocity, scaled by step duration |
| 20–22 | Gripper linear velocity, scaled by step duration |
| 23–24 | Finger velocities, scaled by step duration |

The model uses a 12-D state: gripper XYZ, object XYZ, object XYZ *per-step displacement*, and gripper XYZ *per-step displacement*. Absolute object displacement is raw indices `14:17 + 20:23`. This drops redundant relative positions and the locked finger state. The goal is stored with transitions and enters the controller cost; it is never a model input. The 12-D observation omits some joint and contact details, so its dynamics need not be perfectly Markovian.

## Learning and planning

`data.py` records `(state, action, next_state, goal, episode_id, episode_seed, timestep, sampler)` from real FetchPush rollouts. One third of episodes use uniform random Cartesian actions; the rest use a noisy push heuristic that approaches behind the block. Whole episodes are assigned to train, validation, or test, preventing adjacent transitions from crossing splits.

`model.py` learns a residual: `s_hat[t+1] = s[t] + MLP(normalize([s[t], a[t]])) * delta_std + delta_mean`. All normalization statistics are fit on training episodes only. The checkpoint stores weights and statistics together; validation loss selects the best epoch. `prediction.py` compares held-out one-step RMSE with a zero-change predictor and measures open-loop object position RMSE after five real recorded actions, with no state correction between predictions.

`mpc.py` samples bounded action sequences, rolls them forward **only through the learned MLP**, scores predicted object-to-goal distance plus hand designed gripper placement and action costs, and updates the CEM sampling distribution using its best candidates. It executes only the first chosen action in MuJoCo, observes the result, and replans. The planner never queries hidden simulator futures. `rwm evaluate` runs random, heuristic, and learned MPC policies from identical test reset seeds, reporting final success, final distance, and planning time. Model bias and compounding error can make learned MPC worse than the heuristic; retain that result as evidence.

## Results

Measured on an Apple M1 Pro CPU with Python 3.12, PyTorch 2.14.1, MuJoCo 3.4.0, and Gymnasium-Robotics 1.4.2. The reference run collected 18 episodes × 50 steps = 900 transitions (600 train / 150 validation / 150 test), seeded at 7. About 30.8% of transitions had the gripper within 6.5 cm of the object. The best checkpoint was epoch 30 of 35, with normalized validation MSE 0.2868.

| Held-out prediction | Error |
| --- | ---: |
| Learned model, one-step object position RMSE | 0.00469 m |
| Zero-change baseline, one-step object position RMSE | 0.01273 m |
| Learned model, five-step open-loop object position RMSE | 0.03162 m |

Closed-loop policy comparison used identical reset seeds 100, 101, 102 and 50 steps per episode. These three episodes are a smoke test, not a reliable success-rate estimate.

| Policy | Final success | Mean final object-goal distance | Time to plan each step | Fraction of steps within 6.5 cm of object |
| --- | ---: | ---: | ---: | ---: |
| Random | 0/3 | 0.1783 m | — | 0.000 |
| Push heuristic | 0/3 | 0.1063 m | — | 0.653 |
| Learned dynamics + CEM-MPC | 0/3 | 0.1802 m | 2.39 ms | 0.073 |

The learned MPC executed real actions but rarely brought the gripper close enough to the object. Its short model horizon and imperfect contact predictions likely let the optimizer favor actions that look useful in the model but do not produce physical contact. The heuristic contacted and moved the object much more often, yet also failed to reach the 5 cm success threshold; its repeated approach/push rule can overshoot or oscillate. The initial 12-episode smoke model was worse still: one-step object RMSE 0.0251 m versus 0.0148 m for zero change, and five-step RMSE 0.1974 m. Its held-out episodes differ from the 18-episode run, so this is a warning about small-sample instability rather than a controlled learning-curve comparison.

Artifacts: [sample dataset](data/small.npz), [model checkpoint](results/dynamics.pt), [training history](results/train.json), [prediction metrics](results/prediction.json), [ground-truth comparison plot](results/prediction.png), [policy evaluation](results/evaluation.json), and [MPC rollout GIF](results/mpc.gif). These are genuine results from the commands above, including failures.

![Held-out predicted and ground-truth object positions](results/prediction.png)

![Learned MPC rollout on FetchPush-v4](results/mpc.gif)

## Source and license

Original project code is MIT licensed; see [LICENSE](LICENSE). The environment is [Gymnasium-Robotics FetchPush](https://robotics.farama.org/envs/fetch/push/) (MIT), based on [Plappert et al., 2018](https://arxiv.org/abs/1802.09464). Physics uses [MuJoCo](https://mujoco.readthedocs.io/) (Apache 2.0). PyTorch supplies the MLP. These dependencies are installed separately and are not copied into this repository.
