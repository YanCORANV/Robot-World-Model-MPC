"""FetchPush observation contract and compact physical state."""

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

# The original 25-D observation includes redundant relative positions and locked
# finger states. Velocities in Fetch observations are scaled by the 0.04 s step.
STATE_NAMES = (
    "gripper_x", "gripper_y", "gripper_z",
    "object_x", "object_y", "object_z",
    "object_dx", "object_dy", "object_dz",
    "gripper_dx", "gripper_dy", "gripper_dz",
)
STATE_DIM = len(STATE_NAMES)
ACTION_DIM = 4
OBJECT_SLICE = slice(3, 6)


def make_env(render_mode=None, max_episode_steps=50):
    return gym.make("FetchPush-v4", render_mode=render_mode,
                    max_episode_steps=max_episode_steps)


def state_from_obs(obs):
    raw = np.asarray(obs["observation"], dtype=np.float32)
    if raw.shape != (25,):
        raise ValueError(f"Expected 25-D FetchPush observation, got {raw.shape}")
    grip_step_velocity = raw[20:23]
    object_step_velocity = raw[14:17] + grip_step_velocity
    return np.concatenate((raw[0:6], object_step_velocity,
                           grip_step_velocity)).astype(np.float32)


def goal_from_obs(obs):
    return np.asarray(obs["desired_goal"], dtype=np.float32).copy()


def distance(state, goal):
    return float(np.linalg.norm(np.asarray(state)[OBJECT_SLICE] - goal))
