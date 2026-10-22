"""FetchSlide observation contract. Velocities are per 0.04 s control step."""
import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

SLIDE_STATE_NAMES = (
    'gripper_x','gripper_y','gripper_z',
    'puck_x','puck_y','puck_z',
    'puck_step_vx','puck_step_vy','puck_step_vz',
    'gripper_step_vx','gripper_step_vy','gripper_step_vz',
    'puck_step_omega_x','puck_step_omega_y','puck_step_omega_z',
)
SLIDE_STATE_DIM = len(SLIDE_STATE_NAMES)
STEP_SECONDS = .04


def make_slide_env(render_mode=None, max_episode_steps=50):
    return gym.make('FetchSlide-v4', render_mode=render_mode,
                    max_episode_steps=max_episode_steps)


def slide_state(obs):
    raw = np.asarray(obs['observation'],dtype=np.float32)
    if raw.shape != (25,):
        raise ValueError(f'FetchSlide needs 25-D observation, got {raw.shape}')
    # Fetch returns relative object velocity + gripper velocity, each scaled
    # by the 0.04 s control interval. Angular velocity is already absolute.
    return np.concatenate((raw[:6],raw[14:17]+raw[20:23],raw[20:23],raw[17:20])).astype(np.float32)


def slide_goal(obs):
    return np.asarray(obs['desired_goal'],dtype=np.float32).copy()


def slide_distance(state,goal):
    return float(np.linalg.norm(np.asarray(state)[3:6]-np.asarray(goal)))
