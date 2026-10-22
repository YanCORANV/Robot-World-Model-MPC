"""Observable-state FetchSlide baselines; no simulator foresight."""
import numpy as np


def _unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/max(np.linalg.norm(x),1e-8)


class SlideController:
    """Common approach/descend; geometry or velocity-aware impulse at contact."""
    def __init__(self, kind='geometry', velocity_cutoff=True):
        if kind not in ('geometry','impulse'):
            raise ValueError(kind)
        self.kind=kind
        self.velocity_cutoff=velocity_cutoff
        self.last_phase='approach'
        self.impact_steps=0
        self.coast_steps=0

    def act(self,state,goal):
        g,o=np.asarray(state[:3]),np.asarray(state[3:6])
        v=np.asarray(state[6:8])
        to_goal=np.asarray(goal[:2])-o[:2]
        distance=np.linalg.norm(to_goal)
        direction=_unit(to_goal)
        # Aim a few degrees against lateral drift while retaining reachability.
        if self.kind=='impulse' and np.linalg.norm(v)>.002:
            direction=_unit(direction-.8*v/max(distance,.05))
        behind=o[:2]-.075*direction
        height=o[2]+.012
        action=np.zeros(4,dtype=np.float32)
        if self.impact_steps:
            if self.impact_steps <= 4:
                if self.kind=='impulse' and self.velocity_cutoff:
                    # A subsequent 0.04 s impact can add roughly 15 mm/step
                    # of speed. Stop before a second hit would overshoot the
                    # velocity required for the remaining stopping distance.
                    target_speed=np.sqrt(2*.0016*distance)
                    forward=np.dot(v,direction)
                    if forward>.005 and forward+.015>=target_speed:
                        self.impact_steps=5
                        self.last_phase='coast'
                        return action
                self.last_phase='strike'
                if self.kind=='geometry':
                    drive=1.0
                else:
                    # About 1.6 mm per step of deceleration is observed on
                    # developer rollouts. Convert desired stopping distance to
                    # an impact-speed target (v^2=2*a*d), then to action scale.
                    desired_speed=np.sqrt(2*.0016*distance)
                    drive=np.clip(desired_speed/.04,.35,1.0)
                action[:2]=drive*direction
                action[2]=np.clip((height-g[2])/.05,-.4,.4)
                self.impact_steps+=1
                return action
            self.last_phase='coast'
            self.coast_steps+=1
            if self.coast_steps<32 and np.linalg.norm(v)>.002:
                return action
            if self.coast_steps<32 and distance<.05:
                return action
            self.impact_steps=0
            self.coast_steps=0
        if distance < .045 and np.linalg.norm(v)<.002:
            self.last_phase='hold'
            return action
        if np.linalg.norm(g[:2]-behind)>.018:
            target=np.r_[behind,max(height+.04,g[2])]
            action[:3]=np.clip((target-g)/.035,-1,1)
            self.last_phase='approach'
        elif abs(g[2]-height)>.014:
            target=np.r_[behind,height]
            action[:3]=np.clip((target-g)/.035,-1,1)
            self.last_phase='descend'
        else:
            # Commit to several forward actions, then let the puck coast.
            self.last_phase='strike'
            if self.kind=='geometry':drive=1.0
            else:
                desired_speed=np.sqrt(2*.0016*distance)
                drive=np.clip(desired_speed/.04,.35,1.0)
            action[:2]=drive*direction
            action[2]=np.clip((height-g[2])/.05,-.4,.4)
            self.impact_steps=2
        return action


def slide_random_action(rng):
    return np.r_[rng.uniform(-1,1,3),0].astype(np.float32)
