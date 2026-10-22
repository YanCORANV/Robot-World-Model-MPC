"""Small analytic impact-plus-friction baseline, fitted on train episodes only."""
import json
from pathlib import Path
import numpy as np
import torch


def fit_slide_physics(data,path):
    mask=data['split']==0
    s=data['state'][mask];n=data['next_state'][mask];a=data['action'][mask]
    gain=[];retention=[]
    for axis in range(3):
        x=np.stack((s[:,9+axis],a[:,axis]),axis=1)
        y=n[:,9+axis]
        coef=np.linalg.lstsq(x,y,rcond=None)[0]
        retention.append(float(np.clip(coef[0],0,1)))
        gain.append(float(np.clip(coef[1],0,.06)))
    sliding=data['sliding'][mask].astype(bool)
    fast=np.linalg.norm(s[:,6:8],axis=1)>.005
    valid=sliding&fast
    decel=np.linalg.norm(s[valid,6:8],axis=1)-np.linalg.norm(n[valid,6:8],axis=1)
    drag=float(np.clip(np.median(decel),.0002,.004)) if len(decel) else .0016
    impact=data['impact'][mask].astype(bool)
    sep=s[:,3:5]-s[:,:2]
    unit=sep/np.maximum(np.linalg.norm(sep,axis=1,keepdims=True),1e-6)
    closing=np.sum((n[:,9:11]-s[:,6:8])*unit,axis=1)
    dv=np.sum((n[:,6:8]-s[:,6:8])*unit,axis=1)
    candidates=impact&(closing>.003)&(np.linalg.norm(sep,axis=1)<.065)
    ratios=dv[candidates]/closing[candidates]
    impulse_gain=float(np.clip(np.median(ratios),.4,3)) if len(ratios) else 1.7
    params={'gripper_retention':retention,'gripper_action_gain':gain,
            'sliding_deceleration_m_per_step2':drag,'impact_gain':impulse_gain,
            'impact_fit_transitions':int(candidates.sum()),
            'drag_fit_transitions':int(valid.sum()),'contact_radius_m':.061}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(params,indent=2)+'\n')
    return params


class SimpleSlidePhysics(torch.nn.Module):
    def __init__(self,params):
        super().__init__()
        self.register_buffer('retention',torch.tensor(params['gripper_retention'],dtype=torch.float32))
        self.register_buffer('gain',torch.tensor(params['gripper_action_gain'],dtype=torch.float32))
        self.drag=float(params['sliding_deceleration_m_per_step2'])
        self.impact_gain=float(params['impact_gain'])
        self.radius=float(params['contact_radius_m'])

    def forward(self,state,action):
        # Each candidate future is predicted from the state and action only.
        next_state=state.clone()
        grip_v=self.retention*state[:,9:12]+self.gain*action[:,:3]
        grip=state[:,:3]+grip_v
        velocity=state[:,6:8]
        speed=torch.linalg.norm(velocity,dim=1,keepdim=True)
        new_velocity=velocity*torch.clamp(1-self.drag/speed.clamp_min(1e-6),min=0)
        relative=state[:,3:5]-state[:,:2]
        separation=torch.linalg.norm(relative,dim=1,keepdim=True).clamp_min(1e-6)
        normal=relative/separation
        closing=torch.sum((grip_v[:,:2]-velocity)*normal,dim=1,keepdim=True)
        vertical=torch.abs(state[:,2:3]-state[:,5:6])<.04
        hit=(separation<self.radius)&vertical&(closing>.001)
        new_velocity=new_velocity+hit.float()*self.impact_gain*closing.clamp_min(0)*normal
        # Fetch velocity is scaled by control-step duration; integration below is approximate.
        next_state[:,:3]=grip
        next_state[:,9:12]=grip_v
        next_state[:,3:5]=state[:,3:5]+(velocity+new_velocity)*.5
        next_state[:,6:8]=new_velocity
        next_state[:,8]=0
        next_state[:,12:15]=state[:,12:15]*.9
        return next_state


def load_slide_physics(path):
    return SimpleSlidePhysics(json.loads(Path(path).read_text())).eval()


class HybridSlideDynamics(torch.nn.Module):
    """Use learned dynamics for closing contact, analytic drag for free coast."""
    def __init__(self,learned,physics):
        super().__init__()
        self.learned=learned.eval()
        self.physics=physics.eval()

    def forward(self,state,action):
        relative=state[:,3:5]-state[:,:2]
        sep=torch.linalg.norm(relative,dim=1).clamp_min(1e-6)
        normal=relative/sep[:,None]
        closing=torch.sum((action[:,:2]*.025-state[:,6:8])*normal,dim=1)
        use_learned=(sep<.075)&(torch.abs(state[:,2]-state[:,5])<.04)&(closing>.002)
        learned=self.learned(state,action)
        physics=self.physics(state,action)
        return torch.where(use_learned[:,None],learned,physics)
