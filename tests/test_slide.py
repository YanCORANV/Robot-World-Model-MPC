import numpy as np
import torch

from robot_world_model.model import DynamicsMLP
from robot_world_model.slide_data import phase_labels
from robot_world_model.slide_env import make_slide_env, slide_state
from robot_world_model.slide_evaluate import FINAL_SEEDS, CEM_CONFIG
from robot_world_model.slide_mpc import SlideCEM
from robot_world_model.slide_physics import SimpleSlidePhysics
from robot_world_model.slide_policies import SlideController


def test_slide_state_uses_explicit_linear_and_angular_velocity():
    raw=np.zeros(25,dtype=np.float32)
    raw[14:17]=[.01,.02,.03]
    raw[20:23]=[.04,.05,.06]
    raw[17:20]=[.1,.2,.3]
    s=slide_state({'observation':raw})
    assert s.shape==(15,)
    np.testing.assert_allclose(s[6:9],[.05,.07,.09])
    np.testing.assert_allclose(s[12:15],[.1,.2,.3])
    assert DynamicsMLP(state_dim=15)(torch.zeros(2,15),torch.zeros(2,4)).shape==(2,15)


def test_slide_environment_and_sparse_threshold():
    env=make_slide_env()
    try:
        obs,_=env.reset(seed=402)
        assert obs['observation'].shape==(25,) and env.action_space.shape==(4,)
        a=np.array([1.,0.,0.])
        assert float(env.unwrapped.compute_reward(a+np.array([.049,0,0]),a,{}))==0
        assert float(env.unwrapped.compute_reward(a+np.array([.051,0,0]),a,{}))==-1
        obs,reward,terminated,truncated,info=env.step(np.zeros(4,dtype=np.float32))
        assert np.isfinite(slide_state(obs)).all() and reward in (-1.,0.)
        assert not terminated and not truncated and 'is_success' in info
    finally:env.close()


def test_impact_label_and_physics_coasting():
    s=np.zeros(15,dtype=np.float32)
    s[:3]=[0,0,.426];s[3:6]=[.05,0,.414]
    n=s.copy();n[6]=.03
    assert phase_labels(s,n,False)[0]
    params={'gripper_retention':[0,0,0], 'gripper_action_gain':[.025]*3,
            'sliding_deceleration_m_per_step2':.0016,'impact_gain':1.7,'contact_radius_m':.061}
    model=SimpleSlidePhysics(params)
    state=torch.tensor(s[None])
    state[:,6]=.02
    output=model(state,torch.zeros(1,4))
    assert 0<float(output[0,6])<.02
    state[:,6]=0
    struck=model(state,torch.tensor([[1.,0,0,0]]))
    assert float(struck[0,6])>0


def test_slide_cem_queries_model_only_during_strike():
    class Counting(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,s,a):self.calls+=1;return s
    model=Counting()
    c=SlideCEM(model,.0016,horizon=3,candidates=8,iterations=1)
    s=np.zeros(15,dtype=np.float32)
    s[3:6]=[1.,.75,.414]
    goal=np.array([1.3,.75,.414],dtype=np.float32)
    s[:3]=[.5,.75,.47]
    c.act(s,goal)
    assert c.last_phase=='approach' and model.calls==0
    s[:3]=[.925,.75,.426]
    a,_=c.act(s,goal)
    assert c.last_phase=='strike' and model.calls>0
    assert a.shape==(4,) and np.max(np.abs(a))<=1
    assert len(FINAL_SEEDS)==100 and min(FINAL_SEEDS)>6000
    assert CEM_CONFIG=={'horizon':25,'candidates':48,'iterations':2}


def test_collection_keeps_original_impulse_sampler():
    s=np.zeros(15,dtype=np.float32)
    s[:3]=[.95,.75,.426]
    s[3:6]=[1.,.75,.414]
    s[6]=.03
    goal=np.array([1.3,.75,.414],dtype=np.float32)
    evaluation=SlideController('impulse')
    collection=SlideController('impulse',velocity_cutoff=False)
    evaluation.impact_steps=collection.impact_steps=2
    np.testing.assert_array_equal(evaluation.act(s,goal),np.zeros(4))
    assert evaluation.last_phase=='coast'
    assert np.linalg.norm(collection.act(s,goal)[:2])>0
    assert collection.last_phase=='strike'
