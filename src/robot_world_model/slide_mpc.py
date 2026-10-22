"""Shared geometric approach, then model-only CEM over impulse and coast."""
import time
import numpy as np
import torch

from .slide_policies import SlideController, _unit


class SlideCEM:
    def __init__(self,model,drag,horizon=25,candidates=48,iterations=2,seed=1):
        torch.set_num_threads(min(torch.get_num_threads(),4))
        self.model=model.eval()
        self.drag=float(drag)
        self.horizon=horizon
        self.candidates=candidates
        self.iterations=iterations
        self.elites=max(4,candidates//6)
        self.rng=np.random.default_rng(seed)
        self.staging=SlideController('impulse')
        self.last_phase='approach'

    def act(self,state,goal):
        started=time.perf_counter()
        staged=self.staging.act(state,goal)
        self.last_phase=self.staging.last_phase
        if self.last_phase!='strike':
            return staged,time.perf_counter()-started
        # The same staged state, candidate family, cost, and compute budget
        # are used for zero-change, physics, and MLP dynamics.
        obj=state[3:5]
        drift=state[6:8]
        toward=np.asarray(goal[:2])-obj
        direction=_unit(_unit(toward)-.8*drift/max(np.linalg.norm(toward),.05))
        perp=np.array([-direction[1],direction[0]],dtype=np.float32)
        mean=np.array([0.,.7,3.],dtype=np.float32) # angle, drive, strike length
        std=np.array([.25,.25,1.],dtype=np.float32)
        initial=torch.as_tensor(state,dtype=torch.float32).repeat(self.candidates,1)
        desired=torch.as_tensor(goal[:2],dtype=torch.float32)
        height=float(state[5]+.012)
        with torch.no_grad():
            for _ in range(self.iterations):
                sample=self.rng.normal(mean,std,(self.candidates,3)).astype(np.float32)
                sample[:,0]=np.clip(sample[:,0],-.5,.5)
                sample[:,1]=np.clip(sample[:,1],0,1)
                sample[:,2]=np.clip(np.round(sample[:,2]),1,4)
                sample[0]=mean
                directions=np.cos(sample[:,0,None])*direction+np.sin(sample[:,0,None])*perp
                planar=torch.from_numpy(sample[:,1,None]*directions)
                lengths=torch.from_numpy(sample[:,2])
                states=initial.clone()
                phantom=torch.zeros(self.candidates)
                for t in range(self.horizon):
                    actions=torch.zeros(self.candidates,4)
                    active=(lengths>t).float()
                    actions[:,:2]=planar*active[:,None]
                    if t<4:
                        actions[:,2]=active*float(np.clip((height-state[2])/.05,-.4,.4))
                    before=states
                    states=self.model(states,actions)
                    moved=torch.linalg.norm(states[:,3:5]-before[:,3:5],dim=1)
                    sep=torch.linalg.norm(before[:,:3]-before[:,3:6],dim=1)
                    phantom+=torch.relu(sep-.11)*moved
                velocity=states[:,6:8]
                speed=torch.linalg.norm(velocity,dim=1,keepdim=True)
                remaining_steps=torch.clamp(speed/max(self.drag,1e-5),max=25)
                stop=states[:,3:5]+.5*velocity*remaining_steps
                cost=5*torch.linalg.norm(stop-desired,dim=1)
                cost+=1.5*torch.linalg.norm(states[:,3:5]-desired,dim=1)
                cost+=.8*speed[:,0]+8*phantom
                cost+=.015*torch.from_numpy(sample[:,1]*sample[:,2])
                best=torch.topk(cost,self.elites,largest=False).indices.numpy()
                elite=sample[best]
                mean=elite.mean(0)
                std=np.maximum(elite.std(0),[.08,.08,.4])
        angle=float(np.clip(mean[0],-.5,.5))
        drive=float(np.clip(mean[1],0,1))
        action=np.zeros(4,dtype=np.float32)
        action[:2]=drive*(np.cos(angle)*direction+np.sin(angle)*perp)
        action[2]=np.clip((height-state[2])/.05,-.4,.4)
        return action,time.perf_counter()-started
