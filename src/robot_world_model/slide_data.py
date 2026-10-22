"""Episode-isolated FetchSlide trajectories with explicit motion-phase labels."""
from pathlib import Path
import numpy as np

from .data import robot_object_contact, load
from .slide_env import make_slide_env, slide_state, slide_goal
from .slide_policies import SlideController, slide_random_action

SAMPLERS = ('random','geometry','impulse','perturbed_impact','impulse_then_coast')


def phase_labels(state, next_state, end_contact):
    speed=np.linalg.norm(state[6:8])
    next_speed=np.linalg.norm(next_state[6:8])
    separation=min(np.linalg.norm(state[:3]-state[3:6]),
                   np.linalg.norm(next_state[:3]-next_state[3:6]))
    # A collision during one of the 20 internal substeps may not exist at the
    # final substep. A nearby velocity jump captures such impacts offline.
    impact=bool(end_contact or (separation<.11 and
                               np.linalg.norm(next_state[6:8]-state[6:8])>.003))
    sliding=bool(speed>.002 and not impact)
    decelerating=bool(sliding and next_speed<speed-.0003)
    near=bool(separation<.11 and not impact and not sliding)
    return impact,sliding,decelerating,near


def collect_slide(path,episodes=200,seed=6000,steps=50):
    if episodes<25 or episodes%5:
        raise ValueError('Use at least 25 episodes, divisible by 5')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows={key:[] for key in ('state','action','next_state','goal','episode_id',
          'episode_seed','timestep','sampler','contact','impact','sliding',
          'decelerating','near_contact','success')}
    rng=np.random.default_rng(seed)
    env=make_slide_env(max_episode_steps=steps)
    try:
        for ep in range(episodes):
            obs,_=env.reset(seed=seed+ep)
            mode=ep%5
            # Freeze the exploratory collection behavior used for this dataset.
            # Evaluation later added an observed-speed cutoff; changing the
            # collection policy would change the training distribution.
            controller=(SlideController('geometry' if mode in (1,3) else 'impulse',
                                        velocity_cutoff=False) if mode else None)
            scale=rng.uniform(.4,1.0)
            lateral=rng.uniform(-.22,.22)
            struck=0
            for t in range(steps):
                state=slide_state(obs)
                goal=slide_goal(obs)
                action=slide_random_action(rng) if mode==0 else controller.act(state,goal)
                if mode==3 and controller.last_phase=='strike':
                    direction=action[:2].copy()
                    perp=np.array([-direction[1],direction[0]],dtype=np.float32)
                    action[:2]=np.clip(scale*direction+lateral*perp,-1,1)
                if mode==4:
                    if controller.last_phase=='strike':struck+=1
                    if struck>4:action=np.zeros(4,dtype=np.float32)
                next_obs,_,terminated,truncated,info=env.step(action)
                next_state=slide_state(next_obs)
                contact=robot_object_contact(env)
                impact,sliding,decelerating,near=phase_labels(state,next_state,contact)
                values=(state,action,next_state,goal,ep,seed+ep,t,mode,
                        contact,impact,sliding,decelerating,near,bool(info['is_success']))
                for key,value in zip(rows,values):rows[key].append(value)
                obs=next_obs
                if terminated or truncated:break
    finally:env.close()
    arrays={key:np.asarray(value) for key,value in rows.items()}
    split=np.zeros(len(arrays['episode_id']),dtype=np.int8)
    split_rng=np.random.default_rng(seed+1)
    for mode in range(5):
        ids=split_rng.permutation(np.arange(mode,episodes,5))
        n=max(1,len(ids)//5)
        split[np.isin(arrays['episode_id'],ids[:n])]=1
        split[np.isin(arrays['episode_id'],ids[n:2*n])]=2
    arrays['split']=split
    np.savez_compressed(path,**arrays)
    return {'episodes':episodes,'transitions':len(split),
            'split_counts':{name:int(np.sum(split==i)) for i,name in enumerate(('train','validation','test'))},
            'sampler_counts':{name:int(np.sum(arrays['sampler']==i)) for i,name in enumerate(SAMPLERS)},
            'impact_steps':int(arrays['impact'].sum()),'sliding_steps':int(arrays['sliding'].sum()),
            'decelerating_steps':int(arrays['decelerating'].sum()),
            'end_step_contact_steps':int(arrays['contact'].sum())}


def load_slide(path):
    data=load(path)
    if data['state'].shape[1]!=15:
        raise ValueError('FetchSlide data must contain 15-D state')
    return data
