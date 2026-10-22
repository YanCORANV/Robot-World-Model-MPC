"""Held-out FetchSlide prediction by impact, coast, and stopping event."""
import json
from pathlib import Path
import numpy as np
import torch

from .improved_mpc import ZeroChangeDynamics
from .slide_physics import load_slide_physics, HybridSlideDynamics
from .model import load_model


def _rmse(error):
    return float(np.sqrt(np.mean(np.square(error)))) if len(error) else None


def _regimes(data,indices):
    impact=data['impact'][indices].astype(bool)
    sliding=data['sliding'][indices].astype(bool)
    near=data['near_contact'][indices].astype(bool)
    decel=data['decelerating'][indices].astype(bool)
    return {'free_space':~impact&~sliding&~near,
            'approaching_contact':near,'impact':impact,'sliding':sliding,
            'decelerating':decel}


def _windows(data,indices,horizon):
    groups=[]
    for j in range(len(indices)-horizon+1):
        group=indices[j:j+horizon]
        if (np.all(np.diff(group)==1)
            and np.all(np.diff(data['timestep'][group])==1)
            and data['episode_id'][group[0]]==data['episode_id'][group[-1]]):
            groups.append(group)
    return np.asarray(groups,dtype=int)


def prediction_report(data,models,horizons=(5,20),split=2):
    indices=np.flatnonzero(data['split']==split)
    state=data['state'][indices].astype(np.float32)
    action=data['action'][indices].astype(np.float32)
    target=data['next_state'][indices].astype(np.float32)
    regimes=_regimes(data,indices)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    report={'split':split,'heldout_transitions':len(indices),'heldout_episodes':len(np.unique(data['episode_id'][indices])),
            'regime_counts':{k:int(v.sum()) for k,v in regimes.items()},'models':{}}
    for name,model in models.items():
        model.eval()
        with torch.no_grad():
            pred=model(torch.from_numpy(state),torch.from_numpy(action)).numpy()
        def subset(mask):
            return {'count':int(mask.sum()),
                    'position_rmse_m':_rmse(pred[mask,3:6]-target[mask,3:6]),
                    'velocity_rmse_m_per_step':_rmse(pred[mask,6:9]-target[mask,6:9]),
                    'predicted_mean_xy_motion_m':float(np.linalg.norm(pred[mask,3:5]-state[mask,3:5],axis=1).mean()) if mask.any() else None,
                    'actual_mean_xy_motion_m':float(np.linalg.norm(target[mask,3:5]-state[mask,3:5],axis=1).mean()) if mask.any() else None}
        row={'one_step':{'all':subset(np.ones(len(indices),bool)),
                         **{k:subset(mask) for k,mask in regimes.items()}},'multi_step':{}}
        for horizon in horizons:
            groups=_windows(data,indices,horizon)
            if not len(groups):continue
            roll=torch.from_numpy(data['state'][groups[:,0]].astype(np.float32))
            with torch.no_grad():
                for k in range(horizon):
                    roll=model(roll,torch.from_numpy(data['action'][groups[:,k]].astype(np.float32)))
            predicted=roll.numpy()
            actual=data['next_state'][groups[:,-1]]
            start_impact=data['impact'][groups[:,0]].astype(bool)
            start_sliding=data['sliding'][groups[:,0]].astype(bool)
            row['multi_step'][str(horizon)]={
                'windows':len(groups),
                'object_position_rmse_m':_rmse(predicted[:,3:6]-actual[:,3:6]),
                'object_velocity_rmse_m_per_step':_rmse(predicted[:,6:9]-actual[:,6:9]),
                'impact_start_position_rmse_m':_rmse(predicted[start_impact,3:6]-actual[start_impact,3:6]),
                'impact_start_windows':int(start_impact.sum()),
                'sliding_start_position_rmse_m':_rmse(predicted[start_sliding,3:6]-actual[start_sliding,3:6]),
                'sliding_start_windows':int(start_sliding.sum())}
        # Genuine final resting positions: final observed speed below 3 mm/step,
        # with >=5 recorded transitions after the last impact in that episode.
        stopping=[]
        for ep in np.unique(data['episode_id'][indices]):
            episode=indices[data['episode_id'][indices]==ep]
            hits=episode[data['impact'][episode]]
            if not len(hits):continue
            after=episode[episode>hits[-1]]
            if len(after)<5 or np.linalg.norm(data['next_state'][episode[-1],6:8])>=.003:
                continue
            roll=torch.from_numpy(data['state'][after[0]].astype(np.float32)[None])
            with torch.no_grad():
                for idx in after:
                    roll=model(roll,torch.from_numpy(data['action'][idx].astype(np.float32)[None]))
            err=np.linalg.norm(roll.numpy()[0,3:5]-data['next_state'][episode[-1],3:5])
            stopping.append(float(err))
        row['final_stopping_position']={'episodes':len(stopping),
                                        'mean_xy_error_m':float(np.mean(stopping)) if stopping else None,
                                        'median_xy_error_m':float(np.median(stopping)) if stopping else None}
        report['models'][name]=row
    return report


def plot_prediction(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(report['models'])
    regimes=('free_space','approaching_contact','impact','sliding','decelerating')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(regimes))
    width=.8/len(names)
    for j,name in enumerate(names):
        r=report['models'][name]
        p=[r['one_step'][k]['position_rmse_m']*1000 for k in regimes]
        v=[r['one_step'][k]['velocity_rmse_m_per_step']*1000 for k in regimes]
        shift=(j-(len(names)-1)/2)*width
        axes[0].bar(x+shift,p,width=width*.95,label=name)
        axes[1].bar(x+shift,v,width=width*.95,label=name)
    for ax in axes:
        ax.set_xticks(x,labels=['Free','Near','Impact','Slide','Decel'],rotation=20)
    axes[0].set_ylabel('Object position RMSE (mm)')
    axes[1].set_ylabel('Object step-velocity RMSE (mm/step)')
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_trajectory(data,models,path,horizon=25):
    """Open-loop object XY after a real impact on a held-out episode."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ix=np.flatnonzero(data['split']==2)
    chosen=None
    for idx in ix[data['impact'][ix]]:
        group=np.arange(idx+1,idx+1+horizon)
        if (group[-1]<len(data['state']) and np.all(data['split'][group]==2)
            and np.all(data['episode_id'][group]==data['episode_id'][idx])
            and np.all(np.diff(data['timestep'][group])==1)):
            chosen=group;break
    if chosen is None:
        raise ValueError('No post-impact held-out trajectory of requested length')
    first=data['state'][chosen[0]].astype(np.float32)
    actual=np.vstack((first[3:5],data['next_state'][chosen,3:5]))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(actual[:,0],actual[:,1],'k-o',markersize=2,label='MuJoCo observed')
    axes[1].plot(np.arange(horizon+1),actual[:,0]-actual[0,0],'k-',label='MuJoCo observed')
    for name,model in models.items():
        roll=torch.from_numpy(first[None])
        path_xy=[first[3:5]]
        with torch.no_grad():
            for idx in chosen:
                roll=model(roll,torch.from_numpy(data['action'][idx].astype(np.float32)[None]))
                path_xy.append(roll.numpy()[0,3:5].copy())
        path_xy=np.asarray(path_xy)
        axes[0].plot(path_xy[:,0],path_xy[:,1],label=name)
        axes[1].plot(np.arange(horizon+1),path_xy[:,0]-path_xy[0,0],label=name)
    axes[0].set(xlabel='Puck x (m)',ylabel='Puck y (m)',title='Open-loop XY trajectory')
    axes[1].set(xlabel='Steps after impact',ylabel='X displacement (m)',title='Compounding displacement error')
    axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
    return {'episode_id':int(data['episode_id'][chosen[0]]),
            'episode_seed':int(data['episode_seed'][chosen[0]]),
            'start_timestep':int(data['timestep'][chosen[0]]),'horizon':horizon}


def load_models():
    physics=load_slide_physics('results/slide_physics.json')
    learned=load_model('results/slide.pt')
    return {'zero_change':ZeroChangeDynamics(),
            'simple_physics':physics,
            'learned_mlp':learned,
            'hybrid_learned_physics':HybridSlideDynamics(learned,physics)}
