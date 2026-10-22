"""Matched FetchSlide controller test with a separate no-action settling period."""
import time
import numpy as np

from .data import robot_object_contact
from .phase3 import wilson, paired_stats
from .slide_env import make_slide_env,slide_state,slide_goal,slide_distance
from .slide_policies import SlideController,slide_random_action
from .slide_mpc import SlideCEM

POLICIES=('random','geometry','impulse','zero_change_mpc','physics_mpc','learned_mpc','hybrid_mpc')
FINAL_SEEDS=tuple(range(30000,30100))
CONTROL_STEPS=50
COAST_STEPS=50
CEM_CONFIG={'horizon':25,'candidates':48,'iterations':2}


def add_settling_comparisons(report):
    """Paired post-hoc settling-distance summary from saved test trajectories."""
    reference=[r for r in report['policies']['learned_mpc']['episodes'] if not r['initial_success']]
    result={}
    for name in ('impulse','zero_change_mpc','physics_mpc','hybrid_mpc'):
        other=[r for r in report['policies'][name]['episodes'] if not r['initial_success']]
        assert [r['seed'] for r in reference]==[r['seed'] for r in other]
        diffs=np.asarray([a['settling_position_error_m']-b['settling_position_error_m']
                          for a,b in zip(reference,other)])
        rng=np.random.default_rng(104)
        ix=rng.integers(len(diffs),size=(10000,len(diffs)))
        result[f'learned_minus_{name}']={
            'n':len(diffs),'mean_difference_m':float(np.mean(diffs)),
            'paired_bootstrap_95_ci_m':np.quantile(diffs[ix].mean(1),[.025,.975]).tolist()}
    report['paired_settling_position_comparisons']=result
    return report


def evaluate_slide(models,drag,seeds,steps=CONTROL_STEPS,coast_steps=COAST_STEPS):
    report={'seeds':list(seeds),'controlled_steps':steps,'zero_action_coast_steps':coast_steps,
            'cem_config':CEM_CONFIG,'policies':{},'paired_comparisons':{}}
    for name in POLICIES:
        episodes=[]
        for seed in seeds:
            env=make_slide_env(max_episode_steps=steps+coast_steps)
            rng=np.random.default_rng(seed+12000)
            if name in ('geometry','impulse'):
                controller=SlideController(name)
            elif name.endswith('_mpc'):
                key={'zero_change_mpc':'zero_change','physics_mpc':'simple_physics',
                     'learned_mpc':'learned_mlp','hybrid_mpc':'hybrid_learned_physics'}[name]
                controller=SlideCEM(models[key],drag,seed=seed,**CEM_CONFIG)
            else:controller=None
            try:
                obs,_=env.reset(seed=seed)
                init_state=slide_state(obs)
                init_goal=slide_goal(obs)
                initial=slide_distance(init_state,init_goal)
                ever=initial<.05
                contact=impact=near=0
                plan_time=push_plan_time=0.
                push_count=0
                first_strike_action=None
                first_strike_state=None
                first_strike_goal=None
                path=[init_state[3:5].tolist()] if seed==seeds[0] else None
                grip_path=[init_state[:2].tolist()] if seed==seeds[0] else None
                for t in range(steps):
                    state=slide_state(obs);goal=slide_goal(obs)
                    near+=int(np.linalg.norm(state[:3]-state[3:6])<.11)
                    start=time.perf_counter()
                    if controller is None:
                        action=slide_random_action(rng);phase='random'
                    elif name.endswith('_mpc'):
                        action,elapsed=controller.act(state,goal);phase=controller.last_phase
                    else:
                        action=controller.act(state,goal);phase=controller.last_phase
                    if controller is None or not name.endswith('_mpc'):
                        elapsed=time.perf_counter()-start
                    plan_time+=elapsed
                    if phase=='strike':
                        push_count+=1;push_plan_time+=elapsed
                        if first_strike_action is None:
                            first_strike_action=action[:3].tolist()
                            first_strike_state=state.tolist()
                            first_strike_goal=goal.tolist()
                    next_obs,_,_,_,info=env.step(action)
                    next_state=slide_state(next_obs)
                    contact+=int(robot_object_contact(env))
                    change=np.linalg.norm(next_state[6:8]-state[6:8])
                    sep=min(np.linalg.norm(state[:3]-state[3:6]),
                            np.linalg.norm(next_state[:3]-next_state[3:6]))
                    impact+=int(change>.003 and sep<.11)
                    ever|=bool(info['is_success'])
                    obs=next_obs
                    if path is not None:
                        path.append(next_state[3:5].tolist())
                        grip_path.append(next_state[:2].tolist())
                final_state=slide_state(obs)
                final_distance=slide_distance(final_state,init_goal)
                final_success=bool(info['is_success'])
                for _ in range(coast_steps):
                    obs,_,_,_,settle_info=env.step(np.zeros(4,dtype=np.float32))
                    if path is not None:path.append(slide_state(obs)[3:5].tolist())
                stop_state=slide_state(obs)
                episode={'seed':seed,'initial_state':init_state.tolist(),
                         'initial_goal':init_goal.tolist(),
                         'initial_distance_m':initial,'initial_success':initial<.05,
                         'final_success':final_success,'ever_success':bool(ever),
                         'final_distance_m':final_distance,
                         'final_object_speed_m_per_step':float(np.linalg.norm(final_state[6:8])),
                         'settling_position_error_m':slide_distance(stop_state,init_goal),
                         'settling_speed_m_per_step':float(np.linalg.norm(stop_state[6:8])),
                         'settling_success':bool(settle_info['is_success']),
                         'near_fraction':near/steps,'end_step_contact_fraction':contact/steps,
                         'impact_proxy_fraction':impact/steps,
                         'runtime_ms_per_step':1000*plan_time/steps,
                         'strike_runtime_ms':1000*push_plan_time/push_count if push_count else 0,
                         'strike_steps':push_count,'first_strike_action_xyz':first_strike_action,
                         'first_strike_state':first_strike_state,'first_strike_goal':first_strike_goal}
                if path is not None:
                    episode['object_path_xy']=path
                    episode['gripper_path_xy']=grip_path
                episodes.append(episode)
            finally:env.close()
        unsolved=[r for r in episodes if not r['initial_success']]
        successes=sum(r['final_success'] for r in unsolved)
        report['policies'][name]={
            'initial_successes':len(episodes)-len(unsolved),
            'new_successes':successes,'initially_unsolved_episodes':len(unsolved),
            'new_success_rate':successes/len(unsolved),
            'new_success_wilson_95_ci':wilson(successes,len(unsolved)),
            'overall_successes':sum(r['final_success'] for r in episodes),
            'overall_success_rate':float(np.mean([r['final_success'] for r in episodes])),
            'ever_successes':sum(r['ever_success'] for r in episodes),
            'settling_successes':sum(r['settling_success'] for r in episodes),
            'mean_final_distance_m':float(np.mean([r['final_distance_m'] for r in episodes])),
            'mean_settling_position_error_m':float(np.mean([r['settling_position_error_m'] for r in episodes])),
            'mean_settling_speed_m_per_step':float(np.mean([r['settling_speed_m_per_step'] for r in episodes])),
            'mean_near_fraction':float(np.mean([r['near_fraction'] for r in episodes])),
            'mean_end_step_contact_fraction':float(np.mean([r['end_step_contact_fraction'] for r in episodes])),
            'mean_impact_proxy_fraction':float(np.mean([r['impact_proxy_fraction'] for r in episodes])),
            'mean_runtime_ms_per_step':float(np.mean([r['runtime_ms_per_step'] for r in episodes])),
            'mean_strike_runtime_ms':float(np.mean([r['strike_runtime_ms'] for r in episodes if r['strike_steps']])) if any(r['strike_steps'] for r in episodes) else 0,
            'episodes':episodes}
        print(f'{name}: {successes}/{len(unsolved)} final, {report["policies"][name]["ever_successes"]} ever',flush=True)
    reference=report['policies'][POLICIES[0]]['episodes']
    for name in POLICIES[1:]:
        other=report['policies'][name]['episodes']
        assert all(np.array_equal(a['initial_state'],b['initial_state']) and
                   np.array_equal(a['initial_goal'],b['initial_goal']) for a,b in zip(reference,other))
    learned=[r for r in report['policies']['learned_mpc']['episodes'] if not r['initial_success']]
    for name in POLICIES:
        if name=='learned_mpc':continue
        other=[r for r in report['policies'][name]['episodes'] if not r['initial_success']]
        report['paired_comparisons'][f'learned_minus_{name}']=paired_stats(learned,other)
    hybrid=[r for r in report['policies']['hybrid_mpc']['episodes'] if not r['initial_success']]
    for name in ('physics_mpc','impulse','learned_mpc'):
        other=[r for r in report['policies'][name]['episodes'] if not r['initial_success']]
        report['paired_comparisons'][f'hybrid_minus_{name}']=paired_stats(hybrid,other)
    audit={}
    reference=report['policies']['learned_mpc']['episodes']
    for name in ('impulse','zero_change_mpc','physics_mpc','hybrid_mpc'):
        other=report['policies'][name]['episodes']
        pairs=[(a,b) for a,b in zip(reference,other)
               if a['first_strike_action_xyz'] is not None and b['first_strike_action_xyz'] is not None]
        assert all(np.allclose(a['first_strike_state'],b['first_strike_state'],atol=1e-6)
                   and np.allclose(a['first_strike_goal'],b['first_strike_goal'],atol=1e-6)
                   for a,b in pairs), 'Matched first-strike state differs'
        differences=[np.linalg.norm(np.asarray(a['first_strike_action_xyz'])-
                                    b['first_strike_action_xyz']) for a,b in pairs]
        audit[name]={'matched_states':len(pairs),
                     'mean_l2_action_difference_from_learned':float(np.mean(differences)) if pairs else None}
    report['first_strike_action_audit']=audit
    return add_settling_comparisons(report)


def plot_control(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(POLICIES)
    labels=['Random','Geometry','Impulse','Zero CEM','Physics CEM','Learned CEM','Hybrid CEM']
    p=report['policies'];x=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    rates=[p[n]['new_success_rate'] for n in names]
    lo=[rates[i]-p[n]['new_success_wilson_95_ci'][0] for i,n in enumerate(names)]
    hi=[p[n]['new_success_wilson_95_ci'][1]-rates[i] for i,n in enumerate(names)]
    axes[0].bar(x,rates,color='#4477aa')
    axes[0].errorbar(x,rates,yerr=[lo,hi],fmt='none',color='black',capsize=3)
    axes[0].set(ylabel='Final success, initially unsolved',ylim=(0,1.05))
    axes[1].bar(x,[p[n]['mean_final_distance_m'] for n in names],color='#cc6677')
    axes[1].set(ylabel='Final puck-to-goal distance (m)')
    for ax in axes:ax.set_xticks(x,labels=labels,rotation=25)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_trajectories(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for name,group in report['policies'].items():
        row=group['episodes'][0]
        xy=np.asarray(row['object_path_xy'])
        line,=ax.plot(xy[:51,0],xy[:51,1],label=name)
        ax.plot(xy[50:,0],xy[50:,1],linestyle=':',alpha=.7,color=line.get_color())
    row=report['policies']['random']['episodes'][0]
    ax.scatter(row['initial_goal'][0],row['initial_goal'][1],marker='*',s=150,c='black',label='goal')
    ax.set(xlabel='Puck X (m)',ylabel='Puck Y (m)',title=f'FetchSlide seed {row["seed"]}: controlled then free coast')
    ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
