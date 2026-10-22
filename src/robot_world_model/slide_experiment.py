"""Reproduce FetchSlide feasibility, prediction, and matched policy evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np

from .model import train
from .slide_data import collect_slide,load_slide
from .slide_env import make_slide_env,slide_state,slide_goal,slide_distance,SLIDE_STATE_NAMES
from .slide_policies import SlideController
from .slide_physics import fit_slide_physics
from .slide_prediction import load_models,prediction_report,plot_prediction,plot_trajectory
from .slide_evaluate import evaluate_slide,plot_control,plot_trajectories,FINAL_SEEDS,CEM_CONFIG
from .slide_mpc import SlideCEM


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def probe(seed=400):
    env=make_slide_env(max_episode_steps=50)
    try:
        obs,_=env.reset(seed=seed)
        state=slide_state(obs);goal=slide_goal(obs)
        action=np.array([1,0,0,0],dtype=np.float32)
        next_obs,reward,terminated,truncated,info=env.step(action)
        a=np.array([1.,0.,0.]);inside=a+np.array([.049,0,0]);outside=a+np.array([.051,0,0])
        result={'environment':'FetchSlide-v4','seed':seed,
                'raw_observation_shape':list(obs['observation'].shape),
                'state_dimension':len(state),'state_names':SLIDE_STATE_NAMES,
                'action_shape':list(env.action_space.shape),
                'initial_gripper_xyz_m':state[:3].tolist(),
                'initial_puck_xyz_m':state[3:6].tolist(),
                'initial_puck_step_velocity_m':state[6:9].tolist(),
                'initial_puck_angular_step_velocity_rad':state[12:15].tolist(),
                'goal_xyz_m':goal.tolist(),'initial_distance_m':slide_distance(state,goal),
                'reward_after_step':float(reward),'success_after_step':bool(info['is_success']),
                'terminated':bool(terminated),'truncated':bool(truncated),
                'reward_at_4_9cm':float(env.unwrapped.compute_reward(inside,a,{})),
                'reward_at_5_1cm':float(env.unwrapped.compute_reward(outside,a,{})),
                'gripper_step_position_change_m':(slide_state(next_obs)[:3]-state[:3]).tolist(),
                'observed_gripper_step_velocity_m':slide_state(next_obs)[9:12].tolist()}
        return result
    finally:env.close()


def train_slide(data):
    weights=np.ones(len(data['state']),dtype=np.float32)
    weights[data['near_contact']]=1.5
    weights[data['sliding']]=2.
    weights[data['impact']]=8.
    physics=fit_slide_physics(data,'results/slide_physics.json')
    learned=train(data,'results/slide.pt',epochs=45,seed=17,sample_weights=weights)
    return {'physics_fit':physics,'learned_training':learned,
            'sample_weighting':{'free':1,'near':1.5,'sliding':2,'impact':8}}


def demo(seed=405,path='results/slide_learned_mpc.gif'):
    import imageio.v2 as imageio
    import json as json_module
    drag=json_module.loads(Path('results/slide_physics.json').read_text())['sliding_deceleration_m_per_step2']
    model=load_models()['learned_mlp']
    env=make_slide_env(render_mode='rgb_array',max_episode_steps=100)
    controller=SlideCEM(model,drag,seed=seed,**CEM_CONFIG)
    frames=[]
    try:
        obs,_=env.reset(seed=seed)
        frames.append(env.render())
        for t in range(100):
            action=controller.act(slide_state(obs),slide_goal(obs))[0] if t<50 else np.zeros(4,dtype=np.float32)
            obs,_,_,_,info=env.step(action)
            if t%2==1:frames.append(env.render())
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(path,frames,fps=10)
        return {'seed':seed,'controlled_steps':50,'zero_action_coast_steps':50,
                'final_settling_distance_m':slide_distance(slide_state(obs),slide_goal(obs)),
                'settling_success':bool(info['is_success']),'frames':len(frames)}
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('probe','collect','train','predict','dev','final','demo'))
    args=parser.parse_args()
    if args.command=='probe':
        result=probe();save_json('results/slide_probe.json',result)
    elif args.command=='collect':
        result=collect_slide('data/slide.npz',episodes=200,seed=6000)
        save_json('results/slide_collection.json',result)
    elif args.command=='train':
        result=train_slide(load_slide('data/slide.npz'))
        save_json('results/slide_training.json',result)
    elif args.command=='predict':
        data=load_slide('data/slide.npz')
        models=load_models()
        result=prediction_report(data,models)
        result['trajectory_example']=plot_trajectory(data,models,'results/slide_prediction_trajectory.png')
        save_json('results/slide_prediction.json',result)
        plot_prediction(result,'results/slide_prediction.png')
    elif args.command in ('dev','final'):
        seeds=list(range(400,430)) if args.command=='dev' else list(FINAL_SEEDS)
        drag=json.loads(Path('results/slide_physics.json').read_text())['sliding_deceleration_m_per_step2']
        result=evaluate_slide(load_models(),drag,seeds)
        save_json(f'results/slide_{args.command}_evaluation.json',result)
        if args.command=='final':
            plot_control(result,'results/slide_controllers.png')
            plot_trajectories(result,'results/slide_trajectories.png')
    else:
        result=demo()
        save_json('results/slide_demo.json',result)
    if args.command in ('dev','final'):
        print(json.dumps({name:{'new_successes':row['new_successes'],
                                'initially_unsolved_episodes':row['initially_unsolved_episodes'],
                                'mean_final_distance_m':row['mean_final_distance_m']}
                          for name,row in result['policies'].items()},indent=2))
    elif args.command=='train':
        print(json.dumps({'best_epoch':result['learned_training']['best_epoch'],
                          'best_validation_normalized_mse':result['learned_training']['best_validation_normalized_mse'],
                          'physics_fit':result['physics_fit']},indent=2))
    else:
        print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))


if __name__=='__main__':main()
