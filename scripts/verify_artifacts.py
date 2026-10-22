"""Recompute saved split, checkpoint, prediction, and per-seed evaluation invariants.

Reads committed artifacts; does not collect data or rerun control episodes.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from robot_world_model.data import load
from robot_world_model.model import load_model
from robot_world_model.phase3 import predict
from robot_world_model.slide_prediction import load_models, prediction_report

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [center-radius, center+radius]


def verify_dataset(name, checkpoint, dimension):
    data = load(ROOT / name)
    state, action, next_state = (data[k] for k in ('state', 'action', 'next_state'))
    assert state.shape == next_state.shape and state.shape[1] == dimension
    assert action.shape == (len(state), 4) and data['goal'].shape == (len(state), 3)
    seeds = [set(data['episode_seed'][data['split'] == i]) for i in range(3)]
    assert all(not seeds[i] & seeds[j] for i in range(3) for j in range(i+1, 3))
    assert len(seeds[0]) and len(seeds[1]) and len(seeds[2])
    model = load_model(ROOT / checkpoint)
    assert model.state_dim == dimension
    train = data['split'] == 0
    x = torch.from_numpy(np.concatenate((state[train], action[train]), axis=1).astype(np.float32))
    delta = torch.from_numpy((next_state[train] - state[train]).astype(np.float32))
    close(model.input_mean.numpy(), x.mean(0).numpy())
    close(model.input_std.numpy(), x.std(0).clamp_min(1e-3).numpy())
    close(model.delta_mean.numpy(), delta.mean(0).numpy())
    close(model.delta_std.numpy(), delta.std(0).clamp_min(1e-4).numpy())
    print(f'{name}: {len(state)} transitions, episode split {[len(x) for x in seeds]}, {dimension}-D checkpoint')
    return data


def paired(a, b):
    sa = np.asarray([r['final_success'] for r in a], dtype=float)
    sb = np.asarray([r['final_success'] for r in b], dtype=float)
    da = np.asarray([r['final_distance_m'] for r in a])
    db = np.asarray([r['final_distance_m'] for r in b])
    a_only = int(((sa == 1) & (sb == 0)).sum())
    b_only = int(((sa == 0) & (sb == 1)).sum())
    discordant = a_only + b_only
    p = min(1., 2*sum(math.comb(discordant, k) for k in range(min(a_only,b_only)+1))/2**discordant) if discordant else 1.
    ix = np.random.default_rng(104).integers(len(a), size=(10000,len(a)))
    return dict(n=len(a), a_only_success=a_only, b_only_success=b_only,
                success_rate_difference=float((sa-sb).mean()),
                success_difference_bootstrap_95_ci=np.quantile((sa-sb)[ix].mean(1),[.025,.975]),
                exact_mcnemar_two_sided_p=p,
                mean_distance_difference_m=float((da-db).mean()),
                distance_difference_bootstrap_95_ci_m=np.quantile((da-db)[ix].mean(1),[.025,.975]))


def verify_control(path):
    report = json.loads((ROOT / path).read_text())
    seeds = report['seeds']
    assert len(seeds) == 100 and len(set(seeds)) == 100
    reference = None
    for name, policy in report['policies'].items():
        rows = policy['episodes']
        assert [r['seed'] for r in rows] == seeds
        initial = [(r.get('initial_state'),r.get('initial_goal')) if 'initial_state' in r
                   else r['initial_distance_m'] for r in rows]
        if reference is None: reference = initial
        else: assert initial == reference
        unsolved = [r for r in rows if not r['initial_success']]
        success = sum(r['final_success'] for r in unsolved)
        assert success == policy['new_successes']
        assert len(unsolved) == policy['initially_unsolved_episodes']
        assert sum(r['final_success'] for r in rows) == policy['overall_successes']
        close(policy['new_success_rate'], success/len(unsolved))
        close(policy['new_success_wilson_95_ci'], wilson(success,len(unsolved)))
        close(policy['mean_final_distance_m'], np.mean([r['final_distance_m'] for r in rows]))
        print(f'{path}: {name} {success}/{len(unsolved)}, {policy["mean_final_distance_m"]*1000:.1f} mm')
    for key, saved in report['paired_comparisons'].items():
        a, b = key.split('_minus_')
        a = {'rich':'rich_model_mpc','learned':'learned_mpc','hybrid':'hybrid_mpc'}.get(a,a)
        aa = [r for r in report['policies'][a]['episodes'] if not r['initial_success']]
        bb = [r for r in report['policies'][b]['episodes'] if not r['initial_success']]
        assert [r['seed'] for r in aa] == [r['seed'] for r in bb]
        recalculated = paired(aa,bb)
        for field, value in saved.items(): close(value,recalculated[field])
    print(f'{path}: {len(report["paired_comparisons"])} paired statistics verified')


def verify_predictions(rich, slide):
    push = predict(load_model(ROOT/'results/dynamics.pt'),load_model(ROOT/'results/rich.pt'),rich)
    saved = json.loads((ROOT/'results/phase3_prediction.json').read_text())
    for name in push['models']:
        for regime, row in push['models'][name]['regimes'].items():
            close(row['model_object_rmse_m'],saved['models'][name]['regimes'][regime]['model_object_rmse_m'])
    print('FetchPush prediction RMSE recomputed from held-out transitions')
    models = load_models()
    recalculated = prediction_report(slide,models)
    saved = json.loads((ROOT/'results/slide_prediction.json').read_text())
    for name, row in recalculated['models'].items():
        for regime, stats in row['one_step'].items():
            for metric in ('position_rmse_m','velocity_rmse_m_per_step'):
                close(stats[metric],saved['models'][name]['one_step'][regime][metric])
        for horizon, stats in row['multi_step'].items():
            close(stats['object_position_rmse_m'],saved['models'][name]['multi_step'][horizon]['object_position_rmse_m'])
        close(row['final_stopping_position']['mean_xy_error_m'],
              saved['models'][name]['final_stopping_position']['mean_xy_error_m'])
    print('FetchSlide one-step, multi-step, and stopping errors recomputed')


if __name__ == '__main__':
    import os
    os.chdir(ROOT)  # model loader uses repository-relative paths for the fixed ablation
    verify_dataset('data/small.npz','results/dynamics.pt',12)
    rich = verify_dataset('data/rich.npz','results/rich.pt',12)
    slide = verify_dataset('data/slide.npz','results/slide.pt',15)
    verify_control('results/phase3_final_evaluation.json')
    verify_control('results/slide_final_evaluation.json')
    verify_predictions(rich,slide)
