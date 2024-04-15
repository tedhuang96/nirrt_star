import json
import time
from os.path import join

import yaml
import numpy as np

from path_planning_utils_3d.env_3d import Env
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points
from datasets_3d.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d_v1

def save_raw_dataset(
    raw_dataset,
    dataset_dir,
    mode,
    tmp=False,
):
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        if k == 'token':
            raw_dataset_saved[k] = np.array(raw_dataset[k])
        else:
            raw_dataset_saved[k] = np.stack(raw_dataset[k], axis=0) # (b, n_points, ...)
    if tmp:
        filename = mode+"_tmp.npz"
    else:
        filename = mode+".npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)


config_name = "random_3d"
with open(join("env_configs", config_name+".yml"), 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

dataset_dir = join("data", config_name)

for mode in ['train', 'val', 'test']:
    with open(join(dataset_dir, mode, "envs.json"), 'r') as f:
        env_list = json.load(f)
    raw_dataset = {}
    raw_dataset['token'] = []
    raw_dataset['pc'] = []
    raw_dataset['start'] = []
    raw_dataset['goal'] = []
    raw_dataset['free'] = []
    raw_dataset['astar'] = []
    start_time = time.time()
    for env_idx, env_dict in enumerate(env_list):
        env = Env(
            env_dict['env_dims'],
            env_dict['box_obstacles'],
            env_dict['ball_obstacles'],
            clearance=config['path_clearance'],
            resolution=config['astar_resolution'],
        )

        for sample_idx, (x_start, x_goal) in enumerate(zip(env_dict['start'], env_dict['goal'])):
            env.set_start_goal(x_start, x_goal)
            sample_title = "{0}_{1}".format(env_idx, sample_idx)
            path = np.loadtxt(
                join(dataset_dir, mode, "astar_paths", sample_title+".txt"),
                delimiter=',',
            ) # np (n_path_points, 3)
            token = mode+"-"+sample_title
            pc = generate_rectangle_point_cloud_3d_v1(
                env,
                config['n_points'],
                over_sample_scale=config['over_sample_scale'],
            )
            around_start_mask = get_point_cloud_mask_around_points(
                pc,
                np.array(x_start)[np.newaxis,:],
                neighbor_radius=config['start_radius'],
            ) # (n_points,)
            around_goal_mask = get_point_cloud_mask_around_points(
                pc,
                np.array(x_goal)[np.newaxis,:],
                neighbor_radius=config['goal_radius'],
            ) # (n_points,)
            around_path_mask = get_point_cloud_mask_around_points(
                pc,
                path,
                neighbor_radius=config['path_radius'],
            ) # (n_points,)
            freespace_mask = (1-around_start_mask)*(1-around_goal_mask)
            raw_dataset['token'].append(token)
            raw_dataset['pc'].append(pc.astype(np.float32))
            raw_dataset['start'].append(around_start_mask.astype(np.float32))
            raw_dataset['goal'].append(around_goal_mask.astype(np.float32))
            raw_dataset['free'].append(freespace_mask.astype(np.float32))
            raw_dataset['astar'].append(around_path_mask.astype(np.float32))
        if (env_idx+1) % 25 == 0:
            time_left = (time.time() - start_time) * (len(env_list) / (env_idx + 1) - 1) / 60
            print(mode+" {0}/{1}, remaining time: {2} min".format(env_idx+1, len(env_list), int(time_left)))
    save_raw_dataset(
        raw_dataset,
        dataset_dir,
        mode,
        tmp=False,
    )