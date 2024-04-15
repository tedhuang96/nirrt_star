import json
import time
import yaml
from os.path import join

import cv2
import numpy as np

from datasets.point_cloud_mask_utils import get_binary_mask, get_point_cloud_mask_around_points, \
    generate_rectangle_point_cloud

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


config_name = "random_2d"
with open(join("env_configs", config_name+".yml"), 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

dataset_dir = join("data", config_name)
img_height, img_width = config['env_height'], config['env_width']
n_points = config['n_points']
over_sample_scale = config['over_sample_scale']
start_radius = config['start_radius']
goal_radius = config['goal_radius']
path_radius = config['path_radius']


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
        env_img = cv2.imread(join(dataset_dir, mode, "env_imgs", "{0}.png".format(env_idx)))
        binary_mask = get_binary_mask(env_img)
        for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict['start'], env_dict['goal'])):
            s_start, s_goal = np.array(s_start), np.array(s_goal)
            start_point = s_start[np.newaxis,:]
            goal_point = s_goal[np.newaxis,:]
            sample_title = "{0}_{1}".format(env_idx, sample_idx)
            path = np.loadtxt(
                join(dataset_dir, mode, "astar_paths", sample_title+".txt"),
                delimiter=',',
            )
            token = mode+"-"+sample_title
            pc = generate_rectangle_point_cloud(
                binary_mask,
                n_points,
                over_sample_scale=over_sample_scale,
            ) # (n_points, 2)
            around_start_mask = get_point_cloud_mask_around_points(
                pc,
                start_point,
                neighbor_radius=start_radius,
            ) # (n_points,)
            around_goal_mask = get_point_cloud_mask_around_points(
                pc,
                goal_point,
                neighbor_radius=goal_radius,
            ) # (n_points,)
            around_path_mask = get_point_cloud_mask_around_points(
                pc,
                path,
                neighbor_radius=path_radius,
            ) # (n_points,)
            freespace_mask = (1-around_start_mask)*(1-around_goal_mask)
            raw_dataset['token'].append(token)
            raw_dataset['pc'].append(pc.astype(np.float32))
            raw_dataset['start'].append(around_start_mask.astype(np.float32))
            raw_dataset['goal'].append(around_goal_mask.astype(np.float32))
            raw_dataset['free'].append(freespace_mask.astype(np.float32))
            raw_dataset['astar'].append(around_path_mask.astype(np.float32))
        if (env_idx+1) % 25 == 0:
            save_raw_dataset(
                raw_dataset,
                dataset_dir,
                mode,
                tmp=True,
            )
            time_left = (time.time() - start_time) * (len(env_list) / (env_idx + 1) - 1) / 60
            print(mode+" {0}/{1}, remaining time: {2} min".format(env_idx+1, len(env_list), int(time_left)))
    save_raw_dataset(
        raw_dataset,
        dataset_dir,
        mode,
        tmp=False,
    )