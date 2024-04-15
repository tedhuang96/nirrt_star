import json
from copy import copy
from os.path import join

import math
import numpy as np

from path_planning_utils_3d.rrt_env_3d import Env
from path_planning_classes_3d.collision_check_utils_3d import points_in_balls_boxes


def get_block_env_configs(root_dir='.'):
    '''
    Outputs a list of dictionaries block_env_config.
    '''
    with open(join(root_dir, "data/block_gap/block_gap_configs.json"), 'r') as f:
        block_gap_configs = json.load(f)
    return block_gap_configs['block']

def get_gap_env_configs(root_dir='.'):
    '''
    Outputs a list of dictionaries gap_env_config.
    '''
    with open(join(root_dir, "data/block_gap/block_gap_configs.json"), 'r') as f:
        block_gap_configs = json.load(f)
    return block_gap_configs['gap']

def get_random_2d_env_configs(root_dir='.'):
    '''
    The random 2d world dataset has 4 pairs of start and goal for each obstacle map.
    We transform one obstacle map into 4 environments for evaluation.
    '''
    with open(join("data", "random_2d", "test", "envs.json"), 'r') as f:
        random_2d_map_list = json.load(f)
    env_config_list = []
    for map_idx, env_dict_per_map in enumerate(random_2d_map_list):
        for start_goal_pair_idx in range(len(env_dict_per_map['start'])):
            env_config = {}
            env_config['img_idx'] = map_idx
            env_config['start_goal_idx'] = start_goal_pair_idx
            env_config['env_dict'] = copy(env_dict_per_map)
            env_config['env_dict']['start'] = [env_dict_per_map['start'][start_goal_pair_idx]]
            env_config['env_dict']['goal'] = [env_dict_per_map['goal'][start_goal_pair_idx]]
            env_config_list.append(env_config)
    return env_config_list

def get_random_3d_env_configs(root_dir='.'):
    '''
    The random 3d world dataset.
    '''
    with open(join(root_dir, "data", "random_3d", "test", "envs.json"), 'r') as f:
        random_3d_map_list = json.load(f)
    env_config_list = []
    for map_idx, env_dict in enumerate(random_3d_map_list):
        env_config = {}
        env_config['img_idx'] = map_idx
        env_config['env_dict'] = copy(env_dict)
        env_config_list.append(env_config)
    return env_config_list


def get_random_3d_problem_input(random_3d_env_config):
    '''
    The last None is to match outputs of other get_problem_input functions.
    '''
    env_dict = random_3d_env_config['env_dict']
    x_start = tuple(env_dict['start'][0])
    x_goal = tuple(env_dict['goal'][0])
    problem = {}
    problem['x_start'] = x_start
    problem['x_goal'] = x_goal
    problem['env_dict'] = env_dict
    problem['env'] = Env(env_dict)
    problem['search_radius'] = compute_gamma_rrt_star_3d(problem['env'])
    return problem

def compute_gamma_rrt_star_3d(env):
    dim = 3
    unit_ball_vol = 4./3.*np.pi # 4./3.*np.pi*1**3
    free_vol = approximate_free_vol_3d(env)
    return math.ceil((2*(1+1./dim))**(1./dim)*(free_vol/unit_ball_vol)**(1./dim))

def approximate_free_vol_3d(env, n_points=100000):
    points = np.array((
        np.random.uniform(env.x_range[0], env.x_range[1], n_points),
        np.random.uniform(env.y_range[0], env.y_range[1], n_points),
        np.random.uniform(env.z_range[0], env.z_range[1], n_points),
    )).T # (n_points, 3)
    in_obs = points_in_balls_boxes(
        points,
        np.array(env.obs_ball).astype(np.float64),
        np.array(env.obs_box).astype(np.float64),
        clearance=0,
    ).astype(np.float64) # (n_points, )
    free_vol_approx_ratio = 1-np.mean(in_obs)
    free_vol = (env.x_range[1]-env.x_range[0])*(env.y_range[1]-env.y_range[0])*(env.z_range[1]-env.z_range[0])*free_vol_approx_ratio
    return free_vol