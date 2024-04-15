import json
import math
from copy import copy
from os.path import join

import cv2
import numpy as np

from path_planning_utils.rrt_env import Env
from datasets.point_cloud_mask_utils import get_binary_mask



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

def get_block_problem_input(block_env_config):
    w, d_goal, img_height, img_width, best_path_len = \
        block_env_config['w'], \
        block_env_config['d_goal'], \
        block_env_config['img_height'], \
        block_env_config['img_width'], \
        block_env_config['best_path_len']
    env_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    rectangle_obstacles = []
    rec_obs_x = img_width//2 - w//2
    rec_obs_y = img_height//2 - w//2
    rec_obs_w = w
    rec_obs_h = w
    rectangle_obstacles.append([rec_obs_x, rec_obs_y, rec_obs_w, rec_obs_h])
    cv2.rectangle(env_img, (rec_obs_x, rec_obs_y), (rec_obs_x + rec_obs_w, rec_obs_y + rec_obs_h), (0, 0, 0), -1)
    start_x = img_width//2 - d_goal//2
    start_y = img_height//2
    goal_x = img_width//2 + d_goal//2
    goal_y = img_height//2
    x_start = (start_x, start_y)
    x_goal = (goal_x, goal_y)
    binary_mask = get_binary_mask(env_img)
    env_dims = (img_height, img_width)
    circle_obstacles = []
    x_start_list = [x_start]
    x_goal_list = [x_goal]
    env_dict = {}
    env_dict['env_dims'] = env_dims
    env_dict['rectangle_obstacles'] = rectangle_obstacles
    env_dict['circle_obstacles'] = circle_obstacles
    env_dict['start'] = x_start_list
    env_dict['goal'] = x_goal_list
    problem = {}
    problem['x_start'] = x_start
    problem['x_goal'] = x_goal
    problem['env_dict'] = env_dict
    problem['env'] = Env(env_dict)
    problem['binary_mask'] = binary_mask
    problem['best_path_len'] = best_path_len
    problem['search_radius'] = compute_gamma_rrt_star(binary_mask, dim=2)
    return problem


def get_gap_problem_input(gap_env_config):
    h, t, h_g, y_g, d_goal, img_height, img_width, flank_path_len = \
        gap_env_config['h'], \
        gap_env_config['t'], \
        gap_env_config['h_g'], \
        gap_env_config['y_g'], \
        gap_env_config['d_goal'], \
        gap_env_config['img_height'], \
        gap_env_config['img_width'], \
        gap_env_config['flank_path_len']
    env_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    rectangle_obstacles = []
    rec_obs_x = img_width//2 - t//2
    rec_obs_y = img_height//2 - h//2
    rec_obs_w = t
    rec_obs_h = h-h_g-y_g
    rectangle_obstacles.append([rec_obs_x, rec_obs_y, rec_obs_w, rec_obs_h])
    cv2.rectangle(env_img, (rec_obs_x, rec_obs_y), (rec_obs_x + rec_obs_w, rec_obs_y + rec_obs_h), (0, 0, 0), -1)
    rec_obs_x = img_width//2 - t//2
    rec_obs_y = rec_obs_y + (h-y_g)
    rec_obs_w = t
    rec_obs_h = y_g
    rectangle_obstacles.append([rec_obs_x, rec_obs_y, rec_obs_w, rec_obs_h])
    cv2.rectangle(env_img, (rec_obs_x, rec_obs_y), (rec_obs_x + rec_obs_w, rec_obs_y + rec_obs_h), (0, 0, 0), -1)
    start_x = img_width//2 - d_goal//2
    start_y = img_height//2
    goal_x = img_width//2 + d_goal//2
    goal_y = img_height//2
    x_start = (start_x, start_y)
    x_goal = (goal_x, goal_y)
    binary_mask = get_binary_mask(env_img)
    env_dims = (img_height, img_width)
    circle_obstacles = []
    x_start_list = [x_start]
    x_goal_list = [x_goal]
    env_dict = {}
    env_dict['env_dims'] = env_dims
    env_dict['rectangle_obstacles'] = rectangle_obstacles
    env_dict['circle_obstacles'] = circle_obstacles
    env_dict['start'] = x_start_list
    env_dict['goal'] = x_goal_list
    
    problem = {}
    problem['x_start'] = x_start
    problem['x_goal'] = x_goal
    problem['env_dict'] = env_dict
    problem['env'] = Env(env_dict)
    problem['binary_mask'] = binary_mask
    problem['flank_path_len'] = flank_path_len
    problem['search_radius'] = compute_gamma_rrt_star(binary_mask, dim=2)
    return problem


def get_random_2d_problem_input(random_2d_env_config):
    '''
    The last None is to match outputs of other get_problem_input functions.
    '''
    env_img = cv2.imread(join("data", "random_2d", "test", "env_imgs", "{0}.png".format(random_2d_env_config['img_idx'])))
    binary_mask = get_binary_mask(env_img)
    env_dict = random_2d_env_config['env_dict']
    x_start = tuple(env_dict['start'][0])
    x_goal = tuple(env_dict['goal'][0])

    problem = {}
    problem['x_start'] = x_start
    problem['x_goal'] = x_goal
    problem['env_dict'] = env_dict
    problem['env'] = Env(env_dict)
    problem['binary_mask'] = binary_mask
    problem['search_radius'] = compute_gamma_rrt_star(binary_mask, dim=2)
    return problem

def compute_gamma_rrt_star(binary_mask, dim=2):
    free_vol = binary_mask.sum()
    if dim == 2:
        unit_ball_vol = np.pi # pi*1**2
    elif dim == 3:
        unit_ball_vol = 4./3.*np.pi # 4./3.*np.pi*1**3
    else:
        raise NotImplementedError
    return math.ceil((2*(1+1./dim))**(1./dim)*(free_vol/unit_ball_vol)**(1./dim))

