import json
import time
from os import makedirs
from os.path import join

import yaml
import numpy as np

from path_planning_utils_3d.env_3d import Env
from path_planning_utils_3d.Astar_3d import Weighted_A_star
from path_planning_utils_3d.collision_check_utils import points_in_AABB_3d, points_in_ball_3d



def generate_env_3d(
    xyz_max,
    box_size_range,
    ball_radius_range,
    num_boxes_range,
    num_balls_range,
):
    """
    - inputs
        - xyz_max: tuple, (xmax, ymax, zmax)
        - box_size_range: list, (min, max)
        - ball_radius_range: list, (min, max)
        - num_boxes_range: list, (min, max)
        - num_balls_range: list, (min, max)
    - outputs
        - env_dims: tuple (xmax, ymax, zmax)
        - box_obstacles: list of [x,y,z,w,h,d]
        - ball_obstacles: list of [x,y,z,r]
    """
    xmax, ymax, zmax = xyz_max
    env_dims = xyz_max
    num_boxes = np.random.randint(num_boxes_range[0], num_boxes_range[1])
    num_balls = np.random.randint(num_balls_range[0], num_balls_range[1])
    box_obstacles = []
    ball_obstacles = []

    for i in range(num_boxes):
        not_in_env_3d = True
        while not_in_env_3d:
            x = np.random.randint(0, xmax)
            y = np.random.randint(0, ymax)
            z = np.random.randint(0, zmax)
            w = np.random.randint(box_size_range[0], box_size_range[1])
            h = np.random.randint(box_size_range[0], box_size_range[1])
            d = np.random.randint(box_size_range[0], box_size_range[1])
            if 0 <= x < xmax-w and 0 <= y < ymax-h and 0 <= z < zmax-d:
                not_in_env_3d = False
        box_obstacles.append([x,y,z,w,h,d])

    for i in range(num_balls):
        not_in_env_3d = True
        while not_in_env_3d:
            x = np.random.randint(0, xmax)
            y = np.random.randint(0, ymax)
            z = np.random.randint(0, zmax)
            r = np.random.randint(ball_radius_range[0], ball_radius_range[1])
            if r < x < xmax-r and r < y < ymax-r and r < z < zmax-r:
                not_in_env_3d = False
        ball_obstacles.append([x,y,z,r])
    
    return env_dims, box_obstacles, ball_obstacles


def generate_start_goal_points_3d(env, distance_lower_limit=50, max_attempt_count=100):
    attempt_count = 0
    while True:
        start_goal = np.random.randint(
            low=env.boundary[:3],
            high=env.boundary[3:],
            size=(2, 3),
        )
        start, goal = start_goal[0], start_goal[1]
        if ((start-goal)**2).sum() > distance_lower_limit**2:
            start_goal_in_aabb = points_in_AABB_3d(start_goal, env.box_obstacles, clearance=env.clearance)
            start_goal_in_ball = points_in_ball_3d(start_goal, env.balls_no_clearance, clearance=env.clearance)
            # 1 in obstacle, 0 not in obstacle # (2,) (2,)
            # 0,0
            # 0,0
            if (start_goal_in_aabb+start_goal_in_ball).sum()==0:
                return tuple(start.tolist()), tuple(goal.tolist())
        attempt_count += 1
        if attempt_count>max_attempt_count:
            return None, None

def generate_astar_path(
    env,
):
    start_time = time.time()
    astar = Weighted_A_star(env)
    success =  astar.run()
    if success:
        path = astar.get_path_solution()
        path_success = astar.check_success(path)
        if path_success:
            exec_time = time.time() - start_time
            return path, exec_time
        else:
            exec_time = time.time() - start_time
            return None, exec_time
    else:
        exec_time = time.time() - start_time
        return None, exec_time

config_name = "random_3d"
with open(join("env_configs", config_name+".yml"), 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
dataset_dir = join("data", config_name)

random_seed = config['random_seed']
xyz_max = config['xyz_max']
box_size_range = config['box_size_range']
ball_radius_range = config['ball_radius_range']
num_boxes_range = config['num_boxes_range']
num_balls_range = config['num_balls_range']
astar_resolution = config['astar_resolution']
path_clearance = config['path_clearance']
start_goal_dim_distance_limit = config['start_goal_dim_distance_limit']
start_goal_sampling_attempt_count = config['start_goal_sampling_attempt_count']
num_samples_per_env = config['num_samples_per_env']
redundant_env_size_scale = config['redundant_env_size_scale']

np.random.seed(random_seed)

env_size = {}
for mode in ['train', 'val', 'test']:
    env_size[mode] = config[mode+'_env_size']

total_env_count = 0
invalid_env_count = 0
for mode in ['train', 'val', 'test']:
    mode_dir = join("data", config_name, mode)
    mode_path_dir = join(mode_dir, "astar_paths")
    makedirs(mode_path_dir, exist_ok=True)
    mode_env_list = []
    while len(mode_env_list) < env_size[mode]*redundant_env_size_scale:
        total_env_count += 1
        env_dims, box_obstacles, ball_obstacles = generate_env_3d(
            xyz_max,
            box_size_range,
            ball_radius_range,
            num_boxes_range,
            num_balls_range,
        )
        env = Env(
            env_dims,
            box_obstacles,
            ball_obstacles,
            clearance=path_clearance,
            resolution=astar_resolution,
        )
        valid_env = True
        x_start_list, x_goal_list, path_list, exec_time_list = [], [], [], []
        for _ in range(num_samples_per_env):
            x_start, x_goal = generate_start_goal_points_3d(
                env,
                distance_lower_limit=start_goal_dim_distance_limit,
                max_attempt_count=start_goal_sampling_attempt_count,
            )
            if x_start is None:
                valid_env = False
                break
            env.set_start_goal(x_start, x_goal)
            x_start_list.append(list(x_start))
            x_goal_list.append(list(x_goal))
        if not valid_env:
            invalid_env_count += 1
            print("Invalid env: {0}/{1}".format(invalid_env_count, total_env_count))
            continue
        env_dict = {}
        env_dict['env_dims'] = env_dims
        env_dict['box_obstacles'] = box_obstacles
        env_dict['ball_obstacles'] = ball_obstacles
        env_dict['start'] = x_start_list
        env_dict['goal'] = x_goal_list
        mode_env_list.append(env_dict)
        env_idx = len(mode_env_list)-1
        with open(join(mode_dir, "raw_envs.json"), "w") as f:
            json.dump(mode_env_list, f)
        if len(mode_env_list) % 100 == 0:
            print(str(len(mode_env_list))+' '+mode+' envs and '+\
                str(num_samples_per_env*len(mode_env_list))+' samples are saved.')