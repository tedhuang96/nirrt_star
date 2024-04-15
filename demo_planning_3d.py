import argparse
from importlib import import_module

import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='rrt_star', 
        help='rrt_star, irrt_star, nrrt_star, nirrt_star')
    parser.add_argument('-n', '--neural_net', default='none', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='none', help='none, bfs, astar')
    parser.add_argument('--device', default='cuda', help='cuda, cpu')

    parser.add_argument('--step_len', type=float, default=10)
    parser.add_argument('--iter_max', type=int, default=50000)
    parser.add_argument('--clearance', type=float, default=2, help='2 for random_3d.')
    parser.add_argument('--pc_n_points', type=int, default=2048)
    parser.add_argument('--pc_over_sample_scale', type=int, default=5)
    parser.add_argument('--pc_sample_rate', type=float, default=0.5)
    parser.add_argument('--pc_update_cost_ratio', type=float, default=1.0)
    parser.add_argument('--connect_max_trial_attempts', type=int, default=5)

    parser.add_argument('--problem', default='random_3d', help='only random_3d is available.')
    parser.add_argument('--result_folderpath', default='results')
    parser.add_argument('--path_len_threshold_percentage', type=float, default=0.02, help='block use only.')
    parser.add_argument('--iter_after_initial', type=int, default=1000, help='random_2d use only.')
    
    return parser.parse_args()



args = arg_parse()
# * sanity check
if args.path_planner == 'rrt_star' or args.path_planner == 'irrt_star':
    assert args.neural_net == 'none'
else:
    assert args.neural_net != 'none'
#  * set get_path_planner
if args.neural_net == 'none':
    path_planner_name = args.path_planner
elif args.neural_net == 'pointnet2' or args.neural_net == 'pointnet':
    path_planner_name = args.path_planner+'_png'
elif args.neural_net == 'unet':
    path_planner_name = args.path_planner+'_gng'
else:
    raise NotImplementedError
if args.connect != 'none':
    path_planner_name = path_planner_name+'_c'
path_planner_name = path_planner_name+'_3d'
get_path_planner = getattr(import_module('path_planning_classes_3d.'+path_planner_name), 'get_path_planner')
#  * set NeuralWrapper
if args.neural_net == 'none':
    NeuralWrapper = None
elif args.neural_net == 'pointnet2' or args.neural_net == 'pointnet':
    neural_wrapper_name = args.neural_net+'_wrapper'
    if args.connect != 'none':
        neural_wrapper_name = neural_wrapper_name+'_connect_'+args.connect
    NeuralWrapper = getattr(import_module('wrapper_3d.pointnet_pointnet2.'+neural_wrapper_name), 'PNGWrapper')
else:
    raise NotImplementedError
#  * set planning problem
if args.problem != 'random_3d':
    raise NotImplementedError
get_env_configs = getattr(import_module('datasets_3d.planning_problem_utils_3d'), 'get_'+args.problem+'_env_configs')
get_problem_input = getattr(import_module('datasets_3d.planning_problem_utils_3d'), 'get_'+args.problem+'_problem_input')

# * main
if NeuralWrapper is None:
    neural_wrapper = None
else:
    neural_wrapper = NeuralWrapper(
        device=args.device,
    )
if args.problem == 'random_3d':
    args.clearance = 2
print(args)
env_config_list = get_env_configs()
env_config_index = np.random.randint(len(env_config_list))
print("env_config_index: ", env_config_index)
env_config = env_config_list[env_config_index]
problem = get_problem_input(env_config)
path_planner = get_path_planner(
    args,
    problem,
    neural_wrapper,
)
path_planner.planning(visualize=True) # * we can only run planning once, or we need reset.