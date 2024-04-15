import os
import argparse

import cv2
import numpy as np

from path_planning_classes.rrt_visualizer_2d import RRTStarVisualizer
from datasets.planning_problem_utils_2d import get_random_2d_env_configs as get_env_configs
from datasets.planning_problem_utils_2d import get_random_2d_problem_input as get_problem_input


argparser = argparse.ArgumentParser()
argparser.add_argument('--visual_example_token', type=str, default=None, help='Visual example token, such as 100_0, where img_idx=100 and start_goal_idx=0.')

args = argparser.parse_args()

env_config_list = get_env_configs()
img_folderpath = "visualization/img_with_labels_2d/"
os.makedirs(img_folderpath, exist_ok=True)

for env_config in env_config_list:
    if args.visual_example_token is not None:
        if args.visual_example_token != str(env_config['img_idx'])+'_'+str(env_config['start_goal_idx']):
            continue
    problem = get_problem_input(env_config)
    visual_example_token = str(env_config['img_idx'])+'_'+str(env_config['start_goal_idx'])
    path = np.loadtxt("data/random_2d/test/astar_paths/"+visual_example_token+".txt", delimiter=',')
    path = path.astype(int)

    visualizer = RRTStarVisualizer(problem['x_start'], problem['x_goal'], problem['env'])
    figure_title = visual_example_token
    img_filename =  visual_example_token+"_plt.png"
    visualizer.plot_scene_path(path, figure_title, img_filename, img_folder=img_folderpath) # red start star, yellow goal star, red optimal path
    print(visual_example_token, " plotted.")
    visual_example_env_img = cv2.imread("data/random_2d/test/env_imgs/{0}.png".format(env_config['img_idx']))
    cv2.circle(visual_example_env_img, problem['x_start'], 5, (0,0,255), -1) # red start round mask
    cv2.circle(visual_example_env_img, problem['x_goal'], 5, (0,255,255), -1) # yellow goal round mask
    for pt1, pt2 in zip(path[:-1], path[1:]):
        cv2.line(visual_example_env_img, pt1, pt2, (255,0,0), 3) # blue optimal path with thickness of 3
    flipped_img = cv2.flip(visual_example_env_img, 0) # flipped vertically
    cv2.imwrite(os.path.join(img_folderpath, visual_example_token+"_pixel_vertically_flipped.png"), flipped_img)

    print(visual_example_token, " visualized.")
    if args.visual_example_token is not None:
        break