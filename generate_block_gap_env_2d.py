import os
import json
from os.path import join

import numpy as np


block_gap_configs = {}
block_gap_configs['block'] = []
block_gap_configs['gap'] = []

num_envs = 100
d_goal = 60
block_widths = np.random.randint(10, 50, num_envs)

for l_dgoal_ratio in [2,3,4,5,6]:
    img_height, img_width = d_goal*l_dgoal_ratio, d_goal*l_dgoal_ratio
    for block_width in block_widths:
        best_path_len = block_width+(((d_goal-block_width)//2)**2+(block_width//2)**2)**0.5+\
            (((d_goal-block_width)-(d_goal-block_width)//2)**2+(block_width//2)**2)**0.5
        block_env_config = {
            'w': int(block_width),
            'd_goal': d_goal,
            'img_height': img_height,
            'img_width': img_width,
            'best_path_len': best_path_len,
        }
        block_gap_configs['block'].append(block_env_config)

num_envs = 100
img_height, img_width = 224, 224
h = 90
t = 20
d_goal = 60
flank_path_len = t+2*(((d_goal-t)/2)**2+(h/2)**2)**0.5
for h_g in [7,6,5,4,3]:
    for y_g in np.random.randint(20, 70, num_envs):
        gap_env_config = {
            'h': h,
            't': t, 
            'h_g': h_g,
            'y_g': int(y_g),
            'd_goal': d_goal,
            'img_height': img_height,
            'img_width': img_width,
            'flank_path_len': flank_path_len,
        }
        block_gap_configs['gap'].append(gap_env_config)

folder_path = "data/block_gap"
os.makedirs(folder_path, exist_ok=True)
with open(join(folder_path, "block_gap_configs.json"), "w") as f:
    json.dump(block_gap_configs, f)