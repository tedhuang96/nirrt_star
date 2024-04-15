import os
import pickle
import argparse
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument('--random_dataset_len', type=int, default=500)
args = argparser.parse_args()
random_dataset_len = args.random_dataset_len

random_results = {}
methods = ['rrt', 'irrt', 'nrrt_png', 'nrrt_gng', 'nrrt_png_c', 'nirrt_png', 'nirrt_png_c']
result_filenames = [
    'random_2d-rrt_star-none',
    'random_2d-irrt_star-none',
    'random_2d-nrrt_star-pointnet2',
    'random_2d-nrrt_star-unet',
    'random_2d-nrrt_star-c-bfs-pointnet2',
    'random_2d-nirrt_star-pointnet2',
    'random_2d-nirrt_star-c-bfs-pointnet2',
]
for i in range(len(result_filenames)):
    result_filenames[i] += '-'+str(random_dataset_len)
visualization_folderpath = join('visualization', 'evaluation')
os.makedirs(visualization_folderpath, exist_ok=True)
results_folderpath = 'results/evaluation/2d'
for method, result_filename in zip(methods, result_filenames):
    with open(join(results_folderpath, result_filename+'.pickle'), 'rb') as f:
        random_results[method] = pickle.load(f)
print("2D random world results loaded for analysis.")

iter_after_initial_list = range(0, 3000+250, 250)
random_analysis = {}
for method in methods:
    random_analysis[method] = {}
    for iter_after_initial in iter_after_initial_list:
        random_analysis[method][iter_after_initial] = []
    for i in range(random_dataset_len):
        if len(np.where(np.array(random_results[method][i]['result'])<np.inf)[0])==0:
            import pdb; pdb.set_trace()
        initial_idx = np.where(np.array(random_results[method][i]['result'])<np.inf)[0][0]
        initial_path_cost_rrt = random_results['rrt'][i]['result'][np.where(np.array(random_results['rrt'][i]['result'])<np.inf)[0][0]]
        for iter_after_initial in iter_after_initial_list:
            if initial_idx+iter_after_initial<len(random_results[method][i]['result']):
                random_analysis[method][iter_after_initial].append(random_results[method][i]['result'][initial_idx+iter_after_initial]/initial_path_cost_rrt)
            else:
                random_analysis[method][iter_after_initial].append(random_results[method][i]['result'][-1]/initial_path_cost_rrt)

path_cost_mean = {}
for method in methods:
    path_cost_mean[method] = []
    for iter_key in iter_after_initial_list:
        if type(iter_key)!=str:
            iter_key_str = str(iter_key)
        else:
            iter_key_str = iter_key
        path_cost_mean[method].append(np.mean(random_analysis[method][iter_key]))

fig, ax = plt.subplots()
for method, plot_color, method_label in \
    zip(['rrt', 'irrt', 'nrrt_png', 'nirrt_png', 'nirrt_png_c', 'nrrt_png_c', 'nrrt_gng'],
        ['k', 'gray', 'C0', 'C1', 'C2', 'C4', 'C5'],
        ['RRT*', 'IRRT*', 'NRRT*-PNG',  'NIRRT*-PNG (F)', 'NIRRT*-PNG (FC)', 'NRRT*-PNG (C)', 'NRRT*-GNG']):
    iters_method = []
    plt.plot(iter_after_initial_list, path_cost_mean[method], c=plot_color, marker='.', linestyle='-', label=method_label)
plt.legend()
fig.savefig(join(visualization_folderpath,'random_2d_path_cost_ratio_results.png'))

random_analysis = {}
for method in methods:
    random_analysis[method] = []
    for i in range(random_dataset_len):
        if len(np.where(np.array(random_results[method][i]['result'])<np.inf)[0])==0:
            import pdb; pdb.set_trace()
        initial_idx = np.where(np.array(random_results[method][i]['result'])<np.inf)[0][0]
        random_analysis[method].append(initial_idx)

fig, ax = plt.subplots()
range_limit = 2000
plt.plot(range(0,range_limit+1), range(0,range_limit+1), color='gray', lw=1)
ax.scatter(random_analysis['nirrt_png_c'],random_analysis['irrt'], s=5, c='k')
plt.xlabel('NIRRT*-PNG(FC)')
plt.ylabel('IRRT*')
plt.xlim(0, range_limit)
plt.ylim(0, range_limit)
plt.xticks([0, 500, 1000, 1500, 2000])
plt.yticks([0, 500, 1000, 1500, 2000])
plt.gca().set_aspect('equal', adjustable='box')
fig.savefig(join(visualization_folderpath,'random_2d_iter_scatter_irrt_png_connect-irrt.png'))
print("2D random world results analyzed and visualized. Images can be found in visualization/evaluation/.")



