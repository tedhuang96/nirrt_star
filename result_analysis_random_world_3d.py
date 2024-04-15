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
methods = ['rrt', 'irrt', 'nrrt_png', 'nrrt_png_c', 'nirrt_png', 'nirrt_png_c']
result_filenames = [
    'random_3d-rrt_star-none',
    'random_3d-irrt_star-none',
    'random_3d-nrrt_star-pointnet2',
    'random_3d-nrrt_star-c-bfs-pointnet2',
    'random_3d-nirrt_star-pointnet2',
    'random_3d-nirrt_star-c-bfs-pointnet2',
]
for i in range(len(result_filenames)):
    result_filenames[i] += '-'+str(random_dataset_len)
visualization_folderpath = join('visualization', 'evaluation')
os.makedirs(visualization_folderpath, exist_ok=True)
results_folderpath = 'results/evaluation/3d'
for method, result_filename in zip(methods, result_filenames):
    with open(join(results_folderpath, result_filename+'.pickle'), 'rb') as f:
        random_results[method] = pickle.load(f)
print("3D random world results loaded for analysis.")

iter_after_initial_list = range(0, 3000+250, 250)
invalid_indices = []
for method in methods:
    for i in range(random_dataset_len):
        if len(np.where(np.array(random_results[method][i]['result'])<np.inf)[0])==0:
            invalid_indices.append(i)
invalid_indices_unique = np.unique(invalid_indices)

random_analysis = {}
for method in methods:
    random_analysis[method] = {}
    for iter_after_initial in iter_after_initial_list:
        random_analysis[method][iter_after_initial] = []
    for i in range(random_dataset_len):
        if i not in invalid_indices_unique:
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
    zip(['rrt', 'irrt', 'nrrt_png', 'nirrt_png', 'nirrt_png_c', 'nrrt_png_c'],
        ['k', 'gray', 'C0', 'C1', 'C2', 'C4'],
        ['RRT*', 'IRRT*', 'NRRT*-PNG',  'NIRRT*-PNG (F)', 'NIRRT*-PNG (FC)', 'NRRT*-PNG (C)']):
    # * too drastic drop in first 250 iterations after initial solution. Removed initial solution for clarity.
    plt.plot(iter_after_initial_list[1:], path_cost_mean[method][1:], c=plot_color, marker='.', linestyle='-', label=method_label)
plt.legend()
fig.savefig(join(visualization_folderpath,'random_3d_path_cost_ratio_results.png'))
print("3D random world results analyzed and visualized. Images can be found in visualization/evaluation/.")
