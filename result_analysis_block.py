import os
import pickle
import argparse
from os.path import join

import scipy
import numpy as np
import matplotlib.pyplot as plt


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

argparser = argparse.ArgumentParser()
argparser.add_argument('--random_dataset_len', type=int, default=500)
args = argparser.parse_args()
random_dataset_len = args.random_dataset_len

block_results = {}
block_env_keys = ['w', 'd_goal', 'img_height', 'img_width', 'best_path_len']
methods = ['rrt', 'irrt', 'nrrt_png', 'nrrt_png_c', 'nirrt_png', 'nirrt_png_c']
result_filenames = [
    'block-rrt_star-none',
    'block-irrt_star-none',
    'block-nrrt_star-pointnet2',
    'block-nrrt_star-c-bfs-pointnet2',
    'block-nirrt_star-pointnet2',
    'block-nirrt_star-c-bfs-pointnet2',
]
for i in range(len(result_filenames)):
    result_filenames[i] += '-'+str(random_dataset_len)
visualization_folderpath = join('visualization', 'evaluation')
os.makedirs(visualization_folderpath, exist_ok=True)
results_folderpath = 'results/evaluation/2d'
for method, result_filename in zip(methods, result_filenames):
    with open(join(results_folderpath, result_filename+'.pickle'), 'rb') as f:
        block_results[method] = pickle.load(f)
print("2D block results loaded for analysis.")

data_len = len(block_results['rrt'])
for i in range(data_len):
    for k in block_env_keys:
        values = set()
        for method in methods:
            values.add(block_results[method][i][k])
        if len(values)!=1:
            raise RuntimeError
            # * check env config matches
block_analysis_configs = {}
close_to_optimal = 0.02
block_analysis_configs[close_to_optimal] = {}
iter_upperlimit = 30000

block_analysis_close_to_optimal = {}
close_to_optimal_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
for close_to_optimal in close_to_optimal_list:
    block_analysis_close_to_optimal[close_to_optimal] = {}
    for method in methods:
        block_analysis_close_to_optimal[close_to_optimal][method] = []
        for i in range(500):
            if len(block_results[method][i]['result']) > iter_upperlimit:
                test_idx = iter_upperlimit-1
            else:
                test_idx = -1
            if block_results[method][i]['result'][test_idx] > (1+close_to_optimal)*block_results[method][i]['best_path_len']:
                block_analysis_close_to_optimal[close_to_optimal][method].append(iter_upperlimit)
            else:
                iter_run = np.where(np.array(block_results[method][i]['result'])<(1+close_to_optimal)*block_results[method][i]['best_path_len'])[0][0]
                block_analysis_close_to_optimal[close_to_optimal][method].append(iter_run)
close_to_optimal_list.reverse()
fig, ax = plt.subplots()
plt.subplots_adjust(right=0.7)
for method, plot_color, method_label in \
    zip(['rrt', 'irrt', 'nrrt_png', 'nirrt_png', 'nirrt_png_c', 'nrrt_png_c'],
        ['k', 'gray', 'C0', 'C1', 'C2', 'C4'],
        ['RRT*', 'IRRT*', 'NRRT*-PNG',  'NIRRT*-PNG (F)', 'NIRRT*-PNG (FC)', 'NRRT*-PNG (C)']):
    iters_method_mm = []
    iters_method_ee = []
    for close_to_optimal in close_to_optimal_list:
        mm, hh = mean_confidence_interval(block_analysis_close_to_optimal[close_to_optimal][method])
        iters_method_mm.append(mm)
        iters_method_ee.append(hh)
    plt.plot(np.array(close_to_optimal_list)*100, iters_method_mm, c=plot_color, label=method_label)
    plt.errorbar(np.array(close_to_optimal_list)*100, iters_method_mm,
        yerr=iters_method_ee,
        color=plot_color,
        ecolor=plot_color,
        capsize=5,
        elinewidth=1,
        linestyle='None')
    
plt.yscale('log')
plt.ylim(100, 30000)
plt.xlim(10.5, 1.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig(join(visualization_folderpath,'block_close_to_optimal_10_to_2.png'))


fig, ax = plt.subplots()
plt.subplots_adjust(right=0.7)
for method, plot_color, method_label in \
    zip(['rrt', 'irrt', 'nrrt_png', 'nirrt_png', 'nirrt_png_c', 'nrrt_png_c'],
        ['k', 'gray', 'C0', 'C1', 'C2', 'C4'],
        ['RRT*', 'IRRT*', 'NRRT*-PNG',  'NIRRT*-PNG (F)', 'NIRRT*-PNG (FC)', 'NRRT*-PNG (C)']):
    map_dgoal_ratio_list = []
    iters_method_mm = []
    iters_method_ee = []
    for i in [0, 100, 200, 300, 400]:
        map_dgoal_ratio_list.append(block_results[method][i]['img_width']/block_results[method][i]['d_goal'])
        mm, hh = mean_confidence_interval(block_analysis_close_to_optimal[0.02][method][i:i+100])
        iters_method_mm.append(mm)
        iters_method_ee.append(hh)
    plt.plot(map_dgoal_ratio_list, iters_method_mm, c=plot_color, label=method_label)
    plt.errorbar(map_dgoal_ratio_list, iters_method_mm,
        yerr=iters_method_ee,
        color=plot_color,
        ecolor=plot_color,
        capsize=5,
        elinewidth=1,
        linestyle='None')
    
plt.yscale('log')
plt.xlim(1.5, 6.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig(join(visualization_folderpath,'block_close_to_optimal_2_different_map_dist_ratio.png'))

range_limit = 6000
result_pairs = {}
close_to_optimal = 0.02

pair_key = 'irrt-irrt_png_connect-'+str(close_to_optimal)
result_pairs[pair_key] = []

for i in range(500):
    result_pairs[pair_key].append(
        [block_analysis_close_to_optimal[close_to_optimal]['irrt'][i],
        block_analysis_close_to_optimal[close_to_optimal]['nirrt_png_c'][i]]
    )

result_pairs[pair_key] = np.array(result_pairs[pair_key])
fig, ax = plt.subplots()
plt.plot(range(0,range_limit+1), range(0,range_limit+1), color='gray', lw=1)
ax.scatter(result_pairs[pair_key][:,1], result_pairs[pair_key][:,0], s=5, c='k')
plt.xlabel('NIRRT*-PNG(FC)')
plt.ylabel('IRRT*')
plt.xlim(0, range_limit)
plt.ylim(0, range_limit)
plt.xticks([0, 2000, 4000, 6000])
plt.yticks([0, 2000, 4000, 6000])

plt.gca().set_aspect('equal', adjustable='box')
fig.savefig(join(visualization_folderpath,'block_'+pair_key+'.png'), bbox_inches='tight', pad_inches=0)
print("2D block results analyzed and visualized. Images can be found in visualization/evaluation/.")