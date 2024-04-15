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

gap_results = {}
methods = ['rrt', 'irrt', 'nrrt_png', 'nrrt_gng', 'nrrt_png_c', 'nirrt_png', 'nirrt_png_c']
result_filenames = [
    'gap-rrt_star-none',
    'gap-irrt_star-none',
    'gap-nrrt_star-pointnet2',
    'gap-nrrt_star-unet',
    'gap-nrrt_star-c-bfs-pointnet2',
    'gap-nirrt_star-pointnet2',
    'gap-nirrt_star-c-bfs-pointnet2',
]
for i in range(len(result_filenames)):
    result_filenames[i] += '-'+str(random_dataset_len)
visualization_folderpath = join('visualization', 'evaluation')
os.makedirs(visualization_folderpath, exist_ok=True)
results_folderpath = 'results/evaluation/2d'
for method, result_filename in zip(methods, result_filenames):
    with open(join(results_folderpath, result_filename+'.pickle'), 'rb') as f:
        gap_results[method] = pickle.load(f)
print("2D gap results loaded for analysis.")

gap_env_keys = ['h', 't', 'h_g', 'y_g', 'd_goal', 'img_height', 'img_width', 'flank_path_len']
iter_max = 30000
invalid_results = {}
for method in methods:
    invalid_results[method] = []
for i in range(len(gap_results[methods[0]])):
    for method in methods:
        if len(gap_results[method][i]['result'])<iter_max:
            test_idx = -1
        else:
            test_idx = iter_max-1
        if gap_results[method][i]['result'][test_idx] > gap_results[method][i]['flank_path_len']:
            invalid_results[method].append(i)

path_len_pairs = {}
path_len = {}
for method in methods:
    path_len[method] = []
path_len_pairs['irrt-nirrt_png_c'] = []
for i in range(len(gap_results[methods[0]])):
    for method in methods:
        tmp_len = min(len(gap_results[method][i]['result']), iter_max)
        path_len[method].append(tmp_len)
    tmp_len_irrt = min(len(gap_results['irrt'][i]['result']), iter_max)
    tmp_len_nirrt_png_c = min(len(gap_results['nirrt_png_c'][i]['result']), iter_max)
    path_len_pairs['irrt-nirrt_png_c'].append([tmp_len_irrt, tmp_len_nirrt_png_c])

fig, ax = plt.subplots()
plt.subplots_adjust(right=0.7)
methods = ['rrt', 'irrt', 'nrrt_png', 'nrrt_gng', 'nrrt_png_c', 'nirrt_png', 'nirrt_png_c']
for method, plot_color, method_label in \
    zip(['rrt', 'irrt', 'nrrt_png', 'nirrt_png', 'nirrt_png_c', 'nrrt_png_c', 'nrrt_gng'],
        ['k', 'gray', 'C0', 'C1', 'C2', 'C4', 'C5'],
        ['RRT*', 'IRRT*', 'NRRT*-PNG',  'NIRRT*-PNG (F)', 'NIRRT*-PNG (FC)', 'NRRT*-PNG (C)', 'NRRT*-GNG']):  
    gap_width_list = []
    iter_method_mean = []
    iter_method_ee = []
    curr_len = 0
    for next_len in [100, 200, 300, 400, 500]:
        gap_width_list.append(gap_results[method][curr_len]['h_g'])
        mm, hh = mean_confidence_interval(path_len[method][curr_len:next_len], confidence=0.95)
        iter_method_mean.append(mm)
        iter_method_ee.append(hh)
        curr_len = next_len
    plt.plot(gap_width_list, iter_method_mean, c=plot_color, label=method_label)
    plt.errorbar(gap_width_list, iter_method_mean,
        yerr=iter_method_ee,
        color=plot_color,
        ecolor=plot_color,
        capsize=5,
        elinewidth=1,
        linestyle='None')
plt.yscale('log')
plt.ylim(100, 40000)
plt.xlim(7.5, 2.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig(join(visualization_folderpath,'gap_iter_to_find_passage_vs_gap_width.png'))


path_len_pairs['irrt-nirrt_png_c'] = np.array(path_len_pairs['irrt-nirrt_png_c'])
fig, ax = plt.subplots()
range_limit = 8000
plt.plot(range(0,range_limit+1), range(0,range_limit+1), color='gray', lw=1)
ax.scatter(path_len_pairs['irrt-nirrt_png_c'][:,1], path_len_pairs['irrt-nirrt_png_c'][:,0], s=5, c='k')
plt.xlabel('NIRRT*-PNG(FC)')
plt.ylabel('IRRT*')
plt.xlim(0, range_limit)
plt.ylim(0, range_limit)
plt.xticks([0, 2000, 4000, 6000, 8000])
plt.yticks([0, 2000, 4000, 6000, 8000])
plt.gca().set_aspect('equal', adjustable='box')
fig.savefig(join(visualization_folderpath,'gap_iter_scatter-nirrt_png_c-irrt.png'))
print("2D gap results analyzed and visualized. Images can be found in visualization/evaluation/.")