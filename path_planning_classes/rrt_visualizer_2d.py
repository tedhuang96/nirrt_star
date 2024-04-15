import os
import math
from os.path import join, exists

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as Rot


class RRTStarVisualizer:
    def __init__(self, x_start, x_goal, env):
        self.x_start, self.x_goal = x_start, x_goal
        self.env = env
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle

    def animation(self, vertices, vertex_parents, path, figure_title, animation=False, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))
    
    def plot_scene_path(self, path, figure_title, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))

    def plot_grid(self, figure_title):
        self.fig, self.ax = plt.subplots()
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, w, h) in self.obs_bound:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )
        self.plot_start_goal()
        plt.title(figure_title)
        plt.axis("equal")

    @staticmethod
    def plot_visited(vertices, vertex_parents, animation):
        if animation:
            count = 0
            for vertex_index, vertex_parent_index in enumerate(vertex_parents):
                count += 1
                plt.plot([vertices[vertex_index, 0], vertices[vertex_parent_index, 0]],\
                         [vertices[vertex_index, 1], vertices[vertex_parent_index, 1]], "-g", lw=1)
                plt.gcf().canvas.mpl_connect('key_release_event',
                                                lambda event:
                                                [exit(0) if event.key == 'escape' else None])
                if count % 10 == 0:
                    plt.pause(0.001)
        else:
            for vertex_index, vertex_parent_index in enumerate(vertex_parents):
                plt.plot([vertices[vertex_index, 0], vertices[vertex_parent_index, 0]],\
                         [vertices[vertex_index, 1], vertices[vertex_parent_index, 1]], "-g", lw=1)

    def plot_start_goal(self):
        plt.scatter(self.x_start[0], self.x_start[1], s=30, c='r', marker='*', zorder=10)
        plt.scatter(self.x_goal[0], self.x_goal[1], s=30, c='y', marker='*', zorder=10)

    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot(path[:,0], path[:,1], '-r', linewidth=1, zorder=9)



class IRRTStarVisualizer(RRTStarVisualizer):
    def __init__(self, x_start, x_goal, env):
        super().__init__(x_start, x_goal, env)

    def animation(self, vertices, vertex_parents, path, figure_title,\
                  x_center, c_best, dist, theta, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation=False)
        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))

    @staticmethod
    def draw_ellipse(x_center, c_best, dist, theta):
        if c_best ** 2 - dist ** 2<0:
            eps = 1e-6
        else:
            eps = 0
        a = math.sqrt(c_best ** 2 - dist ** 2+eps) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(px, py, linestyle='--', color='k', linewidth=1)


class NRRTStarPNGVisualizer(RRTStarVisualizer):
    def __init__(self, x_start, x_goal, env, path_point_cloud_pred=None):
        super().__init__(x_start, x_goal, env)
        self.path_point_cloud_pred = path_point_cloud_pred

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred

    def animation(self, vertices, vertex_parents, path, figure_title, animation=False, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation)
        if self.path_point_cloud_pred is not None:
            plt.scatter(self.path_point_cloud_pred[:,0], self.path_point_cloud_pred[:,1], s=2, c='C1')
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))


class NIRRTStarVisualizer(IRRTStarVisualizer):
    def __init__(self, x_start, x_goal, env, path_point_cloud_pred=None):
        super().__init__(x_start, x_goal, env)
        self.path_point_cloud_pred = path_point_cloud_pred
        self.path_point_cloud_other = None

    def set_path_point_cloud_other(self, path_point_cloud_other):
        self.path_point_cloud_other = path_point_cloud_other

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred

    def animation(self, vertices, vertex_parents, path, figure_title,\
                  x_center, c_best, dist, theta, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        # self.plot_visited(vertices, vertex_parents, animation=False)
        if self.path_point_cloud_pred is not None:
            plt.scatter(self.path_point_cloud_pred[:,0], self.path_point_cloud_pred[:,1], s=2, c='C1')
        if self.path_point_cloud_other is not None:
            plt.scatter(self.path_point_cloud_other[:,0], self.path_point_cloud_other[:,1], s=2, c='C0')
        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))

class NRRTStarGNGVisualizer(RRTStarVisualizer):
    def __init__(self, x_start, x_goal, env, path_point_cloud_pred=None, img_path_score=None):
        super().__init__(x_start, x_goal, env)
        self.path_point_cloud_pred = path_point_cloud_pred
        self.img_path_score = img_path_score

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred

    def set_img_path_score(self, img_path_score):
        self.img_path_score = img_path_score

    def plot_prob_heatmap(self):
        if self.img_path_score is not None:
            self.ax.imshow(self.img_path_score, cmap='viridis', zorder=0)

    def animation(self, vertices, vertex_parents, path, figure_title, animation=False, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_prob_heatmap()
        self.plot_visited(vertices, vertex_parents, animation)
        # if self.path_point_cloud_pred is not None:
        #     plt.scatter(self.path_point_cloud_pred[:,0], self.path_point_cloud_pred[:,1], s=2, c='C1')
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))