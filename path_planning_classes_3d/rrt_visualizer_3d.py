import os
from os.path import join, exists

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class RRTStarVisualizer3D:
    def __init__(self, x_start, x_goal, env):
        self.x_start, self.x_goal = x_start, x_goal
        self.env = env
        if len(self.env.obs_ball)>0:
            self.obs_ball = np.array(self.env.obs_ball).astype(np.float64)
        else:
            self.obs_ball = None
        if len(self.env.obs_box)>0:
            self.obs_box = np.array(self.env.obs_box).astype(np.float64)
        else:
            self.obs_box = None
        xmin, xmax = self.env.x_range
        ymin, ymax = self.env.y_range
        zmin, zmax = self.env.z_range
        self.boundary = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

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
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=90., azim=0.)
        plt.title(figure_title)
        if self.obs_ball is not None:
            self.draw_balls()
        if self.obs_box is not None:
            obs_blocks = np.copy(self.obs_box)
            obs_blocks[:,3:] = obs_blocks[:,:3]+obs_blocks[:,3:] # [x,y,z,x+w,y+h,z+d]
            self.draw_blocks(obs_blocks)
        self.draw_blocks(np.array([self.boundary]),alpha=0)
        self.plot_start_goal()
        # adjust the aspect ratio
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        self.ax.set_xlim3d([xmin, xmax])
        self.ax.set_ylim3d([ymin, ymax])
        self.ax.set_zlim3d([zmin, zmax])
        self.ax.axis('off')
        limits = np.array([getattr(self.ax, f'get_{axis}lim')() for axis in 'xyz'])
        self.ax.set_box_aspect(np.ptp(limits, axis = 1))

    @staticmethod
    def create_sphere(center, r, num=30):
        u = np.linspace(0,2* np.pi,num)
        v = np.linspace(0,np.pi,num)
        x = np.outer(np.cos(u),np.sin(v))
        y = np.outer(np.sin(u),np.sin(v))
        z = np.outer(np.ones(np.size(u)),np.cos(v))
        x, y, z = r*x + center[0], r*y + center[1], r*z + center[2]
        return (x,y,z)
    
    def draw_balls(self):
        for ball in self.obs_ball:
            (xs,ys,zs) = self.create_sphere(ball[:3],ball[-1])
            self.ax.plot_wireframe(xs, ys, zs, alpha=0.15,color="b")


    def draw_blocks(self, blocks ,color=None,alpha=0.15):
        '''
        drawing the blocks on the graph
        '''
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                    dtype='float')
        f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
        n = blocks.shape[0]
        d = blocks[:, 3:6] - blocks[:, :3]
        vl = np.zeros((8 * n, 3))
        fl = np.zeros((6 * n, 4), dtype='int64')
        for k in range(n):
            vl[k * 8:(k + 1) * 8, :] = v * d[k] + blocks[k, :3]
            fl[k * 6:(k + 1) * 6, :] = f + k * 8
        if type(self.ax) is Poly3DCollection:
            self.ax.set_verts(vl[fl])
        else:
            pc = Poly3DCollection(vl[fl], alpha=alpha, linewidths=1, edgecolors='k')
            pc.set_facecolor(color)
            h = self.ax.add_collection3d(pc)
            return h



    @staticmethod
    def plot_visited(vertices, vertex_parents, animation):
        if animation:
            count = 0
            for vertex_index, vertex_parent_index in enumerate(vertex_parents):
                count += 1
                plt.plot([vertices[vertex_index, 0], vertices[vertex_parent_index, 0]],\
                         [vertices[vertex_index, 1], vertices[vertex_parent_index, 1]],\
                         [vertices[vertex_index, 2], vertices[vertex_parent_index, 2]], "-g", lw=1)
                plt.gcf().canvas.mpl_connect('key_release_event',
                                                lambda event:
                                                [exit(0) if event.key == 'escape' else None])
                if count % 10 == 0:
                    plt.pause(0.001)
        else:
            for vertex_index, vertex_parent_index in enumerate(vertex_parents):
                plt.plot([vertices[vertex_index, 0], vertices[vertex_parent_index, 0]],\
                         [vertices[vertex_index, 1], vertices[vertex_parent_index, 1]],\
                         [vertices[vertex_index, 2], vertices[vertex_parent_index, 2]], "-g", lw=1)

    def plot_start_goal(self):
        self.ax.plot(self.x_start[0], self.x_start[1], self.x_start[2], 'r*', markersize=7, zorder=10)
        self.ax.plot(self.x_goal[0], self.x_goal[1], self.x_goal[2], 'y*', markersize=7, zorder=10)

    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot(path[:,0], path[:,1], path[:,2], '-r', linewidth=1, zorder=9)



class IRRTStarVisualizer3D(RRTStarVisualizer3D):
    def __init__(self, x_start, x_goal, env):
        super().__init__(x_start, x_goal, env)

    def animation(self, vertices, vertex_parents, path, figure_title,\
                  x_center, c_best, dist, C, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation=False)
        if c_best != np.inf:
            self.draw_ellipsoid(x_center, c_best, dist, C)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))

    def draw_ellipsoid(self, x_center, c_best, dist, C):
        '''
        x_center: (3,)
        '''
        c_max = c_best
        c_min = dist
        if c_max ** 2 - c_min ** 2<0:
            eps = 1e-6
        else:
            eps = 0
        r = np.zeros(3)
        r[0] = c_max /2
        for i in [1,2]:
            r[i] = np.sqrt(c_max**2-c_min**2+eps) / 2
        L = np.diag(r) # R3*3

        samples_x, samples_y, samples_z = self.create_sphere(np.zeros(3), 1, num=256)
        pts = np.array([samples_x, samples_y, samples_z]) # (3, 256, 256)

        pts_reshaped = pts.reshape(3,-1).T # (256*256, 3)
        pts_in_world_frame = np.dot(np.dot(C, L), pts_reshaped.T).T + x_center # (256*256, 3)
        pts_in_world_frame = pts_in_world_frame.T.reshape(3, pts.shape[1], pts.shape[2]) # (3, 256, 256)
        self.ax.plot_surface(pts_in_world_frame[0], pts_in_world_frame[1], pts_in_world_frame[2], alpha=0.1, color="C1")


class NRRTStarPNGVisualizer3D(RRTStarVisualizer3D):
    def __init__(self, x_start, x_goal, env, path_point_cloud_pred=None):
        super().__init__(x_start, x_goal, env)
        self.path_point_cloud_pred = path_point_cloud_pred

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred

    def animation(self, vertices, vertex_parents, path, figure_title, animation=False, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation)
        if self.path_point_cloud_pred is not None:
            self.plot_points(self.path_point_cloud_pred)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))

    def plot_points(self, points):
        self.ax.scatter(points[:,0], points[:,1], points[:,2], s=2, c='C1')


class NIRRTStarVisualizer3D(IRRTStarVisualizer3D):
    def __init__(self, x_start, x_goal, env, path_point_cloud_pred=None):
        super().__init__(x_start, x_goal, env)
        self.path_point_cloud_pred = path_point_cloud_pred

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred

    def animation(self, vertices, vertex_parents, path, figure_title,\
                  x_center, c_best, dist, C, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation=False)
        if self.path_point_cloud_pred is not None:
            self.plot_points(self.path_point_cloud_pred)
        if c_best != np.inf:
            self.draw_ellipsoid(x_center, c_best, dist, C)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))

    def plot_points(self, points):
        self.ax.scatter(points[:,0], points[:,1], points[:,2], s=2, c='C1')