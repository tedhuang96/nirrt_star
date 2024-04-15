import numpy as np

from path_planning_utils_3d.rrt_env_3d import Env
from path_planning_classes_3d.rrt_base_3d import RRTBase3D
from path_planning_classes_3d.rrt_star_3d import RRTStar3D
from path_planning_classes_3d.rrt_visualizer_3d import NRRTStarPNGVisualizer3D
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points

from datasets_3d.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d

class NRRTStarPNG3D(RRTStar3D):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env_dict,
        png_wrapper,
        clearance,
        pc_n_points,
        pc_over_sample_scale,
        pc_sample_rate,
    ):
        RRTBase3D.__init__(
            self,
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            Env(env_dict),
            clearance,
            "NRRT*-PNG 3D",
        )
        self.png_wrapper = png_wrapper
        self.pc_n_points = pc_n_points # * number of points in pc
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.visualizer = NRRTStarPNGVisualizer3D(self.x_start, self.x_goal, self.env)

    def init_pc(self):
        self.update_point_cloud()

    def planning(self, visualize=False):
        self.init_pc()
        RRTStar3D.planning(self, visualize)

    def generate_random_node(self):
        if np.random.random() < self.pc_sample_rate:
            return self.SamplePointCloud()
        else:
            return self.SampleFree()

    def SamplePointCloud(self):
        return self.path_point_cloud_pred[np.random.randint(0,len(self.path_point_cloud_pred))]

    def visualize(self, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "nrrt*-png 3D, iteration " + str(self.iter_max)
        # if img_filename is None:
        #     img_filename = "nrrt*_png_3d_example.png"
        self.visualizer.animation(
            self.vertices[:self.num_vertices],
            self.vertex_parents[:self.num_vertices],
            self.path,
            figure_title,
            animation=False,
            img_filename=img_filename)
    
    def update_point_cloud(self):
        if self.pc_sample_rate == 0:
            self.path_point_cloud_pred = None
            self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
            return
        pc = generate_rectangle_point_cloud_3d(
            self.env,
            self.pc_n_points,
            over_sample_scale=self.pc_over_sample_scale,
        )
        start_mask = get_point_cloud_mask_around_points(
            pc,
            self.x_start[np.newaxis,:],
            self.pc_neighbor_radius,
        ) # (n_points,)
        goal_mask = get_point_cloud_mask_around_points(
            pc,
            self.x_goal[np.newaxis,:],
            self.pc_neighbor_radius,
        ) # (n_points,)
        path_pred, path_score = self.png_wrapper.classify_path_points(
            pc.astype(np.float32),
            start_mask.astype(np.float32),
            goal_mask.astype(np.float32),
        )
        self.path_point_cloud_pred = pc[path_pred.nonzero()[0]] # (<pc_n_points, 2)
        self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)

    def planning_block_gap(
        self,
        path_len_threshold,
    ):
        self.init_pc()
        return RRTStar3D.planning_block_gap(self, path_len_threshold)

    def planning_random(
        self,
        iter_after_initial,
    ):
        self.init_pc()
        return RRTStar3D.planning_random(self, iter_after_initial)


def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    return NRRTStarPNG3D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env_dict'],
        neural_wrapper,
        args.clearance,
        args.pc_n_points,
        args.pc_over_sample_scale,
        args.pc_sample_rate,
    )
