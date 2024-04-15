import numpy as np

from path_planning_utils.rrt_env import Env
from path_planning_classes.rrt_base_2d import RRTBase2D
from path_planning_classes.nrrt_star_png_2d import NRRTStarPNG2D
from path_planning_classes.rrt_visualizer_2d import NRRTStarPNGVisualizer
from datasets.point_cloud_mask_utils import generate_rectangle_point_cloud

class NRRTStarPNGC2D(NRRTStarPNG2D):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env_dict,
        png_wrapper_connect,
        binary_mask,
        clearance,
        pc_n_points,
        pc_over_sample_scale,
        pc_sample_rate,
        connect_max_trial_attempts,
    ):
        RRTBase2D.__init__(
            self,
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            Env(env_dict),
            clearance,
            "NRRT*-PNG(C) 2D",
        )
        self.png_wrapper = png_wrapper_connect
        self.binary_mask = binary_mask
        self.pc_n_points = pc_n_points # * number of points in pc
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.env_dict = env_dict
        self.connect_max_trial_attempts = connect_max_trial_attempts
        self.visualizer = NRRTStarPNGVisualizer(self.x_start, self.x_goal, self.env)

    def visualize(self, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "nrrt*-png(c) 2D, iteration " + str(self.iter_max)
        if img_filename is None:
            img_filename = "nrrt*_png_c_2d_example.png"
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
        pc = generate_rectangle_point_cloud(
            self.binary_mask,
            self.pc_n_points,
            self.pc_over_sample_scale,
        )
        _, _, path_pred = self.png_wrapper.generate_connected_path_points(
            pc.astype(np.float32),
            self.x_start,
            self.x_goal,
            self.env_dict,
            neighbor_radius=self.pc_neighbor_radius,
            max_trial_attempts=self.connect_max_trial_attempts,
        )
        self.path_point_cloud_pred = pc[path_pred.nonzero()[0]] # (<pc_n_points, 2)
        self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)

def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    return NRRTStarPNGC2D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env_dict'],
        neural_wrapper,
        problem['binary_mask'],
        args.clearance,
        args.pc_n_points,
        args.pc_over_sample_scale,
        args.pc_sample_rate,
        args.connect_max_trial_attempts,
    )