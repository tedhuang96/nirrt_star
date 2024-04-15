import numpy as np

from path_planning_utils_3d.rrt_env_3d import Env
from path_planning_classes_3d.rrt_base_3d import RRTBase3D
from path_planning_classes_3d.nirrt_star_png_3d import NIRRTStarPNG3D
from path_planning_classes_3d.rrt_visualizer_3d import NIRRTStarVisualizer3D
from datasets_3d.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d, \
    ellipsoid_point_cloud_sampling_3d


class NIRRTStarPNGC3D(NIRRTStarPNG3D):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env_dict,
        png_wrapper_connect,
        clearance,
        pc_n_points,
        pc_over_sample_scale,
        pc_sample_rate,
        pc_update_cost_ratio,
        connect_max_trial_attempts,
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
            "NIRRT*-PNG(C) 3D",
        )
        self.png_wrapper = png_wrapper_connect
        self.pc_n_points = pc_n_points # * number of points in pc
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.pc_update_cost_ratio = pc_update_cost_ratio
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.env_dict = env_dict
        self.connect_max_trial_attempts = connect_max_trial_attempts
        self.visualizer = NIRRTStarVisualizer3D(self.x_start, self.x_goal, self.env)

    def update_point_cloud(
        self,
        cmax,
        cmin,
    ):
        if self.pc_sample_rate == 0:
            self.path_point_cloud_pred = None
            self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
            return
        if cmax < np.inf:
            max_min_ratio = cmax/cmin
            pc = ellipsoid_point_cloud_sampling_3d(
                self.x_start,
                self.x_goal,
                max_min_ratio,
                self.env,
                self.pc_n_points,
                n_raw_samples=self.pc_n_points*self.pc_over_sample_scale,
            )
        else:
            pc = generate_rectangle_point_cloud_3d(
                self.env,
                self.pc_n_points,
                over_sample_scale=self.pc_over_sample_scale,
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

    def visualize(self, x_center, c_best, start_goal_straightline_dist, C, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "nirrt*(c) 3D, iteration " + str(self.iter_max)
        # if img_filename is None:
        #     img_filename="nirrt*_c_3d_example.png"
        self.visualizer.animation(
            self.vertices[:self.num_vertices],
            self.vertex_parents[:self.num_vertices],
            self.path,
            figure_title,
            x_center,
            c_best,
            start_goal_straightline_dist,
            C,
            img_filename=img_filename,
        )

def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    return NIRRTStarPNGC3D(
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
        args.pc_update_cost_ratio,
        args.connect_max_trial_attempts,
    )