import numpy as np

from path_planning_utils.rrt_env import Env
from path_planning_classes.rrt_base_2d import RRTBase2D
from path_planning_classes.rrt_star_2d import RRTStar2D
from path_planning_classes.rrt_visualizer_2d import NRRTStarPNGVisualizer
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points, \
    generate_rectangle_point_cloud

class NRRTStarPNG2D(RRTStar2D):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env_dict,
        png_wrapper,
        binary_mask,
        clearance,
        pc_n_points,
        pc_over_sample_scale,
        pc_sample_rate,
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
            "NRRT*-PNG 2D",
        )
        self.png_wrapper = png_wrapper
        self.binary_mask = binary_mask
        self.pc_n_points = pc_n_points # * number of points in pc
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.visualizer = NRRTStarPNGVisualizer(self.x_start, self.x_goal, self.env)

    def init_pc(self):
        self.update_point_cloud()

    def planning(self, visualize=False):
        self.init_pc()
        RRTStar2D.planning(self, visualize)

    def generate_random_node(self):
        if np.random.random() < self.pc_sample_rate:
            return self.SamplePointCloud()
        else:
            return self.SampleFree()

    def SamplePointCloud(self):
        return self.path_point_cloud_pred[np.random.randint(0,len(self.path_point_cloud_pred))]

    def visualize(self, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "nrrt*-png 2D, iteration " + str(self.iter_max)
        if img_filename is None:
            img_filename = "nrrt*_png_2d_example.png"
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
        return RRTStar2D.planning_block_gap(self, path_len_threshold)

    def planning_random(
        self,
        iter_after_initial,
    ):
        self.init_pc()
        return RRTStar2D.planning_random(self, iter_after_initial)


def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    return NRRTStarPNG2D(
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
    )
