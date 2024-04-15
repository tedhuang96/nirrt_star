import numpy as np

from path_planning_utils_3d.rrt_env_3d import Env
from path_planning_classes_3d.rrt_base_3d import RRTBase3D
from path_planning_classes_3d.irrt_star_3d import IRRTStar3D
from path_planning_classes_3d.rrt_visualizer_3d import NIRRTStarVisualizer3D
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points
from datasets_3d.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d, \
    ellipsoid_point_cloud_sampling_3d



class NIRRTStarPNG3D(IRRTStar3D):
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
        pc_update_cost_ratio,
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
            "NIRRT*-PNG 3D",
        )
        self.png_wrapper = png_wrapper
        self.pc_n_points = pc_n_points # * number of points in pc
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.pc_update_cost_ratio = pc_update_cost_ratio
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = NIRRTStarVisualizer3D(self.x_start, self.x_goal, self.env)


    def init_pc(self):
        self.update_point_cloud(
            cmax=np.inf,
            cmin=None,
        )

    def planning(
        self,
        visualize=False,
    ):
        start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * nirrt*
        c_best = np.inf
        c_update = c_best # * nirrt*
        for k in range(self.iter_max):
            if k % 1000 == 0:
                print(k)
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        if len(self.path_solutions)>0:
            c_best, x_best = self.find_best_path_solution()
            self.path = self.extract_path(x_best)
        else:
            self.path = []
        if visualize:
            self.visualize(x_center, c_best, start_goal_straightline_dist, C)


    def generate_random_node(
        self,
        c_curr,
        c_min,
        x_center,
        C,
        c_update,
    ):
        '''
        - outputs
            - node_rand: np (2,)
            - c_update: scalar
        '''
        # * tested that np.inf < alpha*np.inf is False, alpha in (0,1]
        if c_curr < self.pc_update_cost_ratio*c_update:
            self.update_point_cloud(c_curr, c_min)
            c_update = c_curr
        if np.random.random() < self.pc_sample_rate:
            return self.SamplePointCloud(), c_update
        else:
            if c_curr < np.inf:
                return self.SampleInformedSubset(
                    c_curr,
                    c_min,
                    x_center,
                    C,
                ), c_update
            else:
                return self.SampleFree(), c_update

    def SamplePointCloud(self):
        return self.path_point_cloud_pred[np.random.randint(0,len(self.path_point_cloud_pred))]

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

    def visualize(self, x_center, c_best, start_goal_straightline_dist, C, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "nirrt* 3D, iteration " + str(self.iter_max)
        # if img_filename is None:
        #     img_filename="nirrt*_3d_example.png"
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
            


    def planning_block_gap(
        self,
        path_len_threshold,
    ):
        path_len_list = []
        start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * nirrt*
        c_best = np.inf
        c_update = c_best # * nirrt*
        better_than_path_len_threshold = False
        for k in range(self.iter_max):
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
            path_len_list.append(c_best)
            if k % 1000 == 0:
                print("{0}/{1} - current: {2:.2f}, threshold: {3:.2f}".format(\
                    k, self.iter_max, c_best, path_len_threshold)) #* not k+1, because we are not getting c_best after iteration is done
            if c_best < path_len_threshold:
                better_than_path_len_threshold = True
                break
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        path_len_list = path_len_list[1:] # * the first one is the initialized c_best before iteration
        if better_than_path_len_threshold:
            return path_len_list
        # * path cost for the last iteration
        if len(self.path_solutions)>0:
            c_best, x_best = self.find_best_path_solution()
        path_len_list.append(c_best)
        # * len(path_len_list)==self.iter_max
        print("{0}/{1} - current: {2:.2f}, threshold: {3:.2f}".format(\
            len(path_len_list), self.iter_max, c_best, path_len_threshold)) #* not k+1, because we are not getting c_best after iteration is done
        return path_len_list

    def planning_random(
        self,
        iter_after_initial,
    ):
        path_len_list = []
        start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * nirrt*
        c_best = np.inf
        c_update = c_best # * nirrt*
        better_than_inf = False
        for k in range(self.iter_max):
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
            path_len_list.append(c_best)
            if k % 1000 == 0:
                if c_best == np.inf:
                    print("{0}/{1} - current: inf".format(k, self.iter_max)) #* not k+1, because we are not getting c_best after iteration is done
            if c_best < np.inf:
                better_than_inf = True
                print("{0}/{1} - current: {2:.2f}".format(k, self.iter_max, c_best))
                break
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        path_len_list = path_len_list[1:] # * the first one is the initialized c_best before iteration
        if better_than_inf:
            initial_path_len = path_len_list[-1]
        else:
            # * path cost for the last iteration
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
            path_len_list.append(c_best)
            initial_path_len = path_len_list[-1]
            if initial_path_len == np.inf:
                # * fail to find initial path solution
                return path_len_list
        path_len_list = path_len_list[:-1] # * for loop below will add initial_path_len to path_len_list
        # * iteration after finding initial solution
        for k in range(iter_after_initial):
            c_best, x_best = self.find_best_path_solution() # * there must be path solutions
            path_len_list.append(c_best)
            if k % 1000 == 0:
                print("{0}/{1} - current: {2:.2f}, initial: {3:.2f}, cmin: {4:.2f}".format(\
                    k, iter_after_initial, c_best, initial_path_len, start_goal_straightline_dist))
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        # * path cost for the last iteration
        c_best, x_best = self.find_best_path_solution() # * there must be path solutions
        path_len_list.append(c_best)
        print("{0}/{1} - current: {2:.2f}, initial: {3:.2f}".format(\
            iter_after_initial, iter_after_initial, c_best, initial_path_len))
        return path_len_list

def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    return NIRRTStarPNG3D(
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
    )


    