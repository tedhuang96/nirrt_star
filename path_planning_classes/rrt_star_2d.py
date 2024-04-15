

import math
import numpy as np

from path_planning_classes.rrt_base_2d import RRTBase2D
from path_planning_classes.rrt_visualizer_2d import RRTStarVisualizer

class RRTStar2D(RRTBase2D):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env,
        clearance,
    ):
        super().__init__(
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            env,
            clearance,
            "RRT* 2D",
        )
        self.visualizer = RRTStarVisualizer(self.x_start, self.x_goal, self.env)

    def planning(
        self,
        visualize=False,
    ):
        for k in range(self.iter_max):
            node_rand = self.generate_random_node()
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
            if (k+1) % 1000 == 0:
                print(k+1)
        goal_parent_index = self.search_goal_parent()
        if goal_parent_index is None:
            if visualize:
                self.visualize()
            return
        self.path = self.extract_path(goal_parent_index)
        if visualize:
            self.visualize()

    def new_state(self, node_start, node_goal):
        '''
        - inputs:
            - node_start: np (2,)
            - node_goal: np (2,)
        - outputs:
            - node_new: np (2,)
        '''
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        dist = min(self.step_len, dist)
        node_new = node_start + dist*np.array([math.cos(theta), math.sin(theta)])
        return node_new

    def choose_parent(self, node_new, neighbor_indices, node_new_index, curr_node_new_cost):
        vec_neighbors_to_new = node_new-self.vertices[:self.num_vertices][neighbor_indices]
        dist_neighbors_to_new = np.hypot(vec_neighbors_to_new[:,0], vec_neighbors_to_new[:,1])
        neighbor_costs = []
        for neighbor_index in neighbor_indices:
            neighbor_costs.append(self.cost(neighbor_index))
        node_new_cost_candidates = np.array(neighbor_costs)+dist_neighbors_to_new
        best_node_new_cost_idx = np.argmin(node_new_cost_candidates)
        if node_new_cost_candidates[best_node_new_cost_idx] < curr_node_new_cost:
            node_new_min_cost_neighbor_index = neighbor_indices[best_node_new_cost_idx]
            self.vertex_parents[node_new_index] = node_new_min_cost_neighbor_index

    def rewire(self, node_new, neighbor_indices, node_new_index):
        vec_new_to_neighbors = self.vertices[:self.num_vertices][neighbor_indices]-node_new
        dist_new_to_neighbors = np.hypot(vec_new_to_neighbors[:,0], vec_new_to_neighbors[:,1])
        node_new_cost = self.cost(node_new_index)
        # * neighbors might interact with each other. so we do not do parallization (like difference between synchronous and asynchronous updates)
        for i, neighbor_index in enumerate(neighbor_indices):
            if self.cost(neighbor_index) > node_new_cost+dist_new_to_neighbors[i]:
                self.vertex_parents[neighbor_index] = node_new_index

    def search_goal_parent(self):
        vec_to_goal = self.x_goal - self.vertices[:self.num_vertices]
        dist_to_goal = np.hypot(vec_to_goal[:,0], vec_to_goal[:,1])
        indices_vertex_within_step_len = np.where(dist_to_goal<=self.step_len)[0]
        if len(indices_vertex_within_step_len)==0:
            return None
        total_cost_candidates = []
        for vertex_index, vertex, dist_vertex_to_goal in \
            zip(indices_vertex_within_step_len, \
                self.vertices[:self.num_vertices][indices_vertex_within_step_len],\
                dist_to_goal[indices_vertex_within_step_len]):
            if not self.utils.is_collision(vertex, self.x_goal):
                total_cost_candidates.append(self.cost(vertex_index)+dist_vertex_to_goal)
            else:
                total_cost_candidates.append(np.inf)
        
        return indices_vertex_within_step_len[np.argmin(total_cost_candidates)]

    def generate_random_node(self):
        '''
        Return a randomly sampled node. np (2,)
        '''
        return self.SampleFree()

    def find_near_neighbors(self, node_new, node_new_index=None):
        '''
        - inputs
            - node_new: np (2,)
        - outputs
            - neighbor_indices: np (n_neighbor_vertices, ) Though it is
            neighbor_indices in [0,num_vertices], it still works for [0,1+iter_max].
        '''
        r = min(self.search_radius*math.sqrt(math.log(self.num_vertices)/self.num_vertices), self.step_len)
        vec_to_node_new = node_new-self.vertices[:self.num_vertices]
        dist_to_node_new = np.hypot(vec_to_node_new[:,0], vec_to_node_new[:,1])
        indices_vertex_within_r = np.where(dist_to_node_new<=r)[0]
        neighbor_indices = []
        if len(indices_vertex_within_r)==0:
            return np.array(neighbor_indices)
        for vertex_index, vertex in zip(indices_vertex_within_r, self.vertices[:self.num_vertices][indices_vertex_within_r]):
            if not self.utils.is_collision(node_new, vertex):
                if node_new_index is not None and vertex_index != node_new_index:
                    neighbor_indices.append(vertex_index)
        return np.array(neighbor_indices)

    def visualize(self, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "rrt* 2D, iteration " + str(self.iter_max)
        if img_filename is None:
            img_filename = "rrt*_2d_example.png"
        self.visualizer.animation(
            self.vertices[:self.num_vertices],
            self.vertex_parents[:self.num_vertices],
            self.path,
            figure_title,
            animation=False,
            img_filename=img_filename)

    def planning_block_gap(
        self,
        path_len_threshold,
    ):
        path_len_list = []
        for k in range(self.iter_max):
            node_rand = self.generate_random_node()
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
            goal_parent_index = self.search_goal_parent()
            if goal_parent_index is None:
                current_path = []
            else:
                current_path = self.extract_path(goal_parent_index)
            current_path_len = self.get_path_len(current_path)
            path_len_list.append(current_path_len)
            if (k+1) % 1000 == 0:
                print("{0}/{1} - current: {2:.2f}, threshold: {3:.2f}".format(\
                    k+1, self.iter_max, current_path_len, path_len_threshold))
            if current_path_len < path_len_threshold:
                break
        return path_len_list

    def planning_random(
        self,
        iter_after_initial,
    ):
        path_len_list = []
        for k in range(self.iter_max):
            node_rand = self.generate_random_node()
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
            goal_parent_index = self.search_goal_parent()
            if goal_parent_index is None:
                current_path = []
            else:
                current_path = self.extract_path(goal_parent_index)
            current_path_len = self.get_path_len(current_path)
            path_len_list.append(current_path_len)
            if (k+1) % 1000 == 0:
                if current_path_len == np.inf:
                    print("{0}/{1} - current: inf".format(k+1, self.iter_max))
            if current_path_len < np.inf:
                print("{0}/{1} - current: {2:.2f}".format(k+1, self.iter_max, current_path_len))
                break
        if path_len_list[-1]==np.inf:
            # * fail to find initial path solution
            return path_len_list
        initial_path_len = path_len_list[-1]
        # * iteration after finding initial solution
        for k in range(iter_after_initial):
            node_rand = self.generate_random_node()
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
            goal_parent_index = self.search_goal_parent()
            current_path = self.extract_path(goal_parent_index) # * must be valid after initial path solution is found
            current_path_len = self.get_path_len(current_path)
            path_len_list.append(current_path_len)
            if (k+1) % 1000 == 0:
                print("{0}/{1} - current: {2:.2f}, initial: {3:.2f}".format(\
                    k+1, iter_after_initial, current_path_len, initial_path_len))
        return path_len_list

def get_path_planner(
    args,
    problem,
    neural_wrapper=None,
):
    return RRTStar2D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env'],
        args.clearance,
    )