import math
import random

import numpy as np

from path_planning_classes.rrt_base_2d import RRTBase2D
from path_planning_classes.rrt_star_2d import RRTStar2D
from path_planning_classes.rrt_visualizer_2d import IRRTStarVisualizer

class IRRTStar2D(RRTStar2D):
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
        RRTBase2D.__init__(
            self,
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            env,
            clearance,
            "IRRT* 2D",
        )
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = IRRTStarVisualizer(self.x_start, self.x_goal, self.env)

    def init(self):
        cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        x_center = np.zeros((3,1))
        x_center[:2,0] = (self.x_start+self.x_goal)/2.
        return theta, cMin, x_center, C

    def planning(
        self,
        visualize=False,
    ):
        theta, start_goal_straightline_dist, x_center, C = self.init()
        c_best = np.inf
        for k in range(self.iter_max):
            if k % 1000 == 0:
                print(k)
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
            node_rand = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C)
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
        if self.iter_max % 1000 == 0:
            print(self.iter_max)
        if len(self.path_solutions)>0:
            c_best, x_best = self.find_best_path_solution()
            self.path = self.extract_path(x_best)
        else:
            self.path = []
        if visualize:
            self.visualize(x_center, c_best, start_goal_straightline_dist, theta)

    def find_best_path_solution(self):
        '''
        - outputs
            - c_best: the current best path cost
            - x_best: index of the current best path solution (goal parent vertex index)
        '''
        path_costs = []
        for goal_parent_vertex_idx in self.path_solutions:
            goal_parent_vertex = self.vertices[:self.num_vertices][goal_parent_vertex_idx]
            path_costs.append(self.cost(goal_parent_vertex_idx)+self.Line(goal_parent_vertex, self.x_goal)) # * fixed bug of irrt implementation
        best_path_idx = np.argmin(path_costs)
        c_best = path_costs[best_path_idx]
        x_best = self.path_solutions[best_path_idx]
        return c_best, x_best

    def generate_random_node(
        self,
        c_max,
        c_min,
        x_center,
        C,
    ):
        '''
        - outputs
            - node_rand: np (2,)
        '''
        if c_max < np.inf:
            node_rand = self.SampleInformedSubset(
                c_max,
                c_min,
                x_center,
                C,
            )
        else:
            node_rand = self.SampleFree()
        return node_rand

    def SampleInformedSubset(
        self,
        c_max,
        c_min,
        x_center,
        C,
    ):
        if c_max ** 2 - c_min ** 2<0:
            eps = 1e-6
        else:
            eps = 0
        r = [
                c_max / 2.0,
                math.sqrt(c_max ** 2 - c_min ** 2+eps) / 2.0,
                math.sqrt(c_max ** 2 - c_min ** 2+eps) / 2.0,
            ]
        L = np.diag(r)
        while True:
            x_ball = self.SampleUnitBall()
            node_rand = np.dot(np.dot(C, L), x_ball) + x_center # np (3,1)
            if self.utils.is_valid((node_rand[0,0], node_rand[1,0])):
                break
        node_rand = node_rand[:2,0] # np (2,)
        return node_rand

    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.zeros((3,1))
        a1[:2,0] = (x_goal-x_start)/L
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
        return C

    def visualize(self, x_center, c_best, start_goal_straightline_dist, theta, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "irrt* 2D, iteration " + str(self.iter_max)
        if img_filename is None:
            img_filename="irrt*_2d_example.png"
        self.visualizer.animation(
            self.vertices[:self.num_vertices],
            self.vertex_parents[:self.num_vertices],
            self.path,
            figure_title,
            x_center,
            c_best,
            start_goal_straightline_dist,
            theta,
            img_filename=img_filename,
        )
        
    def planning_block_gap(
        self,
        path_len_threshold,
    ):
        path_len_list = []
        theta, start_goal_straightline_dist, x_center, C = self.init()
        c_best = np.inf
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
            node_rand = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C)
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
        print("{0}/{1} - current: {2:.2f}, threshold: {3:.2f}".format(\
            len(path_len_list), self.iter_max, c_best, path_len_threshold)) #* not k+1, because we are not getting c_best after iteration is done
        return path_len_list

    def planning_random(
        self,
        iter_after_initial,
    ):
        path_len_list = []
        theta, start_goal_straightline_dist, x_center, C = self.init()
        c_best = np.inf
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
            node_rand = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C)
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
            node_rand = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C)
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
    neural_wrapper=None,
):
    return IRRTStar2D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env'],
        args.clearance,
    )