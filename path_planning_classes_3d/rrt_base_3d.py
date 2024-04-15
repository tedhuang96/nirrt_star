import math

import numpy as np

from path_planning_classes_3d.rrt_utils_3d import Utils

class RRTBase3D:
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env,
        clearance,
        path_planner_name,
    ):
        self.x_start = np.array(x_start).astype(np.float64)
        self.x_goal = np.array(x_goal).astype(np.float64)
        self.step_len = step_len
        self.search_radius = search_radius
        self.iter_max = iter_max

        self.vertices = np.zeros((1+iter_max,3)) # * assign faster than vstack to add new vertex
        self.vertex_parents = np.zeros(1+iter_max).astype(int)
        self.vertices[0] = self.x_start # * start parent is itself
        self.num_vertices = 1
        self.path = []

        self.env = env
        self.utils = Utils(env, clearance)
        self.clearance = clearance
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.z_range = self.env.z_range

        self.path_planner_name = path_planner_name

    def planning(self):
        pass

    def SampleGlobally(self):
        return np.array((np.random.uniform(self.x_range[0] + self.clearance, self.x_range[1] - self.clearance),
                         np.random.uniform(self.y_range[0] + self.clearance, self.y_range[1] - self.clearance),
                         np.random.uniform(self.z_range[0] + self.clearance, self.z_range[1] - self.clearance),
                        ))

    def SampleFree(self):
        in_obstacle = True
        while in_obstacle:
            sampled_point = (
                np.random.uniform(self.x_range[0] + self.clearance, self.x_range[1] - self.clearance),
                np.random.uniform(self.y_range[0] + self.clearance, self.y_range[1] - self.clearance),
                np.random.uniform(self.z_range[0] + self.clearance, self.z_range[1] - self.clearance),
            )
            in_obstacle = self.utils.is_inside_obs(sampled_point)
        return np.array(sampled_point)

    def cost(self, vertex_index):
        cost = 0.
        while vertex_index != 0:
            vertex_parent_index = self.vertex_parents[vertex_index]
            dx, dy, dz = self.vertices[:self.num_vertices][vertex_index] - self.vertices[:self.num_vertices][vertex_parent_index]
            cost += math.hypot(dx, dy, dz)
            vertex_index = vertex_parent_index
        return cost

    def extract_path(self, goal_parent_index):
        path = [self.x_goal]
        path_vertex_index = goal_parent_index
        while path_vertex_index != 0:
            path.append(self.vertices[:self.num_vertices][path_vertex_index])
            path_vertex_index = self.vertex_parents[path_vertex_index]
        path.append(self.vertices[:self.num_vertices][path_vertex_index])
        path.reverse()
        path = np.stack(path, axis=0)
        return path
    
    def check_success(self, path):
        if path is None or len(path)==0:
            return False
        return np.all(path[0]==self.x_start) and np.all(path[-1]==self.x_goal)
    
    def get_path_len(self, path):
        if path is None or len(path)==0:
             return np.inf
        else:
            path = np.array(path)
            path_disp = path[1:]-path[:-1]
            return np.linalg.norm(path_disp,axis=1).sum()

    def InGoalRegion(self, node):
        return (self.Line(node, self.x_goal) < self.step_len and \
                not self.utils.is_collision(node, self.x_goal))
    
    def get_path_planner_name(self):
        return self.path_planner_name

    @staticmethod
    def nearest_neighbor(node_list, n):
        '''
        find the node in node_list which is the closest to n.
        - inputs:
            - node_list: np (num_vertices, 3)
            - n: np (3,)
        - outputs:
            - nearest_n: np (3,)
            - nearest_index
        '''
        vec_to_n = n-node_list
        nearest_index = np.argmin(np.linalg.norm(vec_to_n, axis=1))
        return node_list[nearest_index], nearest_index


    @staticmethod
    def get_distance_and_direction(node_start, node_end):
        '''
        - inputs:
            - node_start, node_end: np (3,)
        - outputs:
            - distance
            - direction unit vector, np (3,)
        '''
        dx, dy, dz = node_end - node_start
        distance = math.hypot(dx, dy, dz)
        if distance == 0:
            return 0, np.zeros(3)
        direction = (node_end - node_start)/distance
        return distance, direction


    @staticmethod
    def Line(x_start, x_goal):
        dx, dy, dz = x_goal - x_start
        return math.hypot(dx, dy, dz)