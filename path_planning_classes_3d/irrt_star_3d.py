import numpy as np

from path_planning_classes_3d.rrt_base_3d import RRTBase3D
from path_planning_classes_3d.rrt_star_3d import RRTStar3D
from path_planning_classes_3d.rrt_visualizer_3d import IRRTStarVisualizer3D

class IRRTStar3D(RRTStar3D):
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
        RRTBase3D.__init__(
            self,
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            env,
            clearance,
            "IRRT* 3D",
        )
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = IRRTStarVisualizer3D(self.x_start, self.x_goal, self.env)

    def init(self):
        cMin, direction = self.get_distance_and_direction(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        x_center = (self.x_start+self.x_goal)/2.
        return cMin, x_center, C

    def planning(
        self,
        visualize=False,
    ):
        start_goal_straightline_dist, x_center, C = self.init()
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
            self.visualize(x_center, c_best, start_goal_straightline_dist, C)

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
            - node_rand: np (3,)
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
        '''
        - inputs:
            - c_max: scalar
            - c_min: distance between x_start, x_goal
            - x_center: (3,). In 2D it is (3,1)
            - C: rotation matrix
        '''
        if c_max ** 2 - c_min ** 2<0:
            eps = 1e-6
        else:
            eps = 0
        r = np.zeros(3)
        r[0] = c_max /2
        for i in [1,2]:
            r[i] = np.sqrt(c_max**2-c_min**2+eps) / 2
        L = np.diag(r) # R3*3
        while True:
            xball = self.SampleUnitBall() # np.array
            node_rand =  C@L@xball + x_center # np (3,)
            if self.utils.is_valid((node_rand[0], node_rand[1], node_rand[2])):
                break
        return node_rand

    @staticmethod
    def SampleUnitBall():
        # uniform sampling in spherical coordinate system in 3D
        # np (3,). in 2d its shape is (3,1)
        r = np.random.uniform(0.0, 1.0)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x,y,z])
    
    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        # S0(n): such that the x_start and x_goal are the center points
        '''
        Should be same as 2d version.
        - inputs:
            - x_start: np (3,)
            - x_goal: np (3,)
            - L: distance between x_start and x_goal
        '''
        a1 = (x_goal - x_start) / L
        M = np.outer(a1,[1,0,0])
        U, S, V = np.linalg.svd(M)
        C = U@np.diag([1, 1, np.linalg.det(U)*np.linalg.det(V)])@V.T
        return C


    def visualize(self, x_center, c_best, start_goal_straightline_dist, C, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "irrt* 3D, iteration " + str(self.iter_max)
        # if img_filename is None:
        #     img_filename="irrt*_3d_example.png"
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
                    # print('use node nearest as node new.')
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
    return IRRTStar3D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env'],
        args.clearance,
    )