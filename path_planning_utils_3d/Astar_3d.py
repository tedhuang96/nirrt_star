import numpy as np

from path_planning_utils_3d.queue_3d import MinheapPQ
from path_planning_utils_3d.utils_3d import getDist, cost, children, heuristic_fun


class Weighted_A_star(object):
    def __init__(self, env):
        self.Alldirec = {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1, \
                        (-1, 0, 0): 1, (0, -1, 0): 1, (0, 0, -1): 1, \
                        (1, 1, 0): np.sqrt(2), (1, 0, 1): np.sqrt(2), (0, 1, 1): np.sqrt(2), \
                        (-1, -1, 0): np.sqrt(2), (-1, 0, -1): np.sqrt(2), (0, -1, -1): np.sqrt(2), \
                        (1, -1, 0): np.sqrt(2), (-1, 1, 0): np.sqrt(2), (1, 0, -1): np.sqrt(2), \
                        (-1, 0, 1): np.sqrt(2), (0, 1, -1): np.sqrt(2), (0, -1, 1): np.sqrt(2), \
                        (1, 1, 1): np.sqrt(3), (-1, -1, -1) : np.sqrt(3), \
                        (1, -1, -1): np.sqrt(3), (-1, 1, -1): np.sqrt(3), (-1, -1, 1): np.sqrt(3), \
                        (1, 1, -1): np.sqrt(3), (1, -1, 1): np.sqrt(3), (-1, 1, 1): np.sqrt(3)}
        self.settings = 'CollisionChecking' # 'NonCollisionChecking' or 'CollisionChecking'                
        self.env = env
        self.start, self.goal = tuple(self.env.start), tuple(self.env.goal)
        self.g = {self.start:0,self.goal:np.inf}
        self.Parent = {}
        self.CLOSED = set()
        self.V = []
        self.done = False
        self.Path = []
        self.ind = 0
        self.x0, self.xt = self.start, self.goal
        self.OPEN = MinheapPQ()  # store [point,priority]
        self.OPEN.put(self.x0, self.g[self.x0] + heuristic_fun(self,self.x0))  # item, priority = g + h
        self.lastpoint = self.x0

    def run(self, N=None):
        xt = self.xt
        xi = self.x0
        while self.OPEN:  # while xt not reached and open is not empty
            xi = self.OPEN.get()
            if xi is None:
                return False
            if xi not in self.CLOSED:
                self.V.append(np.array(xi))
            self.CLOSED.add(xi)  # add the point in CLOSED set
            if getDist(xi,xt) < self.env.resolution:
                break
            # visualization(self)
            for xj in children(self,xi):
                # if xj not in self.CLOSED:
                if xj not in self.g:
                    self.g[xj] = np.inf
                else:
                    pass
                a = self.g[xi] + cost(self, xi, xj)
                if a < self.g[xj]:
                    self.g[xj] = a
                    self.Parent[xj] = xi
                    # assign or update the priority in the open
                    self.OPEN.put(xj, a + 1 * heuristic_fun(self, xj))
            # For specified expanded nodes, used primarily in LRTA*
            if N:
                if len(self.CLOSED) % N == 0:
                    break
            if self.ind % 10000 == 0: print('number node expanded = ' + str(len(self.V)))
            self.ind += 1

        self.lastpoint = xi
        # if the path finding is finished
        if self.lastpoint in self.CLOSED:
            self.done = True
            self.Path = self.path()
            return True

        return False

    def path(self):
        path = []
        x = self.lastpoint
        start = self.x0
        while x != start:
            path.append([x, self.Parent[x]])
            x = self.Parent[x]
        return path
    
    def get_path_solution(self):
        path_solution = []
        x = self.lastpoint
        start = self.x0
        while x != start:
            path_solution.append(x)
            x = self.Parent[x]
        path_solution.append(x)
        path_solution.reverse()
        return path_solution
    
    def check_success(self, path_solution):
        return path_solution[0]==self.start and path_solution[-1]==self.goal
