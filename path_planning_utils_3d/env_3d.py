import numpy as np


class aabb(object):
    # make AABB out of blocks, 
    # P: center point
    # E: extents
    # O: Rotation matrix in SO(3), in {w}
    def __init__(self,AABB):
        self.P = [(AABB[3] + AABB[0])/2, (AABB[4] + AABB[1])/2, (AABB[5] + AABB[2])/2]# center point
        self.E = [(AABB[3] - AABB[0])/2, (AABB[4] - AABB[1])/2, (AABB[5] - AABB[2])/2]# extents
        self.O = [[1,0,0],[0,1,0],[0,0,1]]


class Env():
    def __init__(
        self,
        env_dims,
        box_obstacles,
        ball_obstacles,
        clearance=0,
        resolution=1):
        """
        - inputs
            - env_dims: tuple (xmax, ymax, zmax)
            - box_obstacles: list of [x,y,z,w,h,d]
            - ball_obstacles: list of [x,y,z,r]
            - x_start: tuple of 3 (x,y,z)
            - x_goal: tuple of 3 (x,y,z)
        """
        self.resolution = resolution
        self.clearance = clearance
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = env_dims
        self.boundary_no_clearance = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

        self.box_obstacles = np.array(box_obstacles) # (n_blocks, 6) (x,y,z,w,h,d)
        self.blocks_no_clearance = np.array(box_obstacles) # (n_blocks, 6) (x,y,z,w,h,d)
        self.blocks_no_clearance[:,3:] += self.blocks_no_clearance[:,:3] # (n_blocks, 6) (x1,y1,z1,x2,y2,z2)
        self.AABB_no_clearance = getAABB2(self.blocks_no_clearance)
        self.balls_no_clearance = np.array(ball_obstacles)

        self.boundary = self.boundary_no_clearance.copy() # [xmin+c, ymin+c, zmin+c, xmax-c, ymax-c, zmax-c]
        self.boundary[:3] += clearance
        self.boundary[3:] -= clearance

        self.blocks = self.blocks_no_clearance.copy() # (n_blocks, 6) (x1-c,y1-c,z1-c,x2+c,y2+c,z2+c)
        self.blocks[:,:3] -= clearance
        self.blocks[:,3:] += clearance
        self.AABB = getAABB2(self.blocks)

        self.balls = self.balls_no_clearance.copy() # (n_blocks, 4) (x,y,z,r+c)
        self.balls[:,3] += clearance

        self.start = None
        self.goal = None
        self.t = 0 # time 

    def set_start_goal(
        self,
        x_start,
        x_goal,
    ):
        self.start = np.array(x_start)
        self.goal = np.array(x_goal)


def getAABB2(blocks):
    # used in lineAABB
    AABB = []
    for i in blocks:
        AABB.append(aabb(i))
    return AABB
