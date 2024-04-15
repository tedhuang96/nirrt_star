import numpy as np
from path_planning_classes_3d.collision_check_utils_3d import check_collision_line_balls_boxes, points_in_balls_boxes, points_in_range_3d, points_validity_3d


class Utils:
    def __init__(self, env, clearance):
        self.env = env
        self.clearance = clearance
        if len(self.env.obs_ball)>0:
            self.obs_ball = np.array(self.env.obs_ball).astype(np.float64)
        else:
            self.obs_ball = None
        if len(self.env.obs_box)>0:
            self.obs_box = np.array(self.env.obs_box).astype(np.float64)
        else:
            self.obs_box = None
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.z_range = self.env.z_range


    def is_collision(self, start, end):
        """
        - inputs:
            - start: np or tuple (3,)
            - end: np or tuple (3,)
        - outputs:
            - collision: bool.
        """
        line = np.array([start, end]).astype(np.float64)
        return check_collision_line_balls_boxes(
            line,
            self.obs_ball,
            self.obs_box,
            self.clearance,
        )
    

    def is_inside_obs(self, node):
        """
        - inputs:
            - node: tuple (3,) or np (3,)
        - outputs:
            - in_obstacle: bool.
        """
        return points_in_balls_boxes(
            (node[0], node[1], node[2]),
            self.obs_ball,
            self.obs_box,
            self.clearance,
        )

    
    def is_in_range(self, node):
        """
        - inputs:
            - node: tuple (3,) or np (3,)
        - outputs:
            - in_range: bool.
        """
        return points_in_range_3d(
            (node[0], node[1], node[2]),
            self.x_range,
            self.y_range,
            self.z_range,
            self.clearance,
        )

    def is_valid(self, node):
        """
        Check both in range and not in obstacle.
        - inputs:
            - node: tuple (3,) or np (3,)
        - outputs:
            - validity: bool.
        """
        return points_validity_3d(
            (node[0], node[1], node[2]),
            self.obs_ball,
            self.obs_box,
            self.x_range,
            self.y_range,
            self.z_range,
            obstacle_clearance=self.clearance,
            range_clearance=self.clearance,
        )

