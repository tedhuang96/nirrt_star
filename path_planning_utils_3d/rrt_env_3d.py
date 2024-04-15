class Env:
    def __init__(
        self,
        env_dict,
        ):
        self.env_height, self.env_width, self.env_depth = env_dict['env_dims']
        self.x_range = (0, self.env_width)
        self.y_range = (0, self.env_height)
        self.z_range = (0, self.env_depth)
        self.obs_ball = env_dict['ball_obstacles']
        self.obs_box = env_dict['box_obstacles']