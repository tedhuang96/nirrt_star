class Env:
    def __init__(
        self,
        env_dict,
        ):
        self.img_height, self.img_width = env_dict['env_dims']
        self.x_range = (0, self.img_width)
        self.y_range = (0, self.img_height)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = env_dict['circle_obstacles']
        self.obs_rectangle = env_dict['rectangle_obstacles']

    def obs_boundary(self):
        obs_boundary = [
            [-1, -1, 1, self.img_height+1],
            [-1, self.img_height, self.img_width+1, 1],
            [0, -1, self.img_width+1, 1],
            [self.img_width, 0, 1, self.img_height+1]
        ]
        return obs_boundary
