from os.path import join

import torch
import numpy as np
import matplotlib.pyplot as plt

from pointnet_pointnet2.models.pointnet2 import get_model
from pointnet_pointnet2.models.pointnet2_utils import pc_normalize
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points
from wrapper.utils.bfs_connect_heuristic import get_boundary_mask, bfs_point_cloud_visualization, select_heuristic_boundary_point


class PNGWrapper:
    def __init__(
        self,
        num_classes=2,
        root_dir='.',
        device='cuda',
    ):
        """
        - inputs:
            - num_classes: default 2, for path and not path.
        """
        self.model = get_model(num_classes).to(device)
        model_filepath = join(root_dir, 'results/model_training/pointnet2_3d/checkpoints/best_pointnet2_3d.pth')
        checkpoint = torch.load(model_filepath, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.eval()
        self.device = device
        print("PointNet++ wrapper 3d is initialized. Using BFS for Neural Connect.")
    
    def classify_path_points(
        self,
        pc,
        start_mask,
        goal_mask,
    ):
        """
        - inputs:
            - pc: np float32 (n_points, 3) for XYZ
            - start_mask: np float32 (n_points,) 1-0 mask
            - goal_mask: np float32 (n_points,) 1-0 mask
        - outputs:
            - path_pred: np (n_points, ), 1-0 mask, 1 is path point, 0 is not.
            - path_score: np float32 (n_points, ), value between 0 and 1 whether it could be a path point or not.
        """
        with torch.no_grad():
            # assume type is np.float32
            n_points = pc.shape[0]
            pc_xyz = torch.from_numpy(pc_normalize(pc)).to(self.device) # (n_points, 3)
            free_mask = 1-(start_mask+goal_mask).astype(bool) # (n_points,)
            pc_features = torch.from_numpy(np.stack(
                (start_mask, goal_mask, free_mask.astype(np.float32)),
                axis=-1,
            )).to(self.device) # (n_points, 3)

            model_inputs = torch.cat([pc_xyz, pc_features], dim=1) # (n_points, 6)
            model_inputs = model_inputs.permute(1,0).unsqueeze(0) # (1, n_features, n_points)
            seg_pred, trans_feat = self.model(model_inputs)
            path_pred = np.argmax(seg_pred.detach().to('cpu').numpy(), 2)[0] # (n_points,) # 0 -> not path, 1 -> path 
            path_score = torch.softmax(seg_pred,dim=-1)[0,:,1].detach().to('cpu').numpy()# (1, n_points, 2)->(n_points,)

            return path_pred, path_score
        
        
    def generate_connected_path_points(
        self,
        pc,
        x_start,
        x_goal,
        env_dict,
        neighbor_radius,
        max_trial_attempts,
        visualize=False,
        vis_folderpath="",
        token="",
    ):
        """
        - inputs:
            - pc: np float32 (n_points, 2) for XY or (n_points, 3) for XYZ
            - x_start: np float64 (2,)
            - x_goal: np float64 (2,)
            - env_dict:
                - ['env_dims']: (env_height, env_width, env_depth)
                - ['circle_obstacles']
                - ['rectangle_obstacles']
            - neighbor_radius: 10, also connection radius
            - max_trial_attempts: 5
            - visualize: default False
            - vis_folderpath: default ""
            - token: token for environment-start-goal sample

        - outputs:
            - connection_success: True or False
            - num_png_runs: number of trials, how many times png is run
            - path_pred_mask: (n_points, ) np float32
        """
        env_height, env_width, env_depth = env_dict['env_dims']
        has_path = False
        path_pred_mask = np.zeros(len(pc)).astype(np.float32)
        start_mask = get_point_cloud_mask_around_points(
            pc,
            x_start[np.newaxis].astype(np.float32),
            neighbor_radius,
        )
        goal_mask = get_point_cloud_mask_around_points(
            pc,
            x_goal[np.newaxis].astype(np.float32),
            neighbor_radius,
        )
        for trial_i in range(max_trial_attempts):
            path_pred, path_score = self.classify_path_points(
                pc,
                start_mask,
                goal_mask,
            )
            path_pred_mask = ((path_pred_mask + path_pred)>0).astype(np.float32)
            if visualize:
                img_filename = join(vis_folderpath, token+"_"+str(trial_i+1)+"_path_points.png")
                self.visualize_path_points(
                    pc,
                    x_start,
                    x_goal,
                    path_pred_mask,
                    env_height,
                    env_width,
                    env_depth,
                    img_filename,
                )
            has_path, path_line, visited_mask = bfs_point_cloud_visualization(
                pc,
                path_pred_mask,
                x_start.astype(np.float32),
                x_goal.astype(np.float32),
                neighbor_radius,
            )
            visited_points = pc[visited_mask.astype(bool)]
            unvisited_mask = 1-path_pred_mask
            boundary_mask = get_boundary_mask(
                pc,
                visited_mask,
                unvisited_mask,
                neighbor_radius,
            )
            boundary_point_index, boundary_point, boundary_point_heuristic = select_heuristic_boundary_point(
                pc,
                boundary_mask,
                x_start.astype(np.float32),
                x_goal.astype(np.float32),
            )
            boundary_point_cloud = pc[boundary_mask.astype(bool)]
            if visualize:
                img_filename = join(vis_folderpath, token+"_"+str(trial_i+1)+"_start_goal_connection.png")
                self.visualize_connection_process(
                    pc,
                    x_start,
                    x_goal,
                    path_pred_mask,
                    path_line,
                    visited_points,
                    boundary_point_cloud,
                    boundary_point_heuristic,
                    boundary_point,
                    env_height,
                    env_width,
                    env_depth,
                    img_filename,
                )
            if has_path:
                break
            if boundary_point is None:
                next_start_mask = start_mask
            else:
                next_start_mask = get_point_cloud_mask_around_points(
                    pc,
                    boundary_point,
                    neighbor_radius,
                )
            has_path, path_line, visited_mask = bfs_point_cloud_visualization(
                pc,
                path_pred_mask,
                x_goal.astype(np.float32),
                x_start.astype(np.float32),
                neighbor_radius,
            )
            visited_points = pc[visited_mask.astype(bool)]
            unvisited_mask = 1-path_pred_mask
            boundary_mask = get_boundary_mask(
                pc,
                visited_mask,
                unvisited_mask,
                neighbor_radius,
            )
            boundary_point_index, boundary_point, boundary_point_heuristic = select_heuristic_boundary_point(
                pc,
                boundary_mask,
                x_goal.astype(np.float32),
                x_start.astype(np.float32),
            )
            boundary_point_cloud = pc[boundary_mask.astype(bool)]
            if visualize:
                img_filename = join(vis_folderpath, token+"_"+str(trial_i+1)+"_goal_start_connection.png")
                self.visualize_connection_process(
                    pc,
                    x_start,
                    x_goal,
                    path_pred_mask,
                    path_line,
                    visited_points,
                    boundary_point_cloud,
                    boundary_point_heuristic,
                    boundary_point,
                    env_height,
                    env_width,
                    env_depth,
                    img_filename,
                )
            if has_path:
                break
            if boundary_point is None:
                # import pdb; pdb.set_trace()
                next_goal_mask = goal_mask
            else:
                next_goal_mask = get_point_cloud_mask_around_points(
                    pc,
                    boundary_point,
                    neighbor_radius,
                )
            start_mask = next_start_mask
            goal_mask = next_goal_mask
        connection_success = has_path
        num_png_runs = trial_i+1
        return connection_success, num_png_runs, path_pred_mask

    def visualize_path_points(
        self,
        pc,
        x_start,
        x_goal,
        path_pred_mask,
        env_height,
        env_width,
        env_depth,
        img_filename=None,
    ):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=90., azim=0.)
        ax.plot(x_start[0], x_start[1], x_start[2], 'r*', markersize=7, zorder=10)
        ax.plot(x_goal[0], x_goal[1], x_goal[2], 'y*', markersize=7, zorder=10)

        ax.set_xlim3d([0, env_height])
        ax.set_ylim3d([0, env_width])
        ax.set_zlim3d([0, env_depth])
        ax.axis('off')
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

        path_point_cloud_pred = pc[path_pred_mask.nonzero()[0]]
        other_point_cloud_mask = 1.-path_pred_mask
        other_point_cloud = pc[other_point_cloud_mask.nonzero()[0]]
        fig, ax = plt.subplots()
        ax.scatter(other_point_cloud[:,0], other_point_cloud[:,1], other_point_cloud[:,2], s=2, c='C0')
        ax.scatter(path_point_cloud_pred[:,0], path_point_cloud_pred[:,1], path_point_cloud_pred[:,2], s=2, c='C1')
        plt.axis('off')
        if img_filename is None:
            plt.show()
        else:
            fig.savefig(img_filename, bbox_inches='tight', pad_inches=0)
        return

    def visualize_connection_process(
        self,
        pc,
        x_start,
        x_goal,
        path_pred_mask,
        path_line,
        visited_points,
        boundary_point_cloud,
        boundary_point_heuristic,
        boundary_point,
        env_height,
        env_width,
        env_depth,
        img_filename=None,
    ):
        path_point_cloud_pred = pc[path_pred_mask.nonzero()[0]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=90., azim=0.)
        ax.plot(x_start[0], x_start[1], x_start[2], 'r*', markersize=7, zorder=10)
        ax.plot(x_goal[0], x_goal[1], x_goal[2], 'y*', markersize=7, zorder=10)

        ax.set_xlim3d([0, env_height])
        ax.set_ylim3d([0, env_width])
        ax.set_zlim3d([0, env_depth])
        ax.axis('off')
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

        ax.scatter(path_point_cloud_pred[:,0], path_point_cloud_pred[:,1], path_point_cloud_pred[:,2], s=1, c='C1')
        if path_line is not None:
            ax.plot(path_line[:,0], path_line[:,1], path_line[:,2], c='C1', linewidth=1, zorder=9)

        ax.scatter(visited_points[:,0], visited_points[:,1], visited_points[:,2], s=2, c='C2')

        if boundary_point is not None:
            ax.scatter(boundary_point_cloud[:,0], boundary_point_cloud[:,1], boundary_point_cloud[:,2], c=boundary_point_heuristic, cmap='viridis')
            ax.plot(boundary_point[0], boundary_point[1], boundary_point[2], 'b*',ms=3)
        plt.axis('off')
        if img_filename is None:
            plt.show()
        else:
            plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
