from os.path import join

import torch
import numpy as np

from pointnet_pointnet2.models.pointnet2 import get_model
from pointnet_pointnet2.models.pointnet2_utils import pc_normalize

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
        model_filepath = join(root_dir, 'results/model_training/pointnet2_2d/checkpoints/best_pointnet2_2d.pth')
        checkpoint = torch.load(model_filepath, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.eval()
        self.device = device
        print("PointNet++ wrapper is initialized.")
    
    def classify_path_points(
        self,
        pc,
        start_mask,
        goal_mask,
    ):
        """
        - inputs:
            - pc: np float32 (n_points, 2) for XY or (n_points, 3) for XYZ
            - start_mask: np float32 (n_points,) 1-0 mask
            - goal_mask: np float32 (n_points,) 1-0 mask
        - outputs:
            - path_pred: np (n_points, ), 1-0 mask, 1 is path point, 0 is not.
            - path_score: np float32 (n_points, ), value between 0 and 1 whether it could be a path point or not.
        """
        with torch.no_grad():
            # assume type is np.float32
            n_points = pc.shape[0]
            if pc.shape[1]==2:
                pc = np.concatenate(
                    (pc, np.zeros((n_points, 1)).astype(np.float32)),
                    axis=1,
                )
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