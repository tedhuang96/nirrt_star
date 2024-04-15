import numpy as np
from torch.utils.data import Dataset

from pointnet_pointnet2.models.pointnet2_utils import pc_normalize


class PathPlanDataset(Dataset):
    def __init__(
        self,
        dataset_filepath,
    ):
        """
        dataset_filepath: 'data/random_2d/'+mode+'.npz', where mode is 'train', 'val', or 'test'.
        """
        data = np.load(dataset_filepath)
        self.pc = data['pc'].astype(np.float32) # (N, 2048, 2)
        self.start_mask = data['start'].astype(np.float32) # (N, 2048)
        self.goal_mask = data['goal'].astype(np.float32) # (N, 2048)
        self.free_mask = data['free'].astype(np.float32) # (N, 2048)
        self.astar_mask = data['astar'].astype(np.float32) # (N, 2048)
        self.token = data['token']
        if self.pc.shape[2]==2:
            self.pc = np.concatenate(
                (self.pc, np.zeros((self.pc.shape[0], self.pc.shape[1], 1)).astype(np.float32)),
                axis=2,
            )
        if self.pc.shape[2]!=3:
            raise RuntimeError("Point cloud is not 3D.")
        labelweights, _ = np.histogram(self.astar_mask, range(3))
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)

    def __len__(self):
        return len(self.pc)
    
    def __getitem__(self, index):
        pc_xyz_raw = self.pc[index] # (2048, 3)
        pc_xyz = pc_normalize(pc_xyz_raw)
        pc_features = np.stack(
            (self.start_mask[index], self.goal_mask[index], self.free_mask[index]),
            axis=-1,
        ) # (2048, 3)
        pc_labels = self.astar_mask[index] # (2048,)
        return pc_xyz_raw, pc_xyz, pc_features, pc_labels, self.token[index]