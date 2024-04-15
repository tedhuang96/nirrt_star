import json
import random
from os.path import join

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tsfm

normalize_imagenet = tsfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_binary_mask(env_img):
    """
    - inputs:
        - env_img: np (img_height, img_width, 3)
        - binary_mask: np float 0. or 1. (img_height, img_width)
    """
    env_dims = env_img.shape[:2]
    binary_mask = np.zeros(env_dims).astype(np.float32)
    binary_mask[env_img[:,:,0]!=0]=1
    return binary_mask

class ImageDataset(Dataset):
    def __init__(
        self,
        mode='train',
        img_height=224,
        img_width=224,
        path_thickness=3,
        augment=False,
    ):
        self.mode = mode
        self.augment = augment
        self.img_height = img_height
        self.img_width = img_width
        assert path_thickness % 2 == 1
        self.surrounding_size = (path_thickness-1)//2
        self.dataset_dir = join('data/random_2d', mode)
        self.img_dir = join(self.dataset_dir, 'env_imgs')
        self.label_dir = join(self.dataset_dir, 'astar_paths')
        with open(join(self.dataset_dir, "envs.json"), 'r') as f:
            self.env_list = json.load(f)
        self.load_raw_data()
        
    def load_raw_data(self):
        self.tokens = []
        self.raw_img_inputs = []
        self.img_labels = []
        for env_idx, env_dict in enumerate(self.env_list):
            env_img = cv2.imread("data/random_2d/"+self.mode+"/env_imgs/{0}.png".format(env_idx))
            input_env = get_binary_mask(env_img)
            for sample_idx, (x_start, x_goal) in enumerate(zip(env_dict['start'], env_dict['goal'])):
                token = "{0}_{1}".format(env_idx, sample_idx)
                self.tokens.append(token)
                path = np.loadtxt(join(self.label_dir, token+".txt"), delimiter=',')
                path = path.astype(int)
                input_start = np.zeros((self.img_height, self.img_width)).astype(np.float32)
                input_start[x_start[1]-self.surrounding_size:x_start[1]+self.surrounding_size+1,
                          x_start[0]-self.surrounding_size:x_start[0]+self.surrounding_size+1] = 1.
                input_goal = np.zeros((self.img_height, self.img_width)).astype(np.float32)
                input_goal[x_goal[1]-self.surrounding_size:x_goal[1]+self.surrounding_size+1,
                         x_goal[0]-self.surrounding_size:x_goal[0]+self.surrounding_size+1] = 1.
                raw_img_input = np.stack([input_start, input_goal, input_env], axis=0) # (3, h, w)
                self.raw_img_inputs.append(raw_img_input)
                img_label = np.zeros((1, self.img_height, self.img_width)).astype(np.float32)
                for path_point in path:
                    img_label[:,
                              path_point[1]-self.surrounding_size:path_point[1]+self.surrounding_size+1,
                              path_point[0]-self.surrounding_size:path_point[0]+self.surrounding_size+1,
                    ] = 1.
                self.img_labels.append(img_label)
        self.raw_img_inputs = np.stack(self.raw_img_inputs, axis=0) # (N_dataset, 3, H, W)
        self.img_labels = np.stack(self.img_labels, axis=0) # (N_dataset, 1, H, W)
        
        labelweights, _ = np.histogram(self.img_labels, range(3)) # 0, 1
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        print("The raw "+self.mode+" data is loaded.")

    def augment_input_label(self, raw_img_input, img_input, img_label):
        if random.random()<0.5:
            raw_img_input = tsfm.functional.hflip(raw_img_input)
            img_input = tsfm.functional.hflip(img_input)
            img_label = tsfm.functional.hflip(img_label)
        if random.random()<0.5:
            raw_img_input = tsfm.functional.vflip(raw_img_input)
            img_input = tsfm.functional.vflip(img_input)
            img_label = tsfm.functional.vflip(img_label)
        random_number = random.random()     
        if random_number<0.25:
            raw_img_input = tsfm.functional.rotate(raw_img_input, angle=90)
            img_input = tsfm.functional.rotate(img_input, angle=90)
            img_label = tsfm.functional.rotate(img_label, angle=90)
        elif random_number<0.5:
            raw_img_input = tsfm.functional.rotate(raw_img_input, angle=180)
            img_input = tsfm.functional.rotate(img_input, angle=180)
            img_label = tsfm.functional.rotate(img_label, angle=180)
        elif random_number<0.75:
            raw_img_input = tsfm.functional.rotate(raw_img_input, angle=270)
            img_input = tsfm.functional.rotate(img_input, angle=270)
            img_label = tsfm.functional.rotate(img_label, angle=270)
        return raw_img_input, img_input, img_label

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        """
        - outputs:
            - raw_img_input: tensor, float32, (B, 3, H, W), 0-1
            - img_input: tensor, float32, (B, 3, H, W)
            - img_label: tensor, float32, (B, 1, H, W), binary
            - token: tuple of str, len = B
        """
        token = self.tokens[idx]
        raw_img_input = self.raw_img_inputs[idx]
        raw_img_input = torch.from_numpy(raw_img_input)
        img_input = normalize_imagenet(raw_img_input)
        img_label = torch.from_numpy(self.img_labels[idx])
        if self.augment:
            raw_img_input, img_input, img_label = self.augment_input_label(\
            raw_img_input, img_input, img_label)
        return raw_img_input, img_input, img_label, token