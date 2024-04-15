from os.path import join

import torch
import numpy as np
import torchvision.transforms as tsfm

from unet.models.unet_model import UNet

normalize_imagenet = tsfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class GNGWrapper:
    def __init__(
        self,
        root_dir='.',
        device='cuda',
        surrounding_size=1,
    ):
        """
        - inputs:
            - num_classes: default 2, for path and not path.
        """
        self.model = UNet().to(device)
        model_filepath = join(root_dir, 'results/model_training/unet/checkpoints/best_unet.pt')
        checkpoint = torch.load(model_filepath, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint)
        self.model = self.model.eval()
        self.device = device
        self.surrounding_size = surrounding_size
        print("UNet wrapper is initialized.")
    
    def classify_path_points(
        self,
        binary_mask,
        x_start,
        x_goal,
    ):
        img_height, img_width = binary_mask.shape
        assert img_height % 32 == 0 and img_width % 32 == 0 # * need to satisfy resnet encoder conditions
        input_start = np.zeros_like(binary_mask).astype(np.float32) # (h, w)
        input_start[x_start[1]-self.surrounding_size:x_start[1]+self.surrounding_size+1,
                    x_start[0]-self.surrounding_size:x_start[0]+self.surrounding_size+1] = 1.
        input_goal = np.zeros_like(binary_mask).astype(np.float32) # (h, w)
        input_goal[x_goal[1]-self.surrounding_size:x_goal[1]+self.surrounding_size+1,
                    x_goal[0]-self.surrounding_size:x_goal[0]+self.surrounding_size+1] = 1.
        raw_img_input = np.stack([input_start, input_goal, binary_mask.astype(np.float32)], axis=0) # (3, h, w)
        
        raw_img_input = torch.from_numpy(raw_img_input)
        img_input = normalize_imagenet(raw_img_input) # (3, h, w)
        img_input = img_input.unsqueeze(0).to(self.device) # (1, 3, h, w)
        img_pred_logits = self.model(img_input) # (1, 2, h, w)

        img_path_pred = np.argmax(img_pred_logits.detach().to('cpu').numpy(), 1)[0] # (224, 224)
        img_path_score = torch.softmax(img_pred_logits,dim=1)[0,1].detach().to('cpu').numpy() # (224, 224)
        path_pred_coords = np.flip(np.stack(np.where(img_path_pred==1),axis=1), axis=1) # * (points, 2) coordinates matching x_start and x_goal, not the cv2 image or np h*w matrix
        return path_pred_coords, img_path_pred, img_path_score

