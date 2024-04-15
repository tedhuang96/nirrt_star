import torchvision
import torch.nn as nn
from torchvision.models import ResNet50_Weights

class ResNet50Encoder(nn.Module):
    def __init__(
        self,
        freeze_original_weights=True,
    ):
        super(ResNet50Encoder, self).__init__()
        resnet = torchvision.models.resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if freeze_original_weights:
            for child in resnet.children():
                for param in child.parameters():
                    param.requires_grad = False
        down_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        """
        inputs:
            - x: tensor. (batch_size, 3, H, W)
        outputs:
            - features: list of 6 tensors.
                x: tensor. (batch_size, 3, H, W)
                torch.Size([batch_size, 64, H/2, W/2])
                torch.Size([batch_size, 256, H/4, W/4])
                torch.Size([batch_size, 512, H/8, W/8])
                torch.Size([batch_size, 1024, H/16, W/16])
                torch.Size([batch_size, 2048, H/32, W/32])
        """
        features = [x]
        x = self.input_block(x)
        features.append(x)
        x = self.input_pool(x)
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        return features