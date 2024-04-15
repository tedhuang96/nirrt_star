import torch.nn as nn

from unet.models.unet_decoder import UNetDecoder
from unet.models.resnet_encoder import ResNet50Encoder


class UNet(nn.Module):
    def __init__(self,
        feature_channels_list=(3,64,256,512,1024,2048),
        upsampling_method="conv_transpose",
        freeze_original_weights=True,
    ):
        super(UNet, self).__init__()
        self.encoder = ResNet50Encoder(
            freeze_original_weights,
        )
        self.decoder = UNetDecoder(
            feature_channels_list,
            upsampling_method,
        )
    
    def forward(self, inputs):
        """
        - inputs: (batch_size, 3, 224, 224)
        - outputs: (batch_size, 1, 224, 224)
        """
        features = self.encoder(inputs)
        outputs = self.decoder(features)
        return outputs