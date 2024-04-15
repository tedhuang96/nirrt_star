import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x



class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels, up_conv_out_channels,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self,
        feature_channels_list,
        upsampling_method="conv_transpose",
    ):
        super(UNetDecoder, self).__init__()
        self.generate_channels_list(feature_channels_list)
        self.up_blocks = nn.ModuleList() # num_channels
        for in_channels, out_channels, up_conv_in_channels, up_conv_out_channels in \
            zip(self.in_channels_list, self.out_channels_list, \
                self.up_conv_in_channels_list, self.up_conv_out_channels_list):
            self.up_blocks.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    up_conv_in_channels=up_conv_in_channels,
                    up_conv_out_channels=up_conv_out_channels,
                    upsampling_method=upsampling_method,
                )
            )
        self.out = nn.Conv2d(self.out_channels_list[-1], 2, kernel_size=1, stride=1)
    
    def generate_channels_list(self, feature_channels_list):
        """
        inputs:
            - feature_channels_list: [c1,c2,c3,c4,c5,c6]
        outputs:
            - in_channels_list: [2*c5,2*c4,2*c3,2*c2,2*c1]
            - out_channels_list: [c5,c4,c3,c2,c1]
            - up_conv_in_channels_list: [c6,c5,c4,c3,c2]
            - up_conv_out_channels_list: [c5,c4,c3,c2,c1]
        """
        reverse_feature_channels_list = feature_channels_list[::-1]
        self.feature_channels_list = feature_channels_list
        self.in_channels_list = [2*c for c in reverse_feature_channels_list[1:]]
        self.out_channels_list = reverse_feature_channels_list[1:]
        self.up_conv_in_channels_list = reverse_feature_channels_list[:-1]
        self.up_conv_out_channels_list = reverse_feature_channels_list[1:]
        
    def forward(self, features):
        """
        inputs:
            - features: list of tensors
                torch.Size([batch_size, c1, H, W])
                torch.Size([batch_size, c2, H/2, W/2])
                torch.Size([batch_size, c3, H/4, W/4])
                torch.Size([batch_size, c4, H/8, W/8])
                torch.Size([batch_size, c5, H/16, W/16])
                torch.Size([batch_size, c6, H/32, W/32])
        outputs:
            - x: tensor (batch_size, 1, H, W)
        """
        reverse_features = features[::-1]
        x = reverse_features[0]
        for feature, up_block in zip(reverse_features[1:], self.up_blocks):
            x = up_block(x, feature)
        x = self.out(x)
        return x