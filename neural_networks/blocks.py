import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
from typing import Callable

class DWSeparableConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        norm_layer: Callable[..., nn.Module],
    ):
        super(DWSeparableConvBlock, self).__init__()
        self.dw_conv = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            groups=in_channels,
            bias=False,
            norm_layer=norm_layer,
        )
        self.pw_conv = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class AFF(nn.Module):
    """
    多特征融合 AFF
    """

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
