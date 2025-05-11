from typing import Callable, OrderedDict
import timm
import torch
import torch.nn as nn
from torchvision.models.detection.ssdlite import _normal_init

from .blocks import AFF, DWSeparableConvBlock

class RGBStream(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RGBStream, self).__init__(*args, **kwargs)
        mobilevitv2 = timm.create_model(
          'mobilevitv2_075.cvnets_in1k',
          pretrained=True,
          features_only=True
        )
        
        self.stem = mobilevitv2.stem
        self.stages_0 = mobilevitv2.stages_0
        self.stages_1 = mobilevitv2.stages_1
        self.stages_2 = mobilevitv2.stages_2
        self.stages_3 = mobilevitv2.stages_3
        self.stages_4 = mobilevitv2.stages_4
        self.final_conv = mobilevitv2.final_conv
        
        self.aff_0 = AFF(mobilevitv2.feature_info.info[-3]["num_chs"])
        self.aff_1 = AFF(mobilevitv2.feature_info.info[-2]["num_chs"])
        self.aff_2 = AFF(mobilevitv2.feature_info.info[-1]["num_chs"])
        _normal_init(self.aff_0)
        _normal_init(self.aff_1)
        _normal_init(self.aff_2)
        
        self.feature_info = mobilevitv2.feature_info
        
    def forward(self, x, depth_1, depth_2, depth_3):
        x = self.stem(x)
        x = self.stages_0(x)
        x = self.stages_1(x)
        x = self.stages_2(x)
        x_1 = self.aff_0(x, depth_1)
        x = self.stages_3(x_1)
        x_2 = self.aff_1(x, depth_2)
        x = self.stages_4(x_2)
        x = self.aff_2(x, depth_3)
        x_3 = self.final_conv(x)
        return [x_2, x_3] # SSD only needs the last 2 feature maps

class MobileViTV2FeatureExtractor(nn.Module):
    def __init__(
      self: str,
      norm_layer: Callable[..., nn.Module],
      dual_backbone: bool = False,
    ):
      super().__init__()

      # the depth stream is unmodified
      self.aux = timm.create_model(
        'mobilevitv2_075.cvnets_in1k',
        pretrained=True,
        features_only=True
      )

      # infused with fusion blocks
      self.main = RGBStream() if dual_backbone else nn.Identity()
      
      final_numchannel = -1
      if dual_backbone:
          final_numchannel = self.main.feature_info.info[-1]["num_chs"]
      else:
          final_numchannel = self.aux.feature_info.info[-1]["num_chs"]

      self.extra = nn.ModuleList(
        [
          DWSeparableConvBlock(final_numchannel, 512, 3, norm_layer),
          DWSeparableConvBlock(512, 256, 3, norm_layer),
          DWSeparableConvBlock(256, 256, 3, norm_layer),
          DWSeparableConvBlock(256, 128, 3, norm_layer),
        ]
      )
      _normal_init(self.extra)
      
      self.dual_backbone = dual_backbone

    def forward(self, x: torch.Tensor, aux: torch.Tensor = None):
        output = []
        
        if self.dual_backbone:
            aux = self.aux(aux)

            x = self.main(x, aux[-3], aux[-2], aux[-1])
            output.extend(x)
            x = x[-1]
        else:
            # use the unmodified backbone
            x = self.aux(x)
            output.extend(x[-2:])
            x = x[-1]

        for block in self.extra:
          x = block(x)
          output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])
