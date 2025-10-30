import copy
from typing import Callable, Dict, Optional, OrderedDict
import timm
import torch
import torch.nn as nn
from torchvision.models.detection.ssdlite import _normal_init, _extra_block

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


class SSDLiteDualFeatureExtractorMobileNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Sequential,
        c3_pos: int,
        c4_pos: int,
        norm_layer: Callable[..., nn.Module],
        multimodal,
        width_mult: float = 1.0,
        min_depth: int = 16,
    ):
        super().__init__()
        self.multimodal = multimodal

        if backbone[c3_pos].use_res_connect:
            raise ValueError("backbone[c3_pos].use_res_connect should be False")
        if backbone[c4_pos].use_res_connect:
            raise ValueError("backbone[c4_pos].use_res_connect should be False")

        if multimodal:
            self.features = nn.Sequential(
                # As described in section 6.3 of MobileNetV3 paper
                nn.Sequential(*backbone[:c3_pos], backbone[c3_pos].block[0]),
                nn.Sequential(backbone[c3_pos].block[1:], *backbone[c3_pos + 1 : c4_pos], backbone[c4_pos].block[0]),
                nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1 :]),
            )
        else:
            self.features = nn.Sequential(
                # As described in section 6.3 of MobileNetV3 paper
                nn.Sequential(*backbone[:c4_pos], backbone[c4_pos].block[0]),  # from start until C4 expansion layer
                nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1 :]),  # from C4 depthwise until end
            )
        
        self.features_2 = copy.deepcopy(self.features)
        
        self.aff_0 = AFF(80)
        self.aff_1 = AFF(160)
        self.aff_2 = AFF(960)

        get_depth = lambda d: max(min_depth, int(d * width_mult))  # noqa: E731
        extra = nn.ModuleList(
            [
                _extra_block(backbone[-1].out_channels, get_depth(512), norm_layer),
                _extra_block(get_depth(512), get_depth(256), norm_layer),
                _extra_block(get_depth(256), get_depth(256), norm_layer),
                _extra_block(get_depth(256), get_depth(128), norm_layer),
            ]
        )
        _normal_init(extra)

        self.extra = extra

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Get feature maps from backbone and extra. Can't be refactored due to JIT limitations.
        output = []

        if self.multimodal and y:
            y = self.features_2(y)
            x = self.features[0](x)
            out_c3 = self.aff_0(x, y[0])
            x = self.features[1](out_c3)
            out_c4 = self.aff_1(x, y[1])
            output.append(out_c4)
            x = self.features[2](out_c4)
            out_c5 = self.aff_2(x, y[2])
            output.append(out_c5)
        else:
            for block in self.features:
                x = block(x)
                output.append(x)

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])
