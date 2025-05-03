from functools import partial
import torch
import torch.nn as nn
import timm
import torchvision.models.detection._utils as det_utils
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops.misc import Conv2dNormActivation
from typing import Any, Optional, Callable, OrderedDict

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
      norm_layer=norm_layer
    )
    self.pw_conv = Conv2dNormActivation(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=1,
      bias=False,
      norm_layer=norm_layer
    )

  def forward(self, x: torch.Tensor):
    x = self.dw_conv(x)
    x = self.pw_conv(x)
    return x

class MobileViTV2FeatureExtractor(nn.Module):
    def __init__(
      self: str,
      norm_layer: Callable[..., nn.Module],
    ):
      super().__init__()
      # Muat model MobileViTV2 dasar
      self.mobilevitv2 = timm.create_model(
        'mobilevitv2_075.cvnets_in1k',
        pretrained=True,
        features_only=True
      )

      self.extra = nn.ModuleList(
        [
          DWSeparableConvBlock(self.mobilevitv2.feature_info.info[-1]["num_chs"], 512, 3, norm_layer),
          DWSeparableConvBlock(512, 256, 3, norm_layer),
          DWSeparableConvBlock(256, 256, 3, norm_layer),
          DWSeparableConvBlock(256, 128, 3, norm_layer),
        ]
      )

    def forward(self, x: torch.Tensor):
      output = []
      
      features = self.mobilevitv2(x)
      output.extend(features[-2:])
      x = features[-1]

      for block in self.extra:
        x = block(x)
        output.append(x)

      return OrderedDict([(str(i), v) for i, v in enumerate(output)])

def ssdlite_mobilevit(
    num_classes: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any
) -> SSD:
  if num_classes is None:
    num_classes = 91

  if norm_layer is None:
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

  backbone = MobileViTV2FeatureExtractor(norm_layer=norm_layer)

  size = (320, 320)
  out_channels = det_utils.retrieve_out_channels(backbone, size)
  anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(len(out_channels))], min_ratio=0.2, max_ratio=0.95)
  num_anchors = anchor_generator.num_anchors_per_location()
  if len(out_channels) != len(anchor_generator.aspect_ratios):
    raise ValueError(
      f"The length of the output channels from the backbone {len(out_channels)} do not match the length of the anchor generator aspect ratios {len(anchor_generator.aspect_ratios)}"
    )

  defaults = {
    "score_thresh": 0.001,
    "nms_thresh": 0.55,
    "detections_per_img": 300,
    "topk_candidates": 300,
    # Rescale the input in a way compatible to the backbone:
    # The following mean/std rescale the data from [0, 1] to [-1, 1]
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
  }
  kwargs: Any = {**defaults, **kwargs}
  model = SSD(
      backbone,
      anchor_generator,
      size,
      num_classes,
      head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
      **kwargs,
  )
  
  return model