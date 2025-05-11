from functools import partial
import torch.nn as nn
from typing import Any, Callable, Optional
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

from .utils import retrieve_out_channels

from .model import ModifiedSSD
from .backbone import MobileViTV2FeatureExtractor
from .head import ModifiedSSDLiteHead

__all__ = [
  "ssdlite_mobilevit_unimodal"
  "ssdlite_mobilevit_multimodal"
]

def ssdlite_mobilevit_unimodal(
    num_classes: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any
) -> SSD:
    if num_classes is None:
        num_classes = 91

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = MobileViTV2FeatureExtractor(norm_layer=norm_layer, dual_backbone=False)

    size = (320, 320)
    out_channels = retrieve_out_channels(backbone, size, dual_backbone=False)
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
    model = ModifiedSSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=ModifiedSSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )
    
    return model

def ssdlite_mobilevit_multimodal(
    num_classes: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any
) -> ModifiedSSD:
  if num_classes is None:
    num_classes = 91

  if norm_layer is None:
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

  backbone = MobileViTV2FeatureExtractor(norm_layer=norm_layer, dual_backbone=True)

  size = (320, 320)
  out_channels = retrieve_out_channels(backbone, size, dual_backbone=True)
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
  model = ModifiedSSD(
      backbone,
      anchor_generator,
      size,
      num_classes,
      head=ModifiedSSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
      **kwargs,
  )
  
  return model
