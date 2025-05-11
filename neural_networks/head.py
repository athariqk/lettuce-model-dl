from typing import Callable, Dict, List
from torch import Tensor
import torch.nn as nn
from torchvision.models.detection.ssdlite import (
    SSDLiteClassificationHead,
    SSDLiteRegressionHead,
    SSDScoringHead,
    _prediction_block,
    _normal_init
)

class ModifiedSSDLiteHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_classes: int,
        norm_layer: Callable[..., nn.Module]
    ):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer)
        self.phenotype_head = SSDLitePhenotypeHead(in_channels, num_anchors, norm_layer)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
            "phenotype_regression": self.phenotype_head(x),
        }


class SSDLitePhenotypeHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], norm_layer: Callable[..., nn.Module]):
        phenotype_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            phenotype_reg.append(_prediction_block(channels, 2 * anchors, 3, norm_layer))
        _normal_init(phenotype_reg)
        super().__init__(phenotype_reg, 2)
