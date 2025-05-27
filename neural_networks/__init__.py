import os
from functools import partial
import torch
import torch.nn as nn
from typing import Any, Callable, Optional

from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

from .utils import retrieve_out_channels

from .model import Modified_SSDLiteMobileViT, PhenotypeRegressor
from .backbone import MobileViTV2FeatureExtractor
from .head import ModifiedSSDLiteHead

from my_utils import ROOT_DIR

__all__ = [
    "lettuce_model",
    "lettuce_model_multimodal",
    "lettuce_regressor_model",
]


def lettuce_model(
        trainable_backbone_layers: Optional[int] = None,
        multimodal = False,
        **kwargs: Any
) -> Modified_SSDLiteMobileViT:
    'Loads a model for lettuce growth phenotype estimation'

    model = Modified_SSDLiteMobileViT(
        size=(320, 320),
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        pretrained=os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_2nc_pretrained.pt"),
        **kwargs
    )

    if trainable_backbone_layers is not None:
        for parameter in model.model.encoder.parameters():
            parameter.requires_grad_(trainable_backbone_layers >= 2)
        for parameter in model.model.extra_layers.parameters():
            parameter.requires_grad_(trainable_backbone_layers >= 1)

    return model


def lettuce_model_multimodal(
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any
) -> Modified_SSDLiteMobileViT:
    'Loads a multimodal model for lettuce growth phenotype estimation'
    return lettuce_model(trainable_backbone_layers=trainable_backbone_layers, multimodal=True, **kwargs)


def baseline_model(
        variant: str,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any
) -> Modified_SSDLiteMobileViT:
    if "80" in variant:
        variant = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_81nc_pretrained.pt")
    elif "90" in variant:
        variant = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_91nc_pretrained.pt")
    elif "2" in variant:
        variant = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_2nc_no-pheno_pretrained.pt")
    else:
        raise ValueError(f"Unexpected variant, got: {variant}")

    model = Modified_SSDLiteMobileViT(
        size=(320, 320),
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        pretrained=variant,
        **kwargs
    )
    return model


def lettuce_regressor_model(dual_branch=False):
    return PhenotypeRegressor(dual_branch=dual_branch)
