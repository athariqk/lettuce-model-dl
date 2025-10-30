import os
from functools import partial
import torch
import torch.nn as nn
from typing import Any, Callable, Optional

from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

from .model import ssdlite320_dual_mobilenet_v3_large, Modified_SSDLiteMobileViT, PhenotypeRegressor

from my_utils import ROOT_DIR

__all__ = [
    "lettuce_model",
    "lettuce_model_multimodal",
    "lettuce_regressor_model",
]


def lettuce_model(
        trainable_backbone_layers: Optional[int] = None,
        multimodal=False,
        num_phenotypes=2,
        **kwargs: Any
) -> Modified_SSDLiteMobileViT:
    """Loads a model for lettuce growth phenotype estimation"""

    if num_phenotypes == 1:
        variant = "models/coco-ssd-mobilevitv2-0.75_2nc_1np_pretrained.pt"
    elif num_phenotypes == 2:
        variant = "models/coco-ssd-mobilevitv2-0.75_2nc_2np_pretrained.pt"
    elif num_phenotypes == 3:
        variant = "models/coco-ssd-mobilevitv2-0.75_2nc_3np_pretrained.pt"
    elif num_phenotypes == 4:
        variant = "models/coco-ssd-mobilevitv2-0.75_2nc_4np_pretrained.pt"
    elif num_phenotypes == 5:
        variant = "models/coco-ssd-mobilevitv2-0.75_2nc_5np_pretrained.pt"
    else:
        raise ValueError(f"Unexpected number of phenotypes, expected [1, 2, 3, 4 or 5], got: {num_phenotypes}")

    model = Modified_SSDLiteMobileViT(
        size=(320, 320),
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        pretrained=os.path.join(ROOT_DIR, variant),
        multimodal=multimodal,
        num_phenotypes=num_phenotypes,
        **kwargs
    )

    if trainable_backbone_layers is not None:
        for parameter in model.model.encoder.parameters():
            parameter.requires_grad_(trainable_backbone_layers >= 2)
        if model.model.extra_layers:
            for parameter in model.model.extra_layers.parameters():
                parameter.requires_grad_(trainable_backbone_layers >= 1)

    return model


def lettuce_model_multimodal(
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any
) -> Modified_SSDLiteMobileViT:
    """Loads a multimodal model for lettuce growth phenotype estimation"""
    return lettuce_model(trainable_backbone_layers=trainable_backbone_layers, multimodal=True, **kwargs)


def lettuce_model_multimodal_mobnetv3(multimodal=True, **kwargs: Any):
    """Loads a multimodal model for lettuce growth phenotype estimation with MobileNetV3 backbone"""
    return ssdlite320_dual_mobilenet_v3_large(multimodal=multimodal, **kwargs)


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
