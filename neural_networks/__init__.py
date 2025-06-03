import os
from typing import Any, Optional, List, Tuple, Mapping

import torch
from torch import nn, Tensor
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from custom_types import DualTensor
from .utils import retrieve_out_channels, postprocess_lettuce_detections

from .model import PhenotypeRegressor, SingleBranchLettuceModel, DualBranchLettuceModel
from .backbone import MobileViTV2FeatureExtractor
from .head import ModifiedSSDLiteHead

from my_utils import ROOT_DIR

__all__ = [
    "LettuceModelEval",
    "lettuce_model",
    "lettuce_model_multimodal",
    "lettuce_regressor_model",
]


class LettuceModelEval(nn.Module):
    def __init__(
            self,
            model = None,
            size=(320, 320),
            image_mean: Optional[List[float]] = None,
            image_std: Optional[List[float]] = None,
            phenotype_means: Optional[List[float]] = None,
            phenotype_stds: Optional[List[float]] = None,
            multimodal = False,
            transform = None,
            box_coder = None,
            **kwargs
    ):
        super().__init__()
        if model is None:
            if multimodal:
                model = lettuce_model_multimodal(
                    phenotype_means=phenotype_means, phenotype_stds=phenotype_stds, **kwargs
                )
            else:
                model = lettuce_model(
                    phenotype_means=phenotype_means, phenotype_stds=phenotype_stds, **kwargs
                )
        self.model = model

        self.model.eval()

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        if transform is None:
            transform = GeneralizedRCNNTransform(
                min(size), max(size), image_mean, image_std, size_divisible=1, fixed_size=size, **kwargs
            )
        self.transform = transform

        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder

    def forward(self, images: List[DualTensor | Tensor]):
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        image_tensors: List[Tensor] = [item.x if isinstance(item, DualTensor) else item for item in images]
        aux_tensors: List[Tensor] = [item.y if isinstance(item, DualTensor) else item for item in images]
        images, _ = self.transform(image_tensors)
        aux, _ = self.transform(aux_tensors)
        stacked = torch.stack([images.tensors, aux.tensors], dim=0)

        output = self.model(stacked)

        outputs = postprocess_lettuce_detections(
            output,
            images.image_sizes,
            self.box_coder,
            0.01,
            self.model.phenotype_stds,
            self.model.phenotype_means,
            400,
            200,
            0.5
        )
        outputs = self.transform.postprocess(outputs, images.image_sizes, original_image_sizes)

        return outputs

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        self.model.load_state_dict(state_dict, strict, assign)


def lettuce_model(
        trainable_backbone_layers: Optional[int] = None,
        with_height=True,
        **kwargs: Any
) -> SingleBranchLettuceModel:
    """Loads a model for lettuce growth phenotype estimation"""

    variant = "models/coco-ssd-mobilevitv2-0.75_2nc_pretrained_coremlcompat.pt" if with_height else \
        "models/coco-ssd-mobilevitv2-0.75_2nc_1pheno_pretrained.pt"

    model = SingleBranchLettuceModel(
        size=(320, 320),
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
        pretrained=os.path.join(ROOT_DIR, variant),
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
) -> DualBranchLettuceModel:
    """Loads a multimodal model for lettuce growth phenotype estimation"""

    variant = "models/coco-ssd-mobilevitv2-0.75_2nc_pretrained_coremlcompat.pt"

    model = DualBranchLettuceModel(
        size=(320, 320),
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
        pretrained=os.path.join(ROOT_DIR, variant),
        **kwargs
    )

    if trainable_backbone_layers is not None:
        for parameter in model.model.encoder.parameters():
            parameter.requires_grad_(trainable_backbone_layers >= 2)
        for parameter in model.model.extra_layers.parameters():
            parameter.requires_grad_(trainable_backbone_layers >= 1)

    return model


def baseline_model(
        variant: str,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any
) -> SingleBranchLettuceModel:
    if "80" in variant:
        variant = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_81nc_pretrained.pt")
    elif "90" in variant:
        variant = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_91nc_pretrained.pt")
    elif "2" in variant:
        variant = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_2nc_no-pheno_pretrained.pt")
    else:
        raise ValueError(f"Unexpected variant, got: {variant}")

    model = SingleBranchLettuceModel(
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
