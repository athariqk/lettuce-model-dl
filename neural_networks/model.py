import copy
import os
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection._utils import SSDMatcher
from typing import Dict, List, Optional, Tuple
from cvnets.models.detection.ssd import SingleShotMaskDetector
import torchvision.transforms.v2 as transforms

from my_utils import ROOT_DIR
from neural_networks.blocks import AFF
from custom_types import DualTensor
from neural_networks.custom_types import LettuceDetectionOutputs
from neural_networks.anchors import DefaultBoxGenerator


class ModifiedSSDLiteMobileViTBase(nn.Module):
    """A modified SSDLite-MobileViT architecture for estimating lettuce growth phenotypes"""

    def __init__(
            self,
            aspect_ratios: List[List[int]],
            phenotype_means: Optional[List[float]] = None,
            phenotype_stds: Optional[List[float]] = None,
            pretrained: str = None,
            **kwargs
    ):
        super().__init__()

        if pretrained is None:
            # baseline
            pretrained = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_81nc_pretrained.pt")

        self.model: SingleShotMaskDetector = torch.load(pretrained, weights_only=False)

        if phenotype_means is None:
            phenotype_means = [0.0, 0.0]
        if phenotype_stds is None:
            phenotype_stds = [1.0, 1.0]
        self.register_buffer("phenotype_stds", torch.Tensor(phenotype_stds).unsqueeze(0))
        self.register_buffer("phenotype_means", torch.Tensor(phenotype_means).unsqueeze(0))

        self.anchor_generator = DefaultBoxGenerator(aspect_ratios, min_ratio=0.1, max_ratio=1.05)

    def forward(self, x: Tensor) -> LettuceDetectionOutputs:
        raise NotImplementedError


class SingleBranchLettuceModel(ModifiedSSDLiteMobileViTBase):
    def forward(self, images: Tensor) -> LettuceDetectionOutputs:
        features = self.model.get_backbone_features(images)

        cls_logits, bbox_regression, _, phenotypes_pred = self.model.ssd_forward(
            features, device=images.device
        )

        anchors = self.anchor_generator(images, list(features.values()))

        head_outputs = LettuceDetectionOutputs(
            cls_logits=cls_logits,
            bbox_regression=bbox_regression,
            phenotypes_pred=phenotypes_pred,
            anchors=anchors
        )

        return head_outputs


class DualBranchLettuceModel(ModifiedSSDLiteMobileViTBase):
    def __init__(self, aspect_ratios: List[List[int]], **kwargs):
        super().__init__(aspect_ratios, **kwargs)
        self.aux_encoder = copy.deepcopy(self.model.encoder)

        self.aff_0 = AFF(self.model.enc_l3_channels)
        self.aff_1 = AFF(self.model.enc_l4_channels)
        self.aff_2 = AFF(self.model.enc_l5_channels)

    def get_backbone_features(self, x_main: Tensor, x_aux: Tensor) -> Dict[str, Tensor]:
        aux_enc_features = self.aux_encoder.extract_end_points_all(x_aux)

        x = self.model.encoder.conv_1(x_main)
        x = self.model.encoder.layer_1(x)
        x = self.model.encoder.layer_2(x)
        x = self.model.encoder.layer_3(x)
        out_l3 = self.aff_0(x, aux_enc_features["out_l3"])
        x = self.model.encoder.layer_4(out_l3)
        out_l4 = self.aff_1(x, aux_enc_features["out_l4"])
        x = self.model.encoder.layer_5(out_l4)
        out_l5 = self.aff_2(x, aux_enc_features["out_l5"])

        end_points: Dict = dict()
        for idx, os in enumerate(self.model.output_strides):
            if os == 8:
                end_points["os_{}".format(os)] = out_l3
            elif os == 16:
                end_points["os_{}".format(os)] = out_l4
            elif os == 32:
                end_points["os_{}".format(os)] = out_l5
            else:
                x = end_points["os_{}".format(self.model.output_strides[idx - 1])]
                end_points["os_{}".format(os)] = self.model.extra_layers["os_{}".format(os)](
                    x
                )

        return end_points

    def forward(self, images: Tensor) -> LettuceDetectionOutputs:
        x, y = images.unbind()

        features = self.get_backbone_features(x, y)

        cls_logits, bbox_regression, _, phenotypes_pred = self.model.ssd_forward(
            features, device=x.device
        )

        anchors = self.anchor_generator(images, list(features.values()))

        head_outputs = LettuceDetectionOutputs(
            cls_logits=cls_logits,
            bbox_regression=bbox_regression,
            phenotypes_pred=phenotypes_pred,
            anchors=anchors
        )

        return head_outputs


def mobilenet_branch():
    mnet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    mnet.classifier = nn.Identity()
    return mnet


class PhenotypeRegressor(nn.Module):
    def __init__(
            self,
            dual_branch=False,
            size: Tuple[int, int]=(256, 256),
            image_mean=None,
            image_std=None,
    ):
        super().__init__()
        self.X = mobilenet_branch()
        self.Y = mobilenet_branch() if dual_branch else nn.Identity()

        self.n_regression_outputs = 2
        in_features = 576 * (2 if dual_branch else 1)
        self.fcn = nn.Linear(in_features, self.n_regression_outputs)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.Normalize(image_mean, image_std)
        ])

        self.dual_branch = dual_branch


    def forward(self, input: DualTensor | Tensor) -> Tensor:
        # transform the input
        x = self.transform(input.x if isinstance(input, DualTensor) else input)

        features = self.X(x)

        if self.dual_branch and isinstance(input, DualTensor):
            y_transformed = self.transform(input.y)
            y_features = self.Y(y_transformed)  # Should output [B, 576]
            features = torch.cat((features, y_features), dim=1)  # Concatenate [B, 576] and [B, 576]

        out = self.fcn(features)

        return out
