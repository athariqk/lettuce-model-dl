import copy
from functools import partial
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection._utils import BoxCoder, _topk_min
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import boxes as box_ops
from torchvision.models.detection._utils import SSDMatcher
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Sequence, NamedTuple
from cvnets.models.detection.ssd import SingleShotMaskDetector
import torchvision.transforms.v2 as transforms
from cvnets.models.classification.base_image_encoder import BaseImageEncoder
import torchvision.models.detection._utils as det_utils
from torchvision.models.detection.transform import ImageList
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights, _ovewrite_value_param, _validate_trainable_layers, _normal_init
from torchvision.models.detection.ssd import SSD

from my_utils import ROOT_DIR
from neural_networks.backbone import SSDLiteDualFeatureExtractorMobileNet
from neural_networks.blocks import AFF
from custom_types import DualTensor
from neural_networks.head import ModifiedSSDLiteHead


class HeadOutputs(NamedTuple):
    cls_logits: torch.Tensor      # Expected shape: [B, N, NumClasses]
    bbox_regression: torch.Tensor # Expected shape: [B, N, 4]
    phenotypes_pred: torch.Tensor # Expected shape: [B, N, NumPhenotypes]


class Modified_SSDLiteMobileViT(nn.Module):
    """A modified SSDLite-MobileViT architecture for estimating lettuce growth phenotypes"""

    def __init__(
            self,
            size: Tuple[int, int],
            aspect_ratios: List[List[int]],
            image_mean: Optional[List[float]] = None,
            image_std: Optional[List[float]] = None,
            phenotype_means: Optional[List[float]] = None,
            phenotype_stds: Optional[List[float]] = None,
            num_phenotypes: int = 2,
            boxcox_lambdas: Optional[List[float]] = None,
            minimums: Optional[List[float]] = None,
            maximums: Optional[List[float]] = None,
            log_transform: bool = False,
            score_thresh: float = 0.01,
            nms_thresh: float = 0.5,
            detections_per_img: int = 200,
            topk_candidates: int = 400,
            iou_thresh: float = 0.5,
            pretrained: str |None = None,
            phenotype_loss_weight: float = 0.0001,
            multimodal = False,
            **kwargs
    ):
        super().__init__()

        if pretrained is None:
            # baseline
            pretrained = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_81nc_pretrained.pt")

        self.model: SingleShotMaskDetector = torch.load(pretrained, weights_only=False)

        self.aux_encoder: BaseImageEncoder | nn.Identity = copy.deepcopy(self.model.encoder) if multimodal else nn.Identity()

        self.aff_0 = AFF(self.model.enc_l3_channels) if multimodal else nn.Identity()
        self.aff_1 = AFF(self.model.enc_l4_channels) if multimodal else nn.Identity()
        self.aff_2 = AFF(self.model.enc_l5_channels) if multimodal else nn.Identity()

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(
            min(size), max(size), image_mean, image_std, size_divisible=1, fixed_size=size, **kwargs
        )

        if phenotype_means is None:
            phenotype_means = [0.0] * num_phenotypes
        if phenotype_stds is None:
            phenotype_stds = [1.0] * num_phenotypes
        self.register_buffer("phenotype_stds", torch.Tensor(phenotype_stds).unsqueeze(0))
        self.register_buffer("phenotype_means", torch.Tensor(phenotype_means).unsqueeze(0))

        if boxcox_lambdas is not None:
            self.register_buffer("boxcox_lambdas", torch.Tensor(boxcox_lambdas).unsqueeze(0))
        if minimums is not None:
            self.register_buffer("minimums", torch.Tensor(minimums).unsqueeze(0))
        if maximums is not None:
            self.register_buffer("maximums", torch.Tensor(maximums).unsqueeze(0))

        self.proposal_matcher = SSDMatcher(iou_thresh)
        self.anchor_generator = DefaultBoxGenerator(aspect_ratios, min_ratio=0.1, max_ratio=1.05)
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        # Anchor box related parameters
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        self.neg_to_pos_ratio = 3
        self.label_smoothing = 0.3

        self.phenotype_loss_weight = phenotype_loss_weight
        self.multimodal = multimodal
        
        self.log_transform = log_transform

    # @torch.jit.unused
    def eager_outputs(
            self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        if self.training:
            return losses

        return detections

    def compute_loss(
            self,
            head_outputs: HeadOutputs,
            targets: List[Dict[str, Tensor]],
            anchors: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Computes SSD loss, similar to TorchVision's implementation.
        Args:
            targets (List[Dict[str, Tensor]]): Ground truth, list of dicts with "boxes" and "labels".
                                               Optionally "phenotypes" if you have phenotype targets.
            head_outputs (HeadOutputs): Outputs from SSD heads.
                                             {"cls_logits": [B, N, num_classes],
                                              "bbox_regression": [B, N, 4],
                                              "phenotypes_pred": [B, N, num_phenotypes] (optional)}
            anchors (Tensor): Default boxes from CVNet model, shape [B, N, 4].
        """

        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full(
                        (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                    )
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        bbox_regression = head_outputs.bbox_regression
        cls_logits = head_outputs.cls_logits
        phenotypes_pred = head_outputs.phenotypes_pred

        model_outputs_phenotypes = phenotypes_pred.shape[-1] > 0

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        phenotype_loss = []
        for (
                targets_per_image,
                bbox_regression_per_image,
                cls_logits_per_image,
                phenotypes_pred_per_image,
                anchors_per_image,
                matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, phenotypes_pred, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets.append(gt_classes_target)

            # Calculate phenotype loss (only for foreground objects)
            if "phenotypes" in targets_per_image and foreground_idxs_per_image.numel() > 0 and model_outputs_phenotypes:
                matched_phenotypes = targets_per_image["phenotypes"][foreground_matched_idxs_per_image]
                pred_phenotypes = phenotypes_pred_per_image[foreground_idxs_per_image]
                phenotype_loss_per_image = torch.nn.functional.mse_loss(
                    pred_phenotypes, matched_phenotypes, reduction="sum"
                )
                phenotype_loss.append(phenotype_loss_per_image)
            else:
                phenotype_loss.append(torch.tensor(0.0, device=bbox_regression.device))

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)
        phenotype_loss = torch.stack(phenotype_loss)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(
            cls_targets.size()
        )

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            "bbox_loss": bbox_loss.sum() / N,
            "cls_loss": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
            "phenotype_loss": (phenotype_loss.sum() / N) * self.phenotype_loss_weight,
        }

    def get_backbone_features(self, x_main: Tensor, x_aux: Tensor) -> Dict[str, Tensor]:
        if isinstance(self.aux_encoder, BaseImageEncoder):
            aux_enc_features = self.aux_encoder.extract_end_points_all(x_aux)
        else:
            # Handle Identity case by creating empty features
            aux_enc_features = {"out_l3": x_aux, "out_l4": x_aux, "out_l5": x_aux}

        x = self.model.encoder.conv_1(x_main) # type: ignore
        x = self.model.encoder.layer_1(x) # type: ignore
        x = self.model.encoder.layer_2(x) # type: ignore
        x = self.model.encoder.layer_3(x) # type: ignore
        out_l3 = self.aff_0(x, aux_enc_features["out_l3"])
        x = self.model.encoder.layer_4(out_l3) # type: ignore
        out_l4 = self.aff_1(x, aux_enc_features["out_l4"])
        x = self.model.encoder.layer_5(out_l4) # type: ignore
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
                end_points["os_{}".format(os)] = self.model.extra_layers["os_{}".format(os)]( # type: ignore
                    x
                )

        if self.model.fpn is not None:
            # apply Feature Pyramid Network
            end_points = self.model.fpn(end_points)

        return end_points

    def forward(
            self, images: List[DualTensor | Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> (
            Tuple[Dict[str, Tensor],
            List[Dict[str, Tensor]]] |
            Dict[str, Tensor] |
            List[Dict[str, Tensor]]
    ):
        """
        Returns:
            A (Losses, Detections) tuple if in scripting, otherwise `Losses` if in training mode and `Detections`
            if not in training mode
        """

        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

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
        images_transformed, targets_transformed = self.transform(image_tensors, targets)

        # Check for degenerate boxes
        if targets_transformed is not None:
            for target_idx, target in enumerate(targets_transformed):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        if self.multimodal:
            aux_tensors: List[Tensor] = [item.y if isinstance(item, DualTensor) else item for item in images]
            aux_images_transformed, _ = self.transform(aux_tensors)
            features = self.get_backbone_features(images_transformed.tensors, aux_images_transformed.tensors)
        else:
            features = self.model.get_backbone_features(images_transformed.tensors)

        cls_logits, bbox_regression, _, phenotypes_pred = self.model.ssd_forward(
            features, device=images_transformed.tensors.device
        )

        head_outputs = HeadOutputs(
            cls_logits=cls_logits,
            bbox_regression=bbox_regression,
            phenotypes_pred=phenotypes_pred
        )

        # create the set of anchors
        anchors = self.anchor_generator(images_transformed, list(features.values()))

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets_transformed is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                losses = self.compute_loss(head_outputs, targets_transformed, anchors)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images_transformed.image_sizes)
            detections = self.transform.postprocess(detections, images_transformed.image_sizes, original_image_sizes)
            # returns a list of detections

        return self.eager_outputs(losses, detections)

    def postprocess_detections(
            self, head_outputs: HeadOutputs, image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs.bbox_regression
        pred_scores = F.softmax(head_outputs.cls_logits, dim=-1)
        phenotypes_pred = head_outputs.phenotypes_pred

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, phenotypes, anchors, image_shape in zip(bbox_regression, pred_scores, phenotypes_pred,
                                                                   image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            image_phenotypes = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]
                phenotype = phenotypes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = _topk_min(score, self.topk_candidates, 0)
                score, idxs = score.topk(num_topk)
                box = box[idxs]
                phenotype = phenotype[idxs]

                # transform to original scale
                if hasattr(self, "boxcox_lambdas"):
                    phenotype = torch.pow(phenotype * self.boxcox_lambdas + 1, 1 / self.boxcox_lambdas)
                if hasattr(self, "minimums") and hasattr(self, "maximums"):
                    phenotype = (phenotype * (self.maximums - self.minimums)) + self.minimums  # min max scaling
                if hasattr(self, "phenotype_means") and hasattr(self, "phenotype_stds"):
                    phenotype = (phenotype * self.phenotype_stds) + self.phenotype_means

                if self.log_transform:
                    # reverse log transform if applied
                    phenotype = torch.exp(phenotype)

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))
                image_phenotypes.append(phenotype)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_phenotypes = torch.cat(image_phenotypes, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                    "phenotypes": image_phenotypes[keep]
                }
            )
        return detections


def _mobilenet_extractor(
    backbone,
    trainable_layers: int,
    norm_layer: Callable[..., nn.Module],
    multimodal,
):
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we won't freeze
    if not 0 <= trainable_layers <= num_stages:
        raise ValueError("trainable_layers should be in the range [0, {num_stages}], instead got {trainable_layers}")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDLiteDualFeatureExtractorMobileNet(backbone, stage_indices[-3], stage_indices[-2], norm_layer, multimodal)


def ssdlite320_dual_mobilenet_v3_large(
    *,
    weights: Optional[SSDLite320_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    multimodal = True,
    **kwargs: Any,
) -> SSD:
    weights = SSDLite320_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the parameter.")

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 6, 6
    )

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = weights_backbone is None

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = mobilenet_v3_large(
        weights=weights_backbone, progress=progress, norm_layer=norm_layer, reduced_tail=reduce_tail, **kwargs
    )
    if weights_backbone is None:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)
    backbone = _mobilenet_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer,
        multimodal,
    )

    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
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
    model = SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=ModifiedSSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )
    
    def modified_forward(self: SSD, images: List[DualTensor | Tensor], targets: Optional[List[Dict[str, Tensor]]] = None):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images = [item.x if isinstance(item, DualTensor) else item for item in images]
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        if multimodal:
            aux_tensors: List[Tensor] = [item.y if isinstance(item, DualTensor) else item for item in images]
            aux_images, _ = self.transform(aux_tensors)
            features = self.backbone(images.tensors, aux_images.tensors)
        else:
            # get the features from the backbone
            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

            features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            matched_idxs = []
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for anchors_per_image, targets_per_image in zip(anchors, targets):
                    if targets_per_image["boxes"].numel() == 0:
                        matched_idxs.append(
                            torch.full(
                                (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                            )
                        )
                        continue

                    match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                    matched_idxs.append(self.proposal_matcher(match_quality_matrix))

                losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            # returns a list of detections

        return self.eager_outputs(losses, detections)

    model.forward = modified_forward.__get__(model, SSD)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def mobilenet_branch():
    mnet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    mnet.classifier = nn.Identity() # type: ignore
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


    def forward(self, input: DualTensor) -> Tensor:
        # transform the input
        x = self.transform(input.x)

        features = self.X(x)

        if self.dual_branch:
            y_transformed = self.transform(input.y)
            y_features = self.Y(y_transformed)  # Should output [B, 576]
            features = torch.cat((features, y_features), dim=1)  # Concatenate [B, 576] and [B, 576]

        out = self.fcn(features)

        return out
