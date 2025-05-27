import math
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection._utils import BoxCoder, _topk_min
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import boxes as box_ops
from torchvision.models.detection._utils import SSDMatcher
from typing import Dict, List, Optional, Tuple
from cvnets.models.detection.ssd import SingleShotMaskDetector

from dataset import RGBDepthTensor
from my_utils import ROOT_DIR
from neural_networks.blocks import AFF


class Modified_SSDLiteMobileViT(nn.Module):
    """A modified SSDLite-MobileViT architecture for estimating lettuce growth phenotypes"""

    def __init__(
            self,
            size: Tuple[int, int],
            aspect_ratios: List[List[int]],
            image_mean: Optional[List[float]] = None,
            image_std: Optional[List[float]] = None,
            score_thresh: float = 0.01,
            nms_thresh: float = 0.5,
            detections_per_img: int = 200,
            topk_candidates: int = 400,
            iou_thresh: float = 0.5,
            pretrained: str = None,
            phenotype_loss_weight: float = 0.0001,
            **kwargs
    ):
        super().__init__()

        if pretrained is None:
            # baseline
            pretrained = os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_81nc_pretrained.pt")

        self.model: SingleShotMaskDetector = torch.load(pretrained, weights_only=False)

        self.aux_encoder = copy.deepcopy(self.model.encoder)

        self.aff_0 = AFF(self.model.enc_l3_channels)
        self.aff_1 = AFF(self.model.enc_l4_channels)
        self.aff_2 = AFF(self.model.enc_l5_channels)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(
            min(size), max(size), image_mean, image_std, size_divisible=1, fixed_size=size, **kwargs
        )

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

    @torch.jit.unused
    def eager_outputs(
            self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        if self.training:
            return losses

        return detections

    def compute_loss(
            self,
            head_outputs: Dict[str, Tensor],
            targets: List[Dict[str, Tensor]],
            anchors: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Computes SSD loss, similar to TorchVision's implementation.
        Args:
            targets (List[Dict[str, Tensor]]): Ground truth, list of dicts with "boxes" and "labels".
                                               Optionally "phenotypes" if you have phenotype targets.
            head_outputs (Dict[str, Tensor]): Outputs from SSD heads.
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

        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]
        phenotypes_pred = head_outputs["phenotypes_pred"]

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
            if "phenotypes" in targets_per_image and foreground_idxs_per_image.numel() > 0:
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

    def forward(
        self, images: List[RGBDepthTensor], targets: Optional[List[Dict[str, Tensor]]] = None
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
            val = img.x.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.x.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        image_tensors: List[Tensor] = [item.x for item in images]
        aux_tensors: List[Tensor] = [item.y for item in images]

        # transform the input
        images_transformed, targets_transformed = self.transform(image_tensors, targets)
        aux_images_transformed, _ = self.transform(aux_tensors)

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

        features = self.get_backbone_features(images_transformed.tensors, aux_images_transformed.tensors)

        cls_logits, bbox_regression, _, phenotypes_pred = self.model.ssd_forward(
            features, device=images_transformed.tensors.device
        )

        head_outputs = {
            "cls_logits": cls_logits,           # [B, N, NumClasses]
            "bbox_regression": bbox_regression, # [B, N, 4]
            "phenotypes_pred": phenotypes_pred  # [B, N, NumPhenotypes]
        }

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

        if torch.jit.is_scripting(): # type: ignore
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections

        return self.eager_outputs(losses, detections)

    def postprocess_detections(
        self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)
        phenotypes_pred = head_outputs["phenotypes_pred"]

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
