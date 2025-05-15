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
from typing import Dict, List, Optional, Tuple
from cvnets.models.detection.ssd import SingleShotMaskDetector

from my_utils import ROOT_DIR


class Modified_SSDLiteMobileViT(nn.Module):
    """A modified SSDLite-MobileViT pretrained on COCO2017 for estimating lettuce growth phenotypes"""
    
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
        **kwargs
    ):
        super().__init__()

        self.model: SingleShotMaskDetector = torch.load(
            os.path.join(ROOT_DIR, "models/coco-ssd-mobilevitv2-0.75_2class_pretrained.pt"),
            weights_only=False
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(
            min(size), max(size), image_mean, image_std, size_divisible=1, fixed_size=size, **kwargs
        )

        self.anchor_generator = DefaultBoxGenerator(aspect_ratios, min_ratio=0.1, max_ratio=1.05)
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        # Anchor box related parameters
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        
        self.neg_to_pos_ratio = 3
        self.label_smoothing = 0.3

    @torch.jit.unused  
    def eager_outputs(
        self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        if self.training:
            return losses

        return detections

    def hard_negative_mining(
        self, loss: Tensor, labels: Tensor
    ) -> Tensor:
        pos_mask = labels > 0
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        num_neg = num_pos * self.neg_to_pos_ratio

        loss[pos_mask] = -math.inf
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_mask = orders < num_neg
        return pos_mask | neg_mask

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
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]
        phenotypes = head_outputs["phenotypes_pred"]

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            anchors_per_image,
        ) in zip(targets, bbox_regression, anchors):
            # produce the matching between boxes and targets
            gt_coordinates, gt_labels = self.model.match_prior(
                gt_boxes=targets_per_image["boxes"],
                gt_labels=targets_per_image["labels"],
                anchors=anchors_per_image,
            )

            num_coordinates = bbox_regression_per_image.shape[-1]

            # Calculate regression loss
            pos_mask = gt_labels > 0
            predicted_locations = bbox_regression_per_image[pos_mask].reshape(-1, num_coordinates)
            gt_coordinates = gt_coordinates[pos_mask].reshape(-1, num_coordinates)
            num_foreground += gt_coordinates.shape[0]
            bbox_loss.append(F.smooth_l1_loss(predicted_locations, gt_coordinates, reduction="sum"))
            
            cls_targets.append(gt_labels)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        num_classes = cls_logits.shape[-1]

        with torch.no_grad():
            loss = -F.log_softmax(cls_logits, dim=2)[:, :, 0]
            mask = self.hard_negative_mining(loss, cls_targets)

        cls_logits_masked = cls_logits[mask, :]
        cls_targets_masked = cls_targets[mask]

        label_smoothing = self.label_smoothing if self.training else 0.0

        classification_loss = F.cross_entropy(
            input=cls_logits_masked.reshape(-1, num_classes),
            target=cls_targets_masked,
            reduction="sum",
            label_smoothing=label_smoothing,
        )

        N_reg = max(1, num_foreground)

        num_masked_samples = cls_targets_masked.numel()
        N_cls = max(1, num_masked_samples)

        return {
            "reg_loss": bbox_loss / N_reg,
            "cls_loss": classification_loss / N_cls,
        }

        # --- (Optional) Phenotype Loss ---
        if phenotypes_pred is not None and "phenotypes" in targets[0] and self.num_phenotypes:
            pheno_loss_sum = torch.tensor(0.0, device=cls_logits.device)
            for i in range(batch_size):
                targets_per_image = targets[i]
                phenotypes_pred_per_image = phenotypes_pred[i]
                matched_idxs_per_image = matched_idxs[i]
                
                foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
                foreground_matched_gt_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]

                if foreground_matched_gt_idxs_per_image.numel() > 0:
                    # Assuming target phenotypes are [NumGTBoxes, NumPhenotypes]
                    # And they are continuous variables (using MSE loss as an example)
                    matched_gt_phenotypes = targets_per_image["phenotypes"][foreground_matched_gt_idxs_per_image]
                    pred_pheno_for_gt = phenotypes_pred_per_image[foreground_idxs_per_image, :]
                    pheno_loss_sum += F.mse_loss(
                        pred_pheno_for_gt, matched_gt_phenotypes, reduction="sum"
                    )
            losses["phenotype_regression"] = pheno_loss_sum / N
            
        return losses

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]] | Dict[str, Tensor] | List[Dict[str, Tensor]]:
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

        # transform the input
        images_transformed, targets_transformed = self.transform(images, targets)

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

        features = self.model.get_backbone_features(images_transformed.tensors)

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

        for boxes, scores, phenotypes, anchors, image_shape in zip(bbox_regression, pred_scores, phenotypes_pred, image_anchors, image_shapes):
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
