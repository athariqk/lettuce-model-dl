import math
import sys
import time
from typing import List, Dict, Tuple, Optional

import torch
import torchvision
from torch import Tensor
from torchvision.models.detection._utils import SSDMatcher, BoxCoder
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F

import my_utils as utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from custom_types import DualTensor

from neural_networks.custom_types import LettuceDetectionOutputs
from neural_networks.model import ModifiedSSDLiteMobileViTBase
from neural_networks.utils import postprocess_lettuce_detections


class BaseTrainer:
    def train_one_epoch(self, device, print_freq):
        raise NotImplementedError

    @torch.inference_mode()
    def evaluate(self, device):
        raise NotImplementedError


class LettuceDetectorTrainer(BaseTrainer):
    def __init__(
            self,
            model,
            optimizer,
            data_loader,
            epoch,
            scaler,
            phenotype_names,
            size: Tuple[int, int],
            iou_thresh: float = 0.5,
            phenotype_loss_weight: float = 0.0001,
            image_mean: Optional[List[float]] = None,
            image_std: Optional[List[float]] = None,
            **kwargs
    ):
        if not isinstance(model, ModifiedSSDLiteMobileViTBase):
            raise ValueError("Unsupported model type")

        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.epoch = epoch
        self.scaler = scaler
        self.phenotype_names = phenotype_names

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(
            min(size), max(size), image_mean, image_std, size_divisible=1, fixed_size=size, **kwargs
        )

        self.proposal_matcher = SSDMatcher(iou_thresh)
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        self.neg_to_pos_ratio = 3
        self.label_smoothing = 0.3

        self.phenotype_loss_weight = phenotype_loss_weight

    def compute_loss(
            self,
            head_outputs: LettuceDetectionOutputs,
            targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        bbox_regression = head_outputs.bbox_regression
        cls_logits = head_outputs.cls_logits
        phenotypes_pred = head_outputs.phenotypes_pred
        anchors = head_outputs.anchors

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

    def train_one_epoch(self, device, print_freq):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{self.epoch}]"

        lr_scheduler = None
        if self.epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(self.data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        for images, targets in metric_logger.log_every(self.data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            image_tensors: List[Tensor] = [item.x if isinstance(item, DualTensor) else item for item in images]
            aux_tensors: List[Tensor] = [item.y if isinstance(item, DualTensor) else item for item in images]

            images, targets = self.transform(image_tensors, targets)
            aux, _ = self.transform(aux_tensors)
            stacked = torch.stack([images.tensors, aux.tensors], dim=0)

            with torch.amp.autocast_mode.autocast("cuda", enabled=self.scaler is not None):
                outputs = self.model(stacked)
                loss_dict = self.compute_loss(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses.backward()
                self.optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        return metric_logger


    def _get_iou_types(self):
        model_without_ddp = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types


    @torch.inference_mode()
    def evaluate(self, device):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        coco = get_coco_api_from_dataset(self.data_loader.dataset)
        iou_types = self._get_iou_types()
        coco_evaluator = CocoEvaluator(coco, iou_types, self.phenotype_names, 0.5)

        for images, targets in metric_logger.log_every(self.data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
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
            outputs = self.model(stacked)
            outputs = postprocess_lettuce_detections(
                outputs,
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

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            predictions_for_coco = {}
            targets_for_pheno_eval = {}

            for target_dict, output_dict in zip(targets, outputs):
                img_id: int = target_dict["image_id"]
                predictions_for_coco[img_id] = output_dict
                targets_for_pheno_eval[img_id] = {
                    "boxes": target_dict["boxes"].to(cpu_device),
                    "phenotypes": target_dict["phenotypes"].to(cpu_device)
                }

            evaluator_time = time.time()
            coco_evaluator.update(predictions_for_coco, targets_for_pheno_eval)

            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        torch.set_num_threads(n_threads)

        return coco_evaluator
