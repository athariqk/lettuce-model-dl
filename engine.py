import math
import sys
import time
from typing import Optional

import numpy as np
import torch
import torchvision
import my_utils as utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq,
    scaler: Optional[torch.amp.grad_scaler.GradScaler] = None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]
        with torch.amp.autocast_mode.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    all_pred_phenotypes = []
    all_gt_phenotypes = []

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)

        # Process phenotypes for R-squared and RMSE calculation
        for target_dict, output_dict in zip(targets, outputs):
            try:
                if "scores" not in output_dict or "phenotypes" not in output_dict or output_dict["scores"].numel() == 0:
                    continue

                max_score_idx = torch.argmax(output_dict["scores"])
                pred_phenotype_tensor = output_dict["phenotypes"][max_score_idx].squeeze()
                if pred_phenotype_tensor.ndim == 0 or pred_phenotype_tensor.numel() != 2:
                    continue
                pred_phenotype_np = pred_phenotype_tensor.cpu().numpy().reshape(2)

                gt_phenotypes_for_image_tensor = target_dict["phenotypes"].to(cpu_device)
                if gt_phenotypes_for_image_tensor.ndim != 2 or gt_phenotypes_for_image_tensor.shape[1] != 2:
                    continue
                if gt_phenotypes_for_image_tensor.shape[0] == 0:
                    continue

                for i in range(gt_phenotypes_for_image_tensor.shape[0]):
                    gt_single_instance_tensor = gt_phenotypes_for_image_tensor[i]
                    gt_single_instance_np = gt_single_instance_tensor.cpu().numpy().reshape(2)
                    all_gt_phenotypes.append(gt_single_instance_np)
                    all_pred_phenotypes.append(pred_phenotype_np)
            except KeyError as e:
                print(f"DEBUG: KeyError '{e}' processing image {target_dict.get('image_id')}. Skipping for phenotype.")
                continue
            except Exception as e:
                print(f"DEBUG: Error '{e}' processing image {target_dict.get('image_id')}. Skipping for phenotype.")
                continue

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Calculate R-squared and RMSE for phenotypes
    phenotype_names = ["fresh_weight", "height"]  #
    all_gt_phenotypes_np = np.array(all_gt_phenotypes)  #
    all_pred_phenotypes_np = np.array(all_pred_phenotypes)  #

    phenotype_metrics_results = {}

    print("\nPhenotype Regression Metrics (for this fold/evaluation run):")  # Clarified print statement
    if all_gt_phenotypes_np.size > 0 and all_pred_phenotypes_np.size > 0 and \
            all_gt_phenotypes_np.shape[0] == all_pred_phenotypes_np.shape[0] and \
            all_gt_phenotypes_np.shape[1] == len(phenotype_names):  # Basic sanity checks

        for i, name in enumerate(phenotype_names):
            gt_values = all_gt_phenotypes_np[:, i]  #
            pred_values = all_pred_phenotypes_np[:, i]  #

            current_pheno_metrics = {'r2': np.nan, 'rmse': np.nan, 'mape': np.nan}  # Default to NaN

            # Calculate R-squared and RMSE only if there's enough data and variance
            if len(gt_values) > 1 and np.std(gt_values) > 1e-6:  #
                try:
                    r2 = r2_score(gt_values, pred_values)  #
                    rmse = np.sqrt(mean_squared_error(gt_values, pred_values))  #
                    # Ensure no zero values in gt_values for MAPE if it's problematic, or handle division by zero
                    # For simplicity, sklearn's mean_absolute_percentage_error handles non-finite results with its own logic.
                    mape = mean_absolute_percentage_error(gt_values, pred_values)  #

                    current_pheno_metrics['r2'] = r2
                    current_pheno_metrics['rmse'] = rmse
                    current_pheno_metrics['mape'] = mape
                    print(f"  {name:<20} R-squared: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape * 100:.2f}%")  #
                except Exception as e:
                    print(f"  {name:<20} Error calculating metrics: {e}")
            else:
                print(f"  {name:<20} Not enough data or variance to calculate R-squared/RMSE/MAPE.")  #
            phenotype_metrics_results[name] = current_pheno_metrics
    else:
        print("  Not enough valid ground truth or prediction phenotype data to calculate metrics.")
        for name in phenotype_names:  # Ensure dict has keys even if no data
            phenotype_metrics_results[name] = {'r2': np.nan, 'rmse': np.nan, 'mape': np.nan}

    torch.set_num_threads(n_threads)

    return coco_evaluator, phenotype_metrics_results
