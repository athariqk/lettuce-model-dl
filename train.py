#!/usr/bin/env python3
"""
train.py - experiment-friendly training script with:
 - validation-loss per-epoch and "best-by-val" checkpoint selection
 - per-epoch CSV logging
 - multi-seed orchestration and post-run statistical tests
 - latency/param measurement using the best-by-val checkpoint
"""

import datetime
import os
import pprint
import time
import json
import random
import shutil
import csv
from contextlib import redirect_stdout
from copy import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.ops._utils
from sklearn.model_selection import KFold

# optional stats
try:
    from scipy import stats as _scipy_stats
    SCIPY_AVAILABLE = True
except Exception:
    _scipy_stats = None
    SCIPY_AVAILABLE = False

# Local project imports from train_original.py
import custom_types
from coco_eval import CocoEvaluator
from coco_utils import get_coco, get_coco_kp, get_coco_online
from dataset import get_lettuce_data, get_lettuce_data_no_h
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from torchvision.transforms import InterpolationMode
from neural_networks.utils import get_model
from transforms import SimpleCopyPaste

import presets
import my_utils as utils

# ------------------------------------------------------------
# Helper utilities (from train.py)
# ------------------------------------------------------------

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Counts trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model: torch.nn.Module, sample_input: custom_types.DualTensor, device: torch.device,
                    warmup: int = 50, runs: int = 200) -> float:
    """Measure single-image latency (ms)."""
    model.eval()
    example = [sample_input.to(device)]
    try:
        traced = torch.jit.trace(model.eval(), example)
    except Exception:
        traced = model

    with torch.no_grad():
        for _ in range(warmup):
            _ = traced(example) # type: ignore
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = traced(example) # type: ignore
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    return (t1 - t0) / runs * 1000.0


def _set_bn_eval(module):
    """Sets BatchNorm layers to eval mode."""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        module.eval()


def compute_validation_loss(model, data_loader, device, print_freq=100):
    """
    Compute mean validation loss over data_loader for detection models.
    Sets model.train() to get losses, but BN.eval() to avoid changing stats.
    """
    device = torch.device(device)
    orig_mode = model.training

    model.train()
    model.apply(_set_bn_eval)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val-loss:"

    total_loss = 0.0
    total_items = 0
    loss_components_acc = {}

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            # Guard against empty loss dicts: compute tensor sum explicitly to avoid Python int fallback
            if loss_dict_reduced:
                # Stack scalar tensors then sum to ensure result is a torch.Tensor and has .item()
                loss_value = float(torch.stack([loss for loss in loss_dict_reduced.values()]).sum().item())
            else:
                loss_value = 0.0

            batch_size = len(images)
            total_loss += loss_value * batch_size
            total_items += batch_size

            for k, v in loss_dict_reduced.items():
                loss_components_acc.setdefault(k, 0.0)
                loss_components_acc[k] += float(v.item()) * batch_size

            metric_logger.update(loss=loss_value, **loss_dict_reduced)

    mean_loss = total_loss / total_items if total_items > 0 else float("nan")
    mean_components = {k: (v / total_items) for k, v in loss_components_acc.items()}

    # restore original training/eval mode
    if not orig_mode:
        model.eval()
    else:
        model.train()

    return mean_loss, mean_components


def _convert_metrics_to_serializable(item):
    """Recursively converts numpy/torch types to standard python types for JSON."""
    if isinstance(item, dict):
        return {k: _convert_metrics_to_serializable(v) for k, v in item.items()}
    if isinstance(item, list):
        return [_convert_metrics_to_serializable(v) for v in item]
    if isinstance(item, np.ndarray):
        return item.tolist()
    if hasattr(torch, 'Tensor') and isinstance(item, torch.Tensor):
        return item.tolist()
    # Check for numpy scalar types (using more modern checks)
    if isinstance(item, (np.float16, np.float32, np.float64)): # type: ignore
        return float(item)
    if isinstance(item, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): # type: ignore
        return int(item)
    # Fallback for other np.floating/np.integer types
    if isinstance(item, np.floating):
        return float(item)
    if isinstance(item, np.integer):
        return int(item)
    # Return item as-is if it's already a serializable type
    return item


def get_eval_metrics_dict(evaluator: CocoEvaluator) -> dict:
    """Extracts metrics from the evaluator for CSV logging."""
    eval_metrics_dict = {}
    if evaluator:
        if evaluator.coco_eval:
            for iou_type, coco_eval in evaluator.coco_eval.items():
                if coco_eval.stats is not None:
                    # Convert numpy array to list for JSON serialization
                    eval_metrics_dict[iou_type] = coco_eval.stats.tolist()
        if evaluator.phenotype_metrics_results:
            # Recursively convert numpy/torch types to standard python types
            eval_metrics_dict['phenotype'] = _convert_metrics_to_serializable(
                evaluator.phenotype_metrics_results
            )
    return eval_metrics_dict

# ------------------------------------------------------------
# Core data/model utilities (from train_original.py)
# ------------------------------------------------------------

def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def collate_targets_only(batch):
    """A simple collate function that only extracts the target."""
    return [item[1] for item in batch]


def get_dataset(is_train, args, no_transform: bool = False):
    image_set = "train" if is_train else "val"
    paths = {
        "coco": (args.data_path, get_coco, 91),
        "coco_kp": (args.data_path, get_coco_kp, 2),
        "coco_online": (args.data_path, get_coco_online, 91),
        "lettuce_rgbd": (args.data_path, get_lettuce_data, 2),
        "lettuce_rgbd_no_h": (args.data_path, get_lettuce_data_no_h, 2)
    }
    p, ds_fn, num_classes = paths[args.dataset]

    ds = ds_fn(p, image_set=image_set, transforms=None if no_transform else get_transform(is_train, args),
               use_v2=args.use_v2, phenotype_names=args.phenotype_names)
    return ds, num_classes


def get_transform(is_train, args):
    if args.data_augmentation == "lettuce_rgbd":
        return presets.DetectionPresetLettuceRGBD(
            is_train=is_train,
            no_aug=False,
            phenotype_means=args.phenotype_means,
            phenotype_stds=args.phenotype_stds,
            boxcox_lambdas=args.boxcox_lambdas,
            minimums=args.minimums,
            maximums=args.maximums,
            log_transform=args.log_transform,
        )
    elif args.data_augmentation == "lettuce_rgbd_noaug":
        return presets.DetectionPresetLettuceRGBD(
            is_train=is_train,
            no_aug=True,
            phenotype_means=args.phenotype_means,
            phenotype_stds=args.phenotype_stds,
            boxcox_lambdas=args.boxcox_lambdas,
            minimums=args.minimums,
            maximums=args.maximums,
            log_transform=args.log_transform,
        )
    
    if args.data_augmentation == "lettuce_rgbd_alb":
        return presets.DetectionPresetTrainAlbumentation(
            is_train=is_train,
            no_aug=False,
            phenotype_means=args.phenotype_means,
            phenotype_stds=args.phenotype_stds,
            boxcox_lambdas=args.boxcox_lambdas,
            minimums=args.minimums,
            maximums=args.maximums,
            log_transform=args.log_transform,
        )
    elif args.data_augmentation == "lettuce_rgbd_alb_noaug":
        return presets.DetectionPresetTrainAlbumentation(
            is_train=is_train,
            no_aug=True,
            phenotype_means=args.phenotype_means,
            phenotype_stds=args.phenotype_stds,
            boxcox_lambdas=args.boxcox_lambdas,
            minimums=args.minimums,
            maximums=args.maximums,
            log_transform=args.log_transform,
        )

    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


# ------------------------------------------------------------
# Argument parser (Merged)
# ------------------------------------------------------------

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Detection Training (Experiment-Friendly)", add_help=add_help)

    # --- Core args from train_original.py ---
    parser.add_argument("--data-path", default="data/coco", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="lettuce_model", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (cuda or cpu)")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=26, type=int, metavar="N")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer (sgd, adamw)")
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float)
    parser.add_argument("--lr-scheduler", default="multisteplr", type=str)
    parser.add_argument("--lr-step-size", default=8, type=int)
    parser.add_argument("--lr-steps", default=[16, 22], nargs="+", type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--print-freq", default=20, type=int)
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume-path", default="", type=str, help="Path to specific checkpoint file for resuming a single run.")
    parser.add_argument("--resume", action="store_true", help="Automatically resume multi-seed runs from the last checkpoint.")
    parser.add_argument("--resume-kfold", action="store_true", help="Resume K-Fold training.")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float)
    parser.add_argument("--trainable-backbone-layers", default=None, type=int)
    parser.add_argument("--data-augmentation", default="hflip", type=str)
    parser.add_argument("--sync-bn", dest="sync_bn", action="store_true")
    parser.add_argument("--test-only", dest="test_only", action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--weights-backbone", default=None, type=str)
    parser.add_argument("--saved-weights", default=None, type=str)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp")
    parser.add_argument("--use-copypaste", action="store_true")
    parser.add_argument("--backend", default="PIL", type=str.lower)
    parser.add_argument("--use-v2", action="store_true")
    parser.add_argument("--k-folds", type=int, default=0, help="Number of folds for K-Fold (0 or 1 to disable).")
    parser.add_argument("--val-split", type=float, default=None, help="Proportion of training set for validation.")

    # --- Phenotype args from train_original.py ---
    parser.add_argument("--phenotype-names", nargs="+", type=str)
    parser.add_argument("--phenotype-loss-weight", type=float)
    parser.add_argument("--phenotype-means", required=False, nargs="+", type=float)
    parser.add_argument("--phenotype-stds", required=False, nargs="+", type=float)
    parser.add_argument("--boxcox-lambdas", required=False, nargs="+", type=float)
    parser.add_argument("--minimums", required=False, nargs="+", type=float)
    parser.add_argument("--maximums", required=False, nargs="+", type=float)
    parser.add_argument("--skip-mean-calc", action="store_true")
    parser.add_argument("--skip-std-calc", action="store_true")
    parser.add_argument("--skip-min-calc", action="store_true")
    parser.add_argument("--skip-max-calc", action="store_true")
    parser.add_argument("--log-transform", action="store_true")
    parser.add_argument("--tuning", action="store_true")

    # --- Experiment-tracking args from train.py ---
    parser.add_argument("--seed", type=int, default=42, help="Single seed for reproducibility.")
    parser.add_argument("--measure-latency", action="store_true", help="Measure latency on best model.")
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument("--save-metrics", action="store_true", help="Save run_results.json summary.")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation loss computation.")

    return parser

# ------------------------------------------------------------
# Core logic functions (from train_original.py)
# ------------------------------------------------------------

def calculate_phenotype_stats(subset: torch.utils.data.Subset, phenotype_names: List[str], log_transform: bool,
                              num_workers: int, args) -> Tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates the mean, std, min, and max for phenotype targets in a dataset subset.
    """
    all_phenotypes = []

    data_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=32,
        num_workers=num_workers,
        collate_fn=collate_targets_only,
        persistent_workers=True if num_workers > 0 else False
    )

    for targets_batch in data_loader:
        for target in targets_batch:
            if "phenotypes" in target and target["phenotypes"].numel() > 0:
                phenotypes = target["phenotypes"]
                if log_transform:
                    phenotypes = torch.log1p(phenotypes)
                all_phenotypes.append(phenotypes)

    if not all_phenotypes:
        return None, None, None, None

    combined_phenotypes = torch.cat(all_phenotypes, dim=0)

    mean = torch.mean(combined_phenotypes, dim=0) if not args.skip_mean_calc else None
    std_dev = torch.std(combined_phenotypes, dim=0) if not args.skip_std_calc else None
    mins = torch.min(combined_phenotypes, dim=0).values if not args.skip_min_calc else None
    maxs = torch.max(combined_phenotypes, dim=0).values if not args.skip_max_calc else None

    return mean, std_dev, mins, maxs


def save_evaluator_summary(evaluator: CocoEvaluator, output_path: str):
    """Saves the summary from a CocoEvaluator to a file."""
    if not evaluator:
        if utils.is_main_process():
            print(f"Warning: Attempted to save summary, but evaluator is None.")
        return
    if not utils.is_main_process():
        return
    
    print(f"Saving evaluation summary to {output_path}")
    with open(output_path, "w") as f:
        with redirect_stdout(f):
            evaluator.summarize()


def k_fold_training(args, num_classes, full_dataset):
    """
    Modified K-Fold training loop.
    Integrates CSV logging and best-by-validation-loss checkpointing.
    """
    device = torch.device(args.device)

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42) # Use fixed random_state

    fold_results = [None] * args.k_folds
    fold_phenotype_metrics = [None] * args.k_folds
    resume_fold_idx = -1

    if args.resume_kfold:
        # Scan for the last completed or in-progress fold
        for i in range(args.k_folds, 0, -1):
            fold_dir = os.path.join(args.output_dir, f"fold_{i}")
            checkpoint_path = os.path.join(fold_dir, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                chkpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
                if chkpt["epoch"] == args.epochs - 1:
                    resume_fold_idx = i
                else:
                    resume_fold_idx = i - 1
                break
        if resume_fold_idx != -1:
            if utils.is_main_process():
                print(f"--- Resuming K-Fold training. Starting from Fold {resume_fold_idx + 1} ---")
        else:
            if utils.is_main_process():
                print("--- --resume-kfold specified, but no checkpoints found. Starting from scratch. ---")

    if utils.is_main_process():
        print(f"Starting {args.k_folds}-Fold Cross-Validation")
    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        if fold < resume_fold_idx:
            # Load results from this already-completed fold
            if utils.is_main_process():
                print(f"--- Skipping completed Fold {fold + 1} ---")
            fold_dir = os.path.join(args.output_dir, f"fold_{fold + 1}")
            results_path = os.path.join(fold_dir, "fold_results.npz")
            if os.path.exists(results_path):
                results_data = np.load(results_path, allow_pickle=True)
                if 'coco_stats' in results_data and results_data['coco_stats'].any():
                    fold_results[fold] = results_data['coco_stats']
                if 'pheno_stats' in results_data and results_data['pheno_stats'].any():
                    fold_phenotype_metrics[fold] = results_data['pheno_stats'].item()
                if utils.is_main_process():
                    print(f"Loaded past results for Fold {fold + 1}")
            else:
                if utils.is_main_process():
                    print(f"Warning: Could not find results file for skipped Fold {fold + 1} at {results_path}")
            continue

        if utils.is_main_process():
            print(f"Fold {fold + 1}/{args.k_folds}")

        current_fold_output_dir = os.path.join(args.output_dir, f"fold_{fold + 1}")
        if args.output_dir:
            utils.mkdir(current_fold_output_dir)

        train_subset = torch.utils.data.Subset(full_dataset, train_idx.tolist())
        test_subset = torch.utils.data.Subset(full_dataset, test_idx.tolist())

        # --- Phenotype stats calculation
        if not args.test_only:
            if utils.is_main_process():
                print("-" * 50)
                print(f"Handling phenotype statistics for Fold {fold + 1}:")
            args.phenotype_means, args.phenotype_stds, args.minimums, args.maximums = None, None, None, None
            phenotype_means, phenotype_stds, phenotype_mins, phenotype_maxs = calculate_phenotype_stats(
                train_subset, args.phenotype_names, args.log_transform, args.workers, args
            )
            args.phenotype_means = phenotype_means
            args.phenotype_stds = phenotype_stds
            args.minimums = phenotype_mins
            args.maximums = phenotype_maxs
            
            if utils.is_main_process():
                for i, name in enumerate(args.phenotype_names):
                    mean_str = f"Mean={args.phenotype_means[i]:.4f}" if args.phenotype_means is not None else "Mean=skipped"
                    std_str = f"Std={args.phenotype_stds[i]:.4f}" if args.phenotype_stds is not None else "Std=skipped"
                    min_str = f"Min={args.minimums[i]:.4f}" if args.minimums is not None else "Min=skipped"
                    max_str = f"Max={args.maximums[i]:.4f}" if args.maximums is not None else "Max=skipped"
                    print(f"    - {name}: {mean_str}, {std_str}, {min_str}, {max_str}")

        # --- Dataloader setup
        train_dataset_for_loader = custom_types.TransformedSubset(train_subset, get_transform(is_train=True, args=args))
        test_dataset_for_loader = custom_types.TransformedSubset(test_subset, get_transform(is_train=False, args=args))
        
        if args.distributed:
            train_sampler = torch.utils.data.DistributedSampler(train_dataset_for_loader)
            test_sampler = torch.utils.data.DistributedSampler(test_dataset_for_loader, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset_for_loader)
            test_sampler = torch.utils.data.SequentialSampler(test_dataset_for_loader)

        if args.aspect_ratio_group_factor >= 0:
            try:
                group_ids = create_aspect_ratio_groups(train_dataset_for_loader, k=args.aspect_ratio_group_factor)
                train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
            except Exception as e:
                if utils.is_main_process():
                    print(f"Warning: Could not create aspect ratio groups... Using standard BatchSampler.")
                train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

        train_collate_fn_fold = utils.collate_fn
        if args.use_copypaste:
            train_collate_fn_fold = copypaste_collate_fn
        
        data_loader_train = torch.utils.data.DataLoader(
            train_dataset_for_loader, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=train_collate_fn_fold
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset_for_loader, batch_size=1, sampler=test_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn
        )

        # --- Model setup
        if utils.is_main_process():
            print("Creating model")
        kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers, "weights": args.weights}
        if args.data_augmentation in ["multiscale", "lsj"]:
            kwargs["_skip_resize"] = True
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        kwargs["device"] = device
        kwargs["num_phenotypes"] = len(args.phenotype_names)
        if args.phenotype_loss_weight:
            kwargs["phenotype_loss_weight"] = args.phenotype_loss_weight
        if args.log_transform:
            kwargs["log_transform"] = args.log_transform
        if args.boxcox_lambdas:
            kwargs["boxcox_lambdas"] = args.boxcox_lambdas
        if args.minimums is not None:
            kwargs["minimums"] = args.minimums
        if args.maximums is not None:
            kwargs["maximums"] = args.maximums

        model = get_model(args.model, num_classes=num_classes, **kwargs)
        
        if args.saved_weights:
            if utils.is_main_process():
                print("Loading saved weights: {}".format(args.saved_weights))
            weights = torch.load(args.saved_weights, map_location="cpu", weights_only=False)["model"]
            model.load_state_dict(weights)

        if not args.test_only:
            if hasattr(model, "phenotype_means") and args.phenotype_means is not None:
                model.phenotype_means = args.phenotype_means.unsqueeze(0).type_as(model.phenotype_means) # type: ignore
            if hasattr(model, "phenotype_stds") and args.phenotype_stds is not None:
                model.phenotype_stds = args.phenotype_stds.unsqueeze(0).type_as(model.phenotype_means) # type: ignore

        model.to(device)
        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model.train()

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # --- Test-only logic
        if args.test_only:
            torch.backends.cudnn.deterministic = True
            evaluate(model, data_loader_test, device=device, phenotype_names=args.phenotype_names)
            continue

        # --- Optimizer & Scheduler setup
        if args.norm_weight_decay is None:
            parameters = [p for p in model.parameters() if p.requires_grad]
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(model)
            wd_groups = [args.norm_weight_decay, args.weight_decay]
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters, lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay, nesterov="nesterov" in opt_name,
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

        scaler = torch.amp.grad_scaler.GradScaler() if args.amp else None

        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "multisteplr":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        else:
            raise RuntimeError(f"Invalid lr scheduler '{args.lr_scheduler}'.")

        # --- Resume logic
        args.start_epoch = 0
        if args.resume_kfold and fold == resume_fold_idx:
            checkpoint_path = os.path.join(current_fold_output_dir, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
                model_without_ddp.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
                if scaler and "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                if utils.is_main_process():
                    print(f"--- Successfully resumed Fold {fold + 1} from Epoch {args.start_epoch} ---")

        # --- CSV Logging & Val-Loss setup ---
        epoch_csv_path = os.path.join(current_fold_output_dir, "epoch_log.csv")
        write_header = not os.path.exists(epoch_csv_path) or args.start_epoch == 0
        csv_file = open(epoch_csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if write_header and utils.is_main_process():
            csv_writer.writerow(["epoch", "train_time_s", "val_loss", "val_loss_components_json", "eval_metrics_json", "checkpoint_path", "best_by_val"])
        
        best_val_loss = float('inf')
        best_epoch = -1
        val_history = {}
        # --- End Setup ---

        if utils.is_main_process():
            print(f"Start training for Fold {fold + 1}")
        start_time = time.time()
        last_epoch_evaluator = None
        for epoch in range(args.start_epoch, args.epochs):
            start_time_epoch = time.time()
            model.train()
            if isinstance(train_sampler, torch.utils.data.DistributedSampler):
                train_sampler.set_epoch(epoch)
            
            # --- Train ---
            train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq, scaler)
            train_time_s = time.time() - start_time_epoch
            lr_scheduler.step()

            # --- Compute Validation Loss ---
            if not args.no_validate:
                val_loss, val_components = compute_validation_loss(model, data_loader_test, device, print_freq=args.print_freq)
            else:
                val_loss, val_components = float('nan'), {}
            val_history[epoch] = {"val_loss": val_loss, "components": val_components}

            is_best = False
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                is_best = True
                
            # --- Checkpoint Saving ---
            if current_fold_output_dir:
                # 1. Create the checkpoint dictionary
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch, # <-- This is the *current* epoch
                    "fold": fold + 1,
                    "val_loss": val_loss
                }
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                
                # 2. Save the LATEST checkpoint (for resuming)
                if utils.is_main_process():
                    checkpoint_path = os.path.join(current_fold_output_dir, "checkpoint.pth")
                    tmp_path = checkpoint_path + ".tmp"
                    torch.save(checkpoint, tmp_path)
                    os.rename(tmp_path, checkpoint_path) # Atomic move
                
                # 3. Save the BEST checkpoint (for measurement) if this epoch is the best
                if is_best:
                    if utils.is_main_process():
                        try:
                            shutil.copyfile(os.path.join(current_fold_output_dir, "checkpoint.pth"),
                                            os.path.join(current_fold_output_dir, "model_best_by_val.pth"))
                        except Exception as e:
                            print(f"Warning: copying best checkpoint failed: {e}")

            # --- Evaluate
            evaluator: CocoEvaluator = evaluate(model, data_loader_test, device=device,
                                                phenotype_names=args.phenotype_names)
            last_epoch_evaluator = evaluator

            if evaluator and evaluator.iou_types:
                first_iou_type = evaluator.iou_types[0]
                if first_iou_type in evaluator.coco_eval:
                    eval_obj = evaluator.coco_eval[first_iou_type]
                    if eval_obj.stats is not None and len(eval_obj.stats) > 0:
                        fold_results[fold] = eval_obj.stats
                if evaluator.phenotype_metrics_results:
                    fold_phenotype_metrics[fold] = evaluator.phenotype_metrics_results

            # --- CSV Logging ---
            eval_metrics_dict = get_eval_metrics_dict(evaluator)
            if utils.is_main_process():
                csv_writer.writerow([
                    epoch,
                    f"{train_time_s:.3f}",
                    f"{val_loss:.6f}" if not np.isnan(val_loss) else "nan",
                    json.dumps(val_components),
                    json.dumps(eval_metrics_dict) if eval_metrics_dict else "",
                    os.path.join(current_fold_output_dir, "checkpoint.pth"),
                    "1" if is_best else "0"
                ])
                csv_file.flush()

        # --- End Epoch Loop ---

        # --- Save val_history and close CSV ---
        if utils.is_main_process():
            csv_file.close()
            with open(os.path.join(current_fold_output_dir, "val_loss_history.json"), "w") as f:
                json.dump(val_history, f, indent=2)

        # --- Post-Fold Summary
        if last_epoch_evaluator and args.output_dir:
            summary_file_path = os.path.join(current_fold_output_dir, "evaluation_summary.txt")
            save_evaluator_summary(last_epoch_evaluator, summary_file_path)

            fold_summary_path = os.path.join(current_fold_output_dir, "fold_results.npz")
            coco_stats = fold_results[fold]
            pheno_stats = fold_phenotype_metrics[fold]

            if utils.is_main_process():
                np.savez(fold_summary_path, coco_stats=coco_stats, pheno_stats=pheno_stats)
                print(f"Saved fold numerical results to {fold_summary_path}")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if utils.is_main_process():
            print(f"Training time {total_time_str}")

    # --- End Fold Loop ---

    # --- K-Fold Aggregate Reporting ---
    if any(res is not None for res in fold_results):
        # --- COCO metric aggregation
        valid_fold_stats = [stats for stats in fold_results if stats is not None and len(stats) > 0]
        if valid_fold_stats:
            all_fold_stats_np = np.array(valid_fold_stats)
            mean_stats = np.mean(all_fold_stats_np, axis=0)
            std_stats = np.std(all_fold_stats_np, axis=0)
            if utils.is_main_process():
                print("Average K-Fold Performance Metrics (based on last epoch of each fold):")
            metric_names = [
                "Average Precision  (AP) @[ IoU=0.50:0.95 |area=    all| maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.50      |area=    all| maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.75      |area=    all| maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.50:0.95 |area=  small| maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.50:0.95 |area= medium| maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.50:0.95 |area=  large| maxDets=100 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 |area=    all| maxDets=  1 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 |area=    all| maxDets= 10 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 |area=    all| maxDets=100 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 |area=  small| maxDets=100 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 |area= medium| maxDets=100 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 |area=  large| maxDets=100 ]",
            ]
            if utils.is_main_process():
                for i, name in enumerate(metric_names):
                    if i < len(mean_stats):
                        print(f"  {name}: Mean = {mean_stats[i]:.4f}, Std = {std_stats[i]:.4f}")

            if args.output_dir and utils.is_main_process():
                results_file = os.path.join(args.output_dir, "kfold_summary_stats.txt")
                with open(results_file, "w") as f:
                    f.write(f"K-Fold Cross-Validation Summary ({args.k_folds} folds)\n")
                    f.write("Mean Performance Metrics (based on last epoch of each fold):\n")
                    for i, name in enumerate(metric_names):
                         if i < len(mean_stats):
                            f.write(f"  {name}: Mean = {mean_stats[i]:.4f}, Std = {std_stats[i]:.4f}\n")
                    np.savez(os.path.join(args.output_dir, "kfold_stats.npz"), mean_stats=mean_stats,
                             std_stats=std_stats, all_fold_stats=all_fold_stats_np)
                print(f"K-Fold summary saved to {results_file}")
        else:
            if utils.is_main_process():
                print("No valid stats collected from folds to average.")
    else:
        if utils.is_main_process():
            print("No results collected from K-Folds.")

    # --- Phenotype metric aggregation
    if any(res is not None for res in fold_phenotype_metrics):
        if utils.is_main_process():
            print("Average K-Fold Phenotype Regression Metrics (based on last epoch of each fold):")
        aggregated_pheno_results = {}
        phenotype_keys = args.phenotype_names
        metric_keys = ["r2", "rmse", "mape"]

        valid_pheno_metrics = [m for m in fold_phenotype_metrics if m is not None]

        for p_key in phenotype_keys:
            aggregated_pheno_results[p_key] = {}
            for m_key in metric_keys:
                values = [
                    fold_data[p_key][m_key]
                    for fold_data in valid_pheno_metrics
                    if fold_data and p_key in fold_data and m_key in fold_data[p_key] and not np.isnan(
                        fold_data[p_key][m_key])
                ]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    aggregated_pheno_results[p_key][f'{m_key}_mean'] = mean_val
                    aggregated_pheno_results[p_key][f'{m_key}_std'] = std_val
                    if utils.is_main_process():
                        if m_key == 'mape':
                            print(f"  {p_key:<15} {m_key:<10}: Mean = {mean_val * 100:.2f}%, Std = {std_val * 100:.2f}%")
                        else:
                            print(f"  {p_key:<15} {m_key:<10}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
                else:
                    aggregated_pheno_results[p_key][f'{m_key}_mean'] = np.nan
                    aggregated_pheno_results[p_key][f'{m_key}_std'] = np.nan
                    if utils.is_main_process():
                        print(f"  {p_key:<15} {m_key:<10}: Not enough valid data across folds.")
        
        if args.output_dir and utils.is_main_process():
            summary_path = os.path.join(args.output_dir, "kfold_summary_phenotype_stats.txt")
            with open(summary_path, "w") as f:
                f.write(f"K-Fold Phenotype Regression Summary ({args.k_folds} folds)\n")
                json.dump(aggregated_pheno_results, f, indent=2)

            # Convert Python objects (dicts/lists) to object-dtype numpy arrays so np.savez accepts them
            np.savez(
                os.path.join(args.output_dir, "kfold_phenotype_stats.npz"),
                aggregated_metrics=np.array(aggregated_pheno_results, dtype=object),
                all_fold_metrics=np.array(valid_pheno_metrics, dtype=object),
            )
    else:
        if utils.is_main_process():
            print("No Phenotype metrics collected from K-Folds.")
    
    # K-Fold training doesn't return a single "best model"
    return None


def standard_training_impl(args):
    """
    Standard training loop.
    Integrates CSV logging and best-by-validation-loss checkpointing.
    Returns a dict with results for the orchestration wrapper.
    """
    device = torch.device(args.device)

    # --- Data setup
    calculated_means, calculated_stds, calculated_mins, calculated_maxs = None, None, None, None
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers, "weights": args.weights}
    
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    if args.phenotype_loss_weight:
        kwargs["phenotype_loss_weight"] = args.phenotype_loss_weight
    if args.log_transform:
        kwargs["log_transform"] = args.log_transform
    if args.boxcox_lambdas:
        kwargs["boxcox_lambdas"] = args.boxcox_lambdas
    kwargs["num_phenotypes"] = len(args.phenotype_names)
    if args.minimums:
        kwargs["minimums"] = args.minimums
    if args.maximums:
        kwargs["maximums"] = args.maximums

    if args.val_split and 0 < args.val_split < 1:
        # Use a validation split from the training set
        if utils.is_main_process():
            print(f"Train-validation split enabled. Using {args.val_split:.0%} for validation.")
        full_dataset, num_classes = get_dataset(is_train=True, args=args, no_transform=True)

        dataset_size = len(full_dataset)
        val_size = int(args.val_split * dataset_size)
        train_size = dataset_size - val_size
        if utils.is_main_process():
            print(f"Splitting dataset: {train_size} training images, {val_size} validation images.")

        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        if not args.test_only:
            # Calculate stats ONLY on train_subset
            if not (args.skip_mean_calc and args.skip_std_calc and args.skip_min_calc and args.skip_max_calc):
                if utils.is_main_process():
                    print("-" * 50)
                    print("Calculating phenotype statistics for the training split...")
                calculated_means, calculated_stds, calculated_mins, calculated_maxs = calculate_phenotype_stats(
                    train_subset, args.phenotype_names, args.log_transform, args.workers, args
                )

            if args.phenotype_means is None: args.phenotype_means = calculated_means
            if args.phenotype_stds is None: args.phenotype_stds = calculated_stds
            if args.minimums is None: args.minimums = calculated_mins
            if args.maximums is None: args.maximums = calculated_maxs

            if args.minimums is not None: kwargs["minimums"] = args.minimums
            if args.maximums is not None: kwargs["maximums"] = args.maximums

            if utils.is_main_process():
                print("Phenotype statistics in use for this run:")
                for i, name in enumerate(args.phenotype_names):
                    mean_str = f"Mean={args.phenotype_means[i]:.4f}" if args.phenotype_means is not None else "Mean=unused"
                    std_str = f"Std={args.phenotype_stds[i]:.4f}" if args.phenotype_stds is not None else "Std=unused"
                    min_str = f"Min={args.minimums[i]:.4f}" if args.minimums is not None else "Min=unused"
                    max_str = f"Max={args.maximums[i]:.4f}" if args.maximums is not None else "Max=unused"
                    print(f"    - {name}: {mean_str}, {std_str}, {min_str}, {max_str}")

        # Apply DIFFERENT transforms to train and val
        dataset = custom_types.TransformedSubset(
            train_subset, get_transform(is_train=True, args=args)
        )
        dataset_test = custom_types.TransformedSubset(
            val_subset, get_transform(is_train=False, args=args)
        )
    else:
        # Standard logic: load separate train/val
        if utils.is_main_process():
            print("Loading separate train and validation datasets.")
        dataset, num_classes = get_dataset(is_train=True, args=args)
        dataset_test, _ = get_dataset(is_train=False, args=args)
        
        if args.phenotype_means:
            kwargs["phenotype_means"] = args.phenotype_means
        if args.phenotype_stds:
            kwargs["phenotype_stds"] = args.phenotype_stds

    # --- Model setup
    if utils.is_main_process():
        print("Creating model")
    model = get_model(args.model, num_classes=num_classes, **kwargs)
    
    if args.saved_weights:
        if utils.is_main_process():
            print("Loading saved weights: {}".format(args.saved_weights))
        weights = torch.load(args.saved_weights, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(weights)

    if not args.test_only:
        if hasattr(model, "phenotype_means") and args.phenotype_means is not None:
            model.phenotype_means = torch.as_tensor(args.phenotype_means).unsqueeze(0).type_as(model.phenotype_means) # type: ignore
        if hasattr(model, "phenotype_stds") and args.phenotype_stds is not None:
            model.phenotype_stds = torch.as_tensor(args.phenotype_stds).unsqueeze(0).type_as(model.phenotype_stds) # type: ignore

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.train()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # --- Optimizer & Scheduler setup
    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}.")

    scaler = torch.amp.grad_scaler.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(f"Invalid lr scheduler '{args.lr_scheduler}'.")

    # --- Dataloader setup
    if utils.is_main_process():
        print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(dataset)
        test_sampler = torch.utils.data.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=train_collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    # --- Resume logic
    if args.resume_path:
        if utils.is_main_process():
            print(f"--- Resuming from specific path: {args.resume_path} ---")
        checkpoint = torch.load(args.resume_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    # --- Test-only logic
    if args.test_only:
        torch.backends.cudnn.deterministic = True
        evaluate(model, data_loader_test, device=device, phenotype_names=args.phenotype_names)
        return {"best_checkpoint": None, "param_count": count_parameters(model_without_ddp)}

    # --- CSV Logging & Val-Loss setup ---
    epoch_csv_path = os.path.join(args.output_dir, "epoch_log.csv")
    write_header = not os.path.exists(epoch_csv_path) or args.start_epoch == 0
    csv_file = open(epoch_csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header and utils.is_main_process():
        csv_writer.writerow(["epoch", "train_time_s", "val_loss", "val_loss_components_json", "eval_metrics_json", "checkpoint_path", "best_by_val"])

    best_val_loss = float('inf')
    best_epoch = -1
    val_history = {}
    # --- End Setup ---

    if utils.is_main_process():
        print("Starting standard training (K-Fold is disabled or k_folds <= 1).")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_time_epoch = time.time()
        if isinstance(train_sampler, torch.utils.data.DistributedSampler):
            train_sampler.set_epoch(epoch)
        
        # --- Train ---
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        train_time_s = time.time() - start_time_epoch
        lr_scheduler.step()
        
        # --- Compute Validation Loss ---
        if not args.no_validate:
            val_loss, val_components = compute_validation_loss(model, data_loader_test, device, print_freq=args.print_freq)
        else:
            val_loss, val_components = float('nan'), {}
        val_history[epoch] = {"val_loss": val_loss, "components": val_components}

        is_best = False
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            is_best = True

        # --- Checkpoint Saving ---
        if args.output_dir:
            # 1. Create the checkpoint dictionary
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch, # <-- This is the *current* epoch
                "val_loss": val_loss
            }
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            
            # 2. Save the LATEST checkpoint (for resuming)
            if utils.is_main_process():
                checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
                tmp_path = checkpoint_path + ".tmp"
                torch.save(checkpoint, tmp_path)
                os.rename(tmp_path, checkpoint_path) # Atomic move
            
            # 3. Save the BEST checkpoint (for measurement) if this epoch is the best
            if is_best:
                if utils.is_main_process():
                    try:
                        shutil.copyfile(os.path.join(args.output_dir, "checkpoint.pth"),
                                        os.path.join(args.output_dir, "model_best_by_val.pth"))
                    except Exception as e:
                        print(f"Warning: copying best checkpoint failed: {e}")

        # --- Evaluate ---
        evaluator = evaluate(model, data_loader_test, device=device, phenotype_names=args.phenotype_names)
        
        # --- CSV Logging ---
        eval_metrics_dict = get_eval_metrics_dict(evaluator)
        if utils.is_main_process():
            csv_writer.writerow([
                epoch,
                f"{train_time_s:.3f}",
                f"{val_loss:.6f}" if not np.isnan(val_loss) else "nan",
                json.dumps(val_components),
                json.dumps(eval_metrics_dict) if eval_metrics_dict else "",
                os.path.join(args.output_dir, "checkpoint.pth"),
                "1" if is_best else "0"
            ])
            csv_file.flush()
    
    # --- End Epoch Loop ---

    # --- Save val_history and close CSV ---
    if utils.is_main_process():
        csv_file.close()
        with open(os.path.join(args.output_dir, "val_loss_history.json"), "w") as f:
            json.dump(val_history, f, indent=2)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.is_main_process():
        print(f"Training time {total_time_str}")

    best_ckpt_path = os.path.join(args.output_dir, "model_best_by_val.pth")
    return {
        "best_checkpoint": best_ckpt_path if os.path.exists(best_ckpt_path) else None,
        "best_val_loss": best_val_loss,
        "param_count": count_parameters(model_without_ddp),
        "val_history": val_history,
        "num_classes": num_classes # Pass this up for measurement step
    }


def args_sanity_check(args):
    """(from train_original.py)"""
    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp", "coco_online", "lettuce_rgbd", "lettuce_rgbd_no_h"):
        raise ValueError(f"Dataset should be coco, coco_kp, coco_online or coco-rgbd, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")
    if args.val_split and (args.val_split <= 0 or args.val_split >= 1):
        raise ValueError(f"--val-split must be between 0 and 1, but got {args.val_split}")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    if utils.is_main_process():
        print("--- Arguments ---")
        pprint.pprint(vars(args))
        print("-----------------")


def init_dist_args(args):
    utils.init_distributed_mode(args)
    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

def main():
    args = get_args_parser().parse_args()

    # --- 1. Initialize Distributed Mode FIRST ---
    # This will read env vars from torchrun (RANK, LOCAL_RANK, WORLD_SIZE)
    # and set args.distributed = True, args.gpu = LOCAL_RANK
    init_dist_args(args)

    # --- 2. Set Device based on distributed rank ---
    if args.distributed:
        # Each process gets its own GPU
        args.device = f"cuda:{args.gpu}"
        if utils.is_main_process():
            print(f"--- Running Distributed Training on {args.world_size} GPUs ---")
    else:
        if utils.is_main_process():
            print(f"--- Running Standard Training on {args.device} ---")
    device = torch.device(args.device)

    # --- 3. Set Seed ---
    # All processes must get the *same* seed.
    if utils.is_main_process():
        print(f"\n--- Setting Seed: {args.seed} ---")
    set_seed(args.seed) # This must be called on all processes

    # --- 4. Setup Output Dir ---
    # We assume output_dir is the *base* dir for this experiment.
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # --- 5. Snapshot args (only main process) ---
    snap = copy(vars(args))
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(snap, f, indent=2, default=str)

    # --- 6. Sanity Check & Load Data (only main process prints) ---
    if utils.is_main_process():
        args_sanity_check(args)
        print("Loading data")

    run_summary = {"seed": args.seed, "output_dir": args.output_dir}
    train_results = {}
    num_classes = 2 # Default, will be updated by dataset
    
    # --- 7. Handle Resume Logic ---
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    results_json_path = os.path.join(args.output_dir, "run_results.json")
    
    # K-Fold has its own --resume-kfold flag, so this logic is for standard runs.
    if args.resume and not args.k_folds > 1:
        if os.path.exists(checkpoint_path):
            if utils.is_main_process():
                print(f"Found checkpoint for seed {args.seed}: {checkpoint_path}")
            try:
                # All processes must load checkpoint to sync
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                last_epoch = checkpoint["epoch"]
                
                # Check if the run is already finished
                if last_epoch == args.epochs - 1:
                    if utils.is_main_process():
                        print(f"--- Skipping Seed {args.seed}: Training already complete. ---")
                    if os.path.exists(results_json_path) and utils.is_main_process():
                        print("Loading saved results.")
                        with open(results_json_path, "r") as f:
                            run_summary = json.load(f)
                    return # Exit the script
                else:
                    # Run is incomplete, set the resume_path to continue
                    if utils.is_main_process():
                        print(f"--- Resuming Seed {args.seed} from Epoch {last_epoch + 1} ---")
                    args.resume_path = checkpoint_path # Pass the path to standard_training_impl
            except Exception as e:
                if utils.is_main_process():
                    print(f"Warning: Could not load checkpoint {checkpoint_path} for seed {args.seed}. Retrying from scratch. Error: {e}")
        else:
            if utils.is_main_process():
                print(f"--- No checkpoint found for seed {args.seed}. Starting from scratch. ---")
    
    # --- 8. Run Training ---
    try:
        if args.k_folds > 1:
            if args.distributed and utils.is_main_process():
                print("Warning: K-Fold training with DDP is complex and may lead to issues.")
            dataset, num_classes = get_dataset(is_train=True, args=args, no_transform=True)
            k_fold_training(args, num_classes, dataset)
            # K-fold saves its own aggregate results, no single "best" model
            kfold_summary_file = os.path.join(args.output_dir, "kfold_summary_stats.txt")
            run_summary['kfold_summary_file'] = kfold_summary_file
        else:
            train_results = standard_training_impl(args)
            run_summary.update(train_results)
            num_classes = train_results.get("num_classes", num_classes)

    except Exception as e:
        if utils.is_main_process():
            print(f"!!! Training run for seed {args.seed} failed: {e} !!!")
        import traceback
        traceback.print_exc()
        run_summary["error"] = str(e)
        return # Exit
    # --- End core logic ---

    # --- 9. Post-Run Measurement ---
    if utils.is_main_process():
        print("--- Post-Run Measurement ---")
    
    best_ckpt_path = run_summary.get("best_checkpoint")
    
    # Re-build model kwargs to instantiate model for measurement
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers, "weights": args.weights}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    if args.phenotype_loss_weight:
        kwargs["phenotype_loss_weight"] = args.phenotype_loss_weight
    if args.log_transform:
        kwargs["log_transform"] = args.log_transform
    if args.boxcox_lambdas:
        kwargs["boxcox_lambdas"] = args.boxcox_lambdas
    kwargs["num_phenotypes"] = len(args.phenotype_names)
    if args.minimums:
        kwargs["minimums"] = args.minimums
    if args.maximums:
        kwargs["maximums"] = args.maximums
    kwargs["device"] = device # Add device for model instantiation

    try:
        model_for_measure = get_model(args.model, num_classes=num_classes, **kwargs)
        param_count = count_parameters(model_for_measure)
        run_summary['param_count'] = int(param_count)
        if utils.is_main_process():
            print(f"Model Parameters (trainable): {param_count}")

        if best_ckpt_path and os.path.exists(best_ckpt_path):
            if utils.is_main_process():
                print(f"Loading best checkpoint for measurement: {best_ckpt_path}")
            
            # Measurement only happens on main process, so no need for DDP model
            chk = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            model_for_measure.load_state_dict(chk["model"])
            if 'val_loss' in chk:
                run_summary['val_loss'] = float(chk['val_loss'])
        elif args.k_folds <= 1 and utils.is_main_process():
            print("Warning: No 'model_best_by_val.pth' found. Latency measurement will use initialized weights.")

        # Latency is a single-process (rank 0) measurement
        if args.measure_latency and args.k_folds <= 1 and utils.is_main_process():
            ds_test, _ = get_dataset(is_train=False, args=args, no_transform=True)
            eval_transform = get_transform(is_train=False, args=args)
            sample_img, _ = eval_transform(ds_test[0][0], ds_test[0][1])
            model_for_measure.to(device) # device is already rank-specific
            
            latency = measure_latency(model_for_measure, sample_img, device, warmup=args.latency_warmup, runs=args.latency_runs)
            run_summary['latency_ms'] = float(latency)
            print(f"Model Latency: {latency:.3f} ms")
        elif args.measure_latency and args.k_folds > 1 and utils.is_main_process():
            print("Skipping latency measurement for K-Fold run (no single best model).")

    except Exception as e:
        if utils.is_main_process():
            print(f"Post-run measurement failed: {e}")
        run_summary["measurement_error"] = str(e)

    # --- 10. Save Final Results (only main process) ---
    if args.save_metrics and utils.is_main_process():
        results_path = os.path.join(args.output_dir, "run_results.json")
        with open(results_path, "w") as f:
            json.dump(run_summary, f, indent=2, default=str)
        if utils.is_main_process():
            print(f"Run summary saved to {results_path}")
    
    if utils.is_main_process():
        print("All runs complete.")


if __name__ == "__main__":
    main()
