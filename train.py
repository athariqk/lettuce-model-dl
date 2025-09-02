import datetime
import os
import pprint
import time
from contextlib import redirect_stdout
from copy import copy
from typing import List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.ops._utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ray import tune

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


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


# A simple collate function that only extracts the target from each dataset item.
# This avoids trying to stack images and deals with variable-sized tensors in targets.
def collate_targets_only(batch):
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
               use_v2=args.use_v2)
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
        )

    if is_train:
        if "_alb" in args.data_augmentation:
            return presets.DetectionPresetTrainAlbumentation(
                data_augmentation=args.data_augmentation
            )
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="data/coco", type=str, help="dataset path")
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument("--model", default="lettuce_model", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--resume-kfold",
        action="store_true",
        help="Resume K-Fold training from the last saved checkpoint in the output directory.",
    )
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")
    parser.add_argument("--saved-weights", default=None, type=str, help="the saved weights file path to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    parser.add_argument("--k-folds", type=int, default=0,
                        help="Number of folds for K-Fold cross-validation. Set to 0 or 1 to disable K-Fold and use standard train/val split.")

    # New argument for train-validation split
    parser.add_argument("--val-split", type=float, default=None,
                        help="Proportion of the training set to use for validation (e.g., 0.2). If set, it overrides the default train/val split behavior.")

    parser.add_argument("--phenotype-names", nargs="+", type=str)
    parser.add_argument("--phenotype-loss-weight", type=float)
    parser.add_argument("--phenotype-means", required=False, nargs="+", type=float)
    parser.add_argument("--phenotype-stds", required=False, nargs="+", type=float)
    parser.add_argument("--boxcox-lambdas", required=False, nargs="+", type=float)
    parser.add_argument("--minimums", required=False, nargs="+", type=float)
    parser.add_argument("--maximums", required=False, nargs="+", type=float)

    parser.add_argument("--log-transform", action="store_true")

    parser.add_argument("--tuning", action="store_true")

    return parser


def calculate_phenotype_stats(subset: torch.utils.data.Subset, phenotype_names: List[str], log_transform: bool,
                              num_workers: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the mean, std, min, and max for phenotype targets in a dataset subset.
    Uses a DataLoader to speed up the process by parallelizing data loading.

    Args:
        subset (torch.utils.data.Subset): The data subset (train or test).
        phenotype_names (List[str]): List of phenotype names to analyze.
        log_transform (bool): If True, applies a log transformation (log1p) to phenotype values.
        num_workers (int): The number of worker processes to use for data loading.

    Returns:
        A tuple containing (mean, std_dev, mins, maxs) for each phenotype.
    """
    all_phenotypes = []

    # Create a DataLoader to fetch data in parallel.
    # We iterate over the original dataset (wrapped by the subset) to avoid loading augmented data.
    data_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=32,  # A larger batch size is fine as we are not using GPU memory here.
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
        num_phenotypes = len(phenotype_names)
        nan_tensor = torch.full((num_phenotypes,), float('nan'))
        return nan_tensor, nan_tensor, nan_tensor, nan_tensor

    combined_phenotypes = torch.cat(all_phenotypes, dim=0)
    mean = torch.mean(combined_phenotypes, dim=0)
    std_dev = torch.std(combined_phenotypes, dim=0)
    mins = torch.min(combined_phenotypes, dim=0).values
    maxs = torch.max(combined_phenotypes, dim=0).values

    return mean, std_dev, mins, maxs


def save_evaluator_summary(evaluator: CocoEvaluator, output_path: str):
    """Saves the summary from a CocoEvaluator to a file."""
    if not evaluator:
        print(f"Warning: Attempted to save summary, but evaluator is None.")
        return
    if not utils.is_main_process():
        return
    print(f"Saving evaluation summary to {output_path}")
    with open(output_path, "w") as f:
        with redirect_stdout(f):
            # The summarize method prints the results.
            evaluator.summarize()


def k_fold_training(args, num_classes, full_dataset):
    init_dist_args(args)

    device = torch.device(args.device)

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=10)

    fold_results = [None] * args.k_folds
    fold_phenotype_metrics = [None] * args.k_folds
    resume_fold_idx = -1

    if args.resume_kfold:
        # Check folders backwards to find the last one with a checkpoint
        for i in range(args.k_folds, 0, -1):
            fold_dir = os.path.join(args.output_dir, f"fold_{i}")
            checkpoint_path = os.path.join(fold_dir, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                chkpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
                # Check if the fold was completed
                if chkpt["epoch"] == args.epochs - 1:
                    # This fold finished, so we resume from the *next* fold
                    resume_fold_idx = i
                else:
                    # This fold was interrupted, so we resume *this* fold
                    resume_fold_idx = i - 1
                break
        if resume_fold_idx != -1:
            # If we are resuming from the next fold, the index is correct.
            # If we are resuming from an incomplete fold, the index is also correct.
            # E.g., if fold 3 (index 2) is incomplete, resume_fold_idx is 2. The loop will skip 0, 1.
            print(f"--- Resuming K-Fold training. Starting from Fold {resume_fold_idx + 1} ---")
        else:
            print("--- --resume-kfold specified, but no checkpoints found. Starting from scratch. ---")

    print(f"Starting {args.k_folds}-Fold Cross-Validation")
    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        # Logic to skip completed folds.
        if fold < resume_fold_idx:
            print(f"--- Skipping completed Fold {fold + 1} ---")
            # Load the saved numerical results for this skipped fold to ensure correct final averaging
            fold_dir = os.path.join(args.output_dir, f"fold_{fold + 1}")
            results_path = os.path.join(fold_dir, "fold_results.npz")
            if os.path.exists(results_path):
                results_data = np.load(results_path, allow_pickle=True)
                if 'coco_stats' in results_data and results_data['coco_stats'].any():
                    fold_results[fold] = results_data['coco_stats']
                if 'pheno_stats' in results_data and results_data['pheno_stats'].any():
                    fold_phenotype_metrics[fold] = results_data['pheno_stats'].item()
                print(f"Loaded past results for Fold {fold + 1}")
            else:
                print(f"Warning: Could not find results file for skipped Fold {fold + 1} at {results_path}")
            continue

        print(f"Fold {fold + 1}/{args.k_folds}")

        current_fold_output_dir = os.path.join(args.output_dir, f"fold_{fold + 1}")
        if args.output_dir:
            utils.mkdir(current_fold_output_dir)

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)

        # Kalkulasi untuk subset Latih (Train)
        if not args.test_only:
            print("-" * 50)
            print(f"Calculating phenotype statistics for Fold {fold + 1}:")

            phenotype_means, phenotype_stds, phenotype_mins, phenotype_maxs = calculate_phenotype_stats(train_subset, args.phenotype_names,
                                                                        args.log_transform, args.workers)
            for i, name in enumerate(args.phenotype_names):
                # Cek jika kalkulasi valid (bukan NaN)
                if not torch.isnan(phenotype_means[i]):
                    print(f"    - {name}: Mean = {phenotype_means[i]:.4f}, Std Dev = {phenotype_stds[i]:.4f}, Min = {phenotype_mins[i]:.4f}, Max = {phenotype_maxs[i]:.4f}")
                else:
                    print(f"    - {name}: No phenotype data found.")

            args.phenotype_means = phenotype_means
            args.phenotype_stds = phenotype_stds
            args.minimums = phenotype_mins
            args.maximums = phenotype_maxs

        train_dataset_for_loader = custom_types.TransformedSubset(train_subset, get_transform(is_train=True, args=args))
        test_dataset_for_loader = custom_types.TransformedSubset(test_subset, get_transform(is_train=False, args=args))

        if args.distributed:
            # Distributed samplers need to be aware of the subset
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
                print(
                    f"Warning: Could not create aspect ratio groups for fold {fold + 1} (Error: {e}). Using standard BatchSampler.")
                train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size,
                                                                    drop_last=True)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size,
                                                                drop_last=True)

        train_collate_fn_fold = utils.collate_fn
        if args.use_copypaste:
            if args.data_augmentation != "lsj":
                raise RuntimeError("SimpleCopyPaste only supports 'lsj' data augmentation.")
            train_collate_fn_fold = copypaste_collate_fn

        data_loader_train = torch.utils.data.DataLoader(
            train_dataset_for_loader, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=train_collate_fn_fold
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset_for_loader, batch_size=1, sampler=test_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn
            # Standard collate for eval
        )

        print(f"Fold {fold + 1}: Train size: {len(train_dataset_for_loader)}, Val size: {len(test_dataset_for_loader)}")

        print("Creating model")
        kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers, "weights": args.weights}
        if args.data_augmentation in ["multiscale", "lsj"]:
            kwargs["_skip_resize"] = True
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        kwargs["device"] = device
        # do the same for standard_training
        if args.phenotype_loss_weight:
            kwargs["phenotype_loss_weight"] = args.phenotype_loss_weight
        # if args.phenotype_means:
        #     kwargs["phenotype_means"] = args.phenotype_means
        # if args.phenotype_stds:
        #     kwargs["phenotype_stds"] = args.phenotype_stds
        if args.log_transform:
            kwargs["log_transform"] = args.log_transform
        if args.boxcox_lambdas:
            kwargs["boxcox_lambdas"] = args.boxcox_lambdas
        if args.minimums:
            kwargs["minimums"] = args.minimums
        if args.maximums:
            kwargs["maximums"] = args.maximums

        model = get_model(args.model, num_classes=num_classes, **kwargs)

        if args.saved_weights:
            print("Loading saved weights: {}".format(args.saved_weights))
            weights = torch.load(args.saved_weights, map_location="cpu", weights_only=False)["model"]
            model.load_state_dict(weights)

        if not args.test_only:
            if hasattr(model, "phenotype_means"):
                model.phenotype_means = phenotype_means.unsqueeze(0).type_as(model.phenotype_means)
            if hasattr(model, "phenotype_stds"):
                model.phenotype_stds = phenotype_stds.unsqueeze(0).type_as(model.phenotype_means)

        model.to(device)
        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model.train()

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        if args.test_only:
            torch.backends.cudnn.deterministic = True
            evaluate(model, data_loader_test, device=device, phenotype_names=args.phenotype_names)
            continue

        if args.norm_weight_decay is None:
            parameters = [p for p in model.parameters() if p.requires_grad]
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(model)
            wd_groups = [args.norm_weight_decay, args.weight_decay]
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

        scaler = torch.amp.grad_scaler.GradScaler() if args.amp else None

        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "multisteplr":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps,
                                                                gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
            )

        args.start_epoch = 0  # Reset start_epoch for each new fold
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
                print(f"--- Successfully resumed Fold {fold + 1} from Epoch {args.start_epoch} ---")

        print(f"Start training for Fold {fold + 1}")
        start_time = time.time()
        last_epoch_evaluator = None
        for epoch in range(args.start_epoch, args.epochs):
            model.train()
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq, scaler)
            lr_scheduler.step()
            if current_fold_output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "fold": fold + 1
                }
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                utils.save_on_master(checkpoint, os.path.join(current_fold_output_dir, f"model_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(current_fold_output_dir, "checkpoint.pth"))

            # evaluate after every epoch
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

        # Save both text summary and numerical results at the end of each fold.
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
        print(f"Training time {total_time_str}")

    if any(res is not None for res in fold_results):
        valid_fold_stats = [stats for stats in fold_results if stats is not None and len(stats) > 0]
        if valid_fold_stats:
            all_fold_stats_np = np.array(valid_fold_stats)
            mean_stats = np.mean(all_fold_stats_np, axis=0)
            std_stats = np.std(all_fold_stats_np, axis=0)
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
            for i, name in enumerate(metric_names):
                if i < len(mean_stats):
                    print(f"  {name}: Mean = {mean_stats[i]:.4f}, Std = {std_stats[i]:.4f}")

            if args.output_dir:
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
            print("No valid stats collected from folds to average.")
    else:
        print("No results collected from K-Folds.")

    if any(res is not None for res in fold_phenotype_metrics):
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
                    if m_key == 'mape':
                        print(f"  {p_key:<15} {m_key:<10}: Mean = {mean_val * 100:.2f}%, Std = {std_val * 100:.2f}%")
                    else:
                        print(f"  {p_key:<15} {m_key:<10}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
                else:
                    aggregated_pheno_results[p_key][f'{m_key}_mean'] = np.nan
                    aggregated_pheno_results[p_key][f'{m_key}_std'] = np.nan
                    print(f"  {p_key:<15} {m_key:<10}: Not enough valid data across folds.")

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "kfold_summary_phenotype_stats.txt"), "w") as f:
                f.write(
                    f"K-Fold Phenotype Regression Summary ({args.k_folds} folds\nMean Performance (last epoch of each fold):\n")
                for p_key in phenotype_keys:
                    f.write(f" Phenotype: {p_key}\n")
                    for m_key in metric_keys:
                        mean_val = aggregated_pheno_results[p_key].get(f'{m_key}_mean', np.nan)
                        std_val = aggregated_pheno_results[p_key].get(f'{m_key}_std', np.nan)
                        if not np.isnan(mean_val):
                            if m_key == 'mape':
                                f.write(f"    {m_key:<8}: Mean = {mean_val * 100:.2f}%, Std = {std_val * 100:.2f}%\n")
                            else:
                                f.write(f"    {m_key:<8}: Mean = {mean_val:.4f}, Std = {std_val:.4f}\n")
                        else:
                            f.write(f"    {m_key:<8}: Not enough valid data across folds.\n")
            np.savez(os.path.join(args.output_dir, "kfold_phenotype_stats.npz"),
                     aggregated_metrics=aggregated_pheno_results,
                     all_fold_metrics=valid_pheno_metrics)
    else:
        print("No Phenotype metrics collected from K-Folds.")


def standard_training(args):
    config = {
        "lr": tune.grid_search([0.00009, 0.001]),
        "phenotype_loss_weight": tune.grid_search([0.1, 0.9]),
    }

    if args.tuning:
        tuner = tune.with_parameters(standard_training_impl, args=args)
        tune.run(tuner, config=config, num_samples=10)
    else:
        standard_training_impl(config, args)


# gak bisa dipake dgn ray tune
def standard_training_impl(config, args):
    init_dist_args(args)

    device = torch.device(args.device)

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
    if args.minimums:
        kwargs["minimums"] = args.minimums
    if args.maximums:
        kwargs["maximums"] = args.maximums

    # --- START: MODIFICATION FOR TRAIN-VAL SPLIT ---

    # Check if a validation split is requested from the command line arguments
    if args.val_split and 0 < args.val_split < 1:
        print(f"Train-validation split enabled. Using {args.val_split:.0%} of the data for validation.")

        # 1. Load the full training dataset without applying transforms initially.
        full_dataset, num_classes = get_dataset(is_train=True, args=args, no_transform=True)

        # 2. Calculate the size of each split.
        dataset_size = len(full_dataset)
        val_size = int(args.val_split * dataset_size)
        train_size = dataset_size - val_size
        print(f"Splitting dataset: {train_size} training images, {val_size} validation images.")

        # 3. Perform the random split using a fixed seed for reproducibility.
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(96)
        )

        # 4. Calculate phenotype statistics on the training subset before applying augmentations.
        if not args.test_only:
            # If stats arguments are not provided, calculate them from the training subset.
            if args.phenotype_means is None or args.phenotype_stds is None or args.minimums is None or args.maximums is None:
                print("-" * 50)
                print("Calculating phenotype statistics for the training split (means, stds, mins, maxs)...")
                phenotype_means, phenotype_stds, phenotype_mins, phenotype_maxs = calculate_phenotype_stats(
                    train_subset, args.phenotype_names, args.log_transform, args.workers
                )
                # Assign calculated values to args
                args.phenotype_means = phenotype_means
                args.phenotype_stds = phenotype_stds
                args.minimums = phenotype_mins
                args.maximums = phenotype_maxs
            else:
                print("-" * 50)
                print("Using user-provided phenotype statistics (means, stds, mins, maxs).")
                # Ensure provided stats are tensors for consistency
                args.phenotype_means = torch.as_tensor(args.phenotype_means, dtype=torch.float32)
                args.phenotype_stds = torch.as_tensor(args.phenotype_stds, dtype=torch.float32)
                args.minimums = torch.as_tensor(args.minimums, dtype=torch.float32)
                args.maximums = torch.as_tensor(args.maximums, dtype=torch.float32)

            # Print the final statistics being used for the training run
            print("Phenotype statistics in use for this run:")
            for i, name in enumerate(args.phenotype_names):
                # Check if calculation was valid (not NaN)
                if not torch.isnan(args.phenotype_means[i]):
                    print(f"    - {name}: Mean={args.phenotype_means[i]:.4f}, Std={args.phenotype_stds[i]:.4f}, Min={args.minimums[i]:.4f}, Max={args.maximums[i]:.4f}")
                else:
                    print(f"    - {name}: No phenotype data found.")


        # 5. Apply the correct transformations to each subset on-the-fly.
        dataset = custom_types.TransformedSubset(
            train_subset, get_transform(is_train=True, args=args)
        )
        dataset_test = custom_types.TransformedSubset(
            val_subset, get_transform(is_train=False, args=args)
        )
    else:
        # Original behavior: Load train and validation sets from separate sources.
        print("Loading separate train and validation datasets.")
        dataset, num_classes = get_dataset(is_train=True, args=args)
        dataset_test, _ = get_dataset(is_train=False, args=args)

        if args.phenotype_means:
            kwargs["phenotype_means"] = args.phenotype_means
            args.phenotype_means = torch.Tensor(args.phenotype_means).unsqueeze(0)
        if args.phenotype_stds:
            kwargs["phenotype_stds"] = args.phenotype_stds
            args.phenotype_stds = torch.Tensor(args.phenotype_stds).unsqueeze(0)

    # --- END: MODIFICATION FOR TRAIN-VAL SPLIT ---

    print("Creating model")

    model = get_model(args.model, num_classes=num_classes, **kwargs)

    if args.saved_weights:
        print("Loading saved weights: {}".format(args.saved_weights))
        weights = torch.load(args.saved_weights, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(weights)

    if not args.test_only:
        # Ensure args.phenotype_means/stds exist and are tensors before assigning to model
        if hasattr(model, "phenotype_means") and args.phenotype_means is not None:
            model.phenotype_means = torch.as_tensor(args.phenotype_means).unsqueeze(0).type_as(model.phenotype_means)
        if hasattr(model, "phenotype_stds") and args.phenotype_stds is not None:
            model.phenotype_stds = torch.as_tensor(args.phenotype_stds).unsqueeze(0).type_as(model.phenotype_stds)


    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.train()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
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
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

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
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    # # Kalkulasi untuk subset Latih (Train)
    # if not args.test_only:
    #     print(f"Calculating phenotype statistics")
    #
    #     phenotype_means, phenotype_stds = calculate_phenotype_stats(dataset, args.phenotype_names,
    #                                                                 args.log_transform)
    #     for i, name in enumerate(args.phenotype_names):
    #         # Cek jika kalkulasi valid (bukan NaN)
    #         if not torch.isnan(phenotype_means[i]):
    #             print(f"    - {name}: Mean = {phenotype_means[i]:.4f}, Std Dev = {phenotype_stds[i]:.4f}")
    #         else:
    #             print(f"    - {name}: No phenotype data found.")
    #
    #     args.phenotype_means = phenotype_means
    #     args.phenotype_stds = phenotype_stds
    #
    # train_dataset_for_loader = custom_types.TransformedSet(train_sampler, get_transform(is_train=True, args=args))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=train_collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_collate_fn)

    # if not args.test_only:
    #     if hasattr(model, "phenotype_means"):
    #         model.phenotype_means = phenotype_means.unsqueeze(0).type_as(model.phenotype_means)
    #     if hasattr(model, "phenotype_stds"):
    #         model.phenotype_stds = phenotype_stds.unsqueeze(0).type_as(model.phenotype_means)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.deterministic = True
        evaluate(model, data_loader_test, device=device, phenotype_names=args.phenotype_names)
        return

    print("Starting standard training (K-Fold is disabled or k_folds <= 1).")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device, phenotype_names=args.phenotype_names)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def args_sanity_check(args):
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

    print(args)


def init_dist_args(args):
    utils.init_distributed_mode(args)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


def main(args):
    args_sanity_check(args)

    # Data loading code
    print("Loading data")

    if args.k_folds > 1:
        # is_train is ignored
        dataset, num_classes = get_dataset(is_train=True, args=args, no_transform=True)
        k_fold_training(args, num_classes, dataset)
    else:
        standard_training(args)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)