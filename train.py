import datetime
import os
import time

import numpy as np
import torch
import torchvision
import torchvision.ops._utils
from sklearn.model_selection import KFold

from coco_utils import get_coco, get_coco_kp, get_coco_online
from dataset import get_rgbd_data
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


def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    paths = {
        "coco": (args.data_path, get_coco, 91),
        "coco_kp": (args.data_path, get_coco_kp, 2),
        "coco_online": (args.data_path, get_coco_online, 91),
        "coco_rgbd": (args.data_path, get_rgbd_data, 2)
    }
    p, ds_fn, num_classes = paths[args.dataset]

    ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train, args), use_v2=args.use_v2)
    return ds, num_classes


def get_transform(is_train, args):
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


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="data\coco", type=str, help="dataset path")
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

    return parser


def k_fold_training(args, num_classes, full_dataset, device):
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold_results = []
    fold_phenotype_metrics = []

    print(f"Starting {args.k_folds}-Fold Cross-Validation")
    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        print(f"Fold {fold + 1}/{args.k_folds}")

        current_fold_output_dir = os.path.join(args.output_dir, f"fold_{fold + 1}")
        if args.output_dir:
            utils.mkdir(current_fold_output_dir)

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)

        if args.distributed:
            # Distributed samplers need to be aware of the subset
            train_sampler = torch.utils.data.DistributedSampler(train_subset)
            test_sampler = torch.utils.data.DistributedSampler(test_subset, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_subset)
            test_sampler = torch.utils.data.SequentialSampler(test_subset)

        if args.aspect_ratio_group_factor >= 0:
            try:
                group_ids = create_aspect_ratio_groups(train_subset, k=args.aspect_ratio_group_factor)
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
            train_subset, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=train_collate_fn_fold
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_subset, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
            # Standard collate for eval
        )

        print(f"Fold {fold + 1}: Train size: {len(train_subset)}, Val size: {len(test_subset)}")

        print("Creating model")
        kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers, "weights": args.weights}
        if args.data_augmentation in ["multiscale", "lsj"]:
            kwargs["_skip_resize"] = True
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh

        model = get_model(args.model, num_classes=num_classes, **kwargs)

        if args.saved_weights:
            print("Loading saved weights: {}".format(args.saved_weights))
            weights = torch.load(args.saved_weights, weights_only=False)["model"]
            model.load_state_dict(weights)

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
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps,
                                                                gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
            )

        print(f"Start training for Fold {fold + 1}")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
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
            evaluator, pheno_metrics = evaluate(model, data_loader_test, device=device)

            if evaluator and evaluator.iou_types:
                first_iou_type = evaluator.iou_types[0]  # Typically "bbox"
                if first_iou_type in evaluator.coco_eval:
                    eval_obj = evaluator.coco_eval[first_iou_type]
                    fold_epoch_stats = eval_obj.stats
                    if fold_epoch_stats is not None and len(fold_epoch_stats) > 0:
                        if len(fold_results) > fold:  # if entry for this fold exists
                            fold_results[fold] = fold_epoch_stats  # update with latest epoch
                        else:
                            fold_results.append(fold_epoch_stats)

            if pheno_metrics:
                if len(fold_phenotype_metrics) > fold:  # if entry for this fold exists
                    fold_phenotype_metrics[fold] = pheno_metrics  # update with latest epoch
                else:
                    fold_phenotype_metrics.append(pheno_metrics)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")

    if fold_results:
        valid_fold_stats = [stats for stats in fold_results if stats is not None and len(stats) > 0]
        if valid_fold_stats:
            all_fold_stats_np = np.array(valid_fold_stats)
            mean_stats = np.mean(all_fold_stats_np, axis=0)
            std_stats = np.std(all_fold_stats_np, axis=0)
            print("\nAverage K-Fold Performance Metrics (based on last epoch of each fold):")
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

    if fold_phenotype_metrics:
        print("Average K-Fold Phenotype Regression Metrics (based on last epoch of each fold):")
        aggregated_pheno_results = {}
        phenotype_keys = ["fresh_weight", "height"]  # From engine.py
        metric_keys = ["r2", "rmse", "mape"]

        for p_key in phenotype_keys:
            aggregated_pheno_results[p_key] = {}
            for m_key in metric_keys:
                # Collect valid (non-NaN) metrics for this phenotype and metric type from all folds
                values = [
                    fold_data[p_key][m_key]
                    for fold_data in fold_phenotype_metrics
                    if
                    fold_data and p_key in fold_data and m_key in fold_data[p_key] and not np.isnan(
                        fold_data[p_key][m_key])
                ]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    aggregated_pheno_results[p_key][f'{m_key}_mean'] = mean_val
                    aggregated_pheno_results[p_key][f'{m_key}_std'] = std_val
                    if m_key == 'mape':  # MAPE is printed as percentage
                        print(
                            f"  {p_key:<15} {m_key:<10}: Mean = {mean_val * 100:.2f}%, Std = {std_val * 100:.2f}%")
                    else:
                        print(f"  {p_key:<15} {m_key:<10}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
                else:
                    aggregated_pheno_results[p_key][f'{m_key}_mean'] = np.nan  # Store NaN if no valid data
                    aggregated_pheno_results[p_key][f'{m_key}_std'] = np.nan
                    print(f"  {p_key:<15} {m_key:<10}: Not enough valid data across folds.")

        # Save Phenotype summary
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
                                f.write(
                                    f"    {m_key:<8}: Mean = {mean_val * 100:.2f}%, Std = {std_val * 100:.2f}%\n")
                            else:
                                f.write(f"    {m_key:<8}: Mean = {mean_val:.4f}, Std = {std_val:.4f}\n")
                        else:
                            f.write(f"    {m_key:<8}: Not enough valid data across folds.\n")
            np.savez(os.path.join(args.output_dir, "kfold_phenotype_stats.npz"),
                     aggregated_metrics=aggregated_pheno_results,
                     all_fold_metrics=fold_phenotype_metrics)
    else:
        print("No Phenotype metrics collected from K-Folds.")


def standard_training(args, num_classes, dataset, dataset_test, device):
    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers, "weights": args.weights}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    model = get_model(args.model, num_classes=num_classes, **kwargs)

    if args.saved_weights:
        print("Loading saved weights: {}".format(args.saved_weights))
        weights = torch.load(args.saved_weights, weights_only=False)["model"]
        model.load_state_dict(weights)

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

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=train_collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_collate_fn)

    model_without_ddp = model
    if args.distributed:
        model_without_ddp = model.module

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
        evaluate(model, data_loader_test, device=device)
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
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def main(args):
    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp", "coco_online", "coco_rgbd"):
        raise ValueError(f"Dataset should be coco, coco_kp, coco_online or coco-rgbd, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(is_train=True, args=args)

    if args.k_folds > 1:
        k_fold_training(args, num_classes, dataset, device)
    else:
        dataset_test, _ = get_dataset(is_train=False, args=args)
        standard_training(args, num_classes, dataset, dataset_test, device)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
