import argparse
import os

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from dataset import get_rgbd_data
from neural_networks import lettuce_regressor_model
from sklearn.model_selection import KFold

import my_utils as utils
from neural_networks.types import DualTensor


def calculate_metrics(y_true_tensor, y_pred_tensor):
    """
    Calculates R-squared, RMSE, NRMSE, and MAPE.
    Args:
        y_true_tensor (torch.Tensor): True target values (1D).
        y_pred_tensor (torch.Tensor): Predicted values (1D).
    Returns:
        tuple: (r2, rmse, nrmse, mape)
    """
    y_true_np = y_true_tensor.cpu().numpy().flatten()
    y_pred_np = y_pred_tensor.cpu().numpy().flatten()

    if len(y_true_np) == 0 or len(y_pred_np) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    if len(y_true_np) != len(y_pred_np):
        print(
            f"Warning: y_true ({len(y_true_np)}) and y_pred ({len(y_pred_np)}) have different lengths. Metrics will be NaN.")
        return float('nan'), float('nan'), float('nan'), float('nan')

    r2 = r2_score(y_true_np, y_pred_np)
    mse = np.mean((y_true_np - y_pred_np) ** 2)
    rmse = np.sqrt(mse)

    # NRMSE (normalized by mean of true values)
    mean_y_true = np.mean(y_true_np)
    if np.abs(mean_y_true) < 1e-9:  # Avoid division by zero or very small numbers
        nrmse = np.nan if rmse > 1e-9 else 0.0
    else:
        nrmse = rmse / mean_y_true

    # MAPE
    non_zero_mask = y_true_np != 0
    if np.sum(non_zero_mask) == 0:  # All true values are zero
        mape = 0.0 if np.allclose(y_pred_np, 0) else np.inf
    else:
        abs_error = np.abs(y_true_np[non_zero_mask] - y_pred_np[non_zero_mask])
        mape = np.mean(abs_error / np.abs(y_true_np[non_zero_mask])) * 100

    return r2, rmse, nrmse, mape


def evaluate_model(model, device, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    batches_processed = 0
    all_outputs_list = []
    all_targets_list = []

    with torch.no_grad():
        for batch_idx, (images, targets_dict_list) in enumerate(dataloader):
            tensors_to_stack = []
            for entry in targets_dict_list:
                raw_item = entry["phenotypes"][0]
                tensors_to_stack.append(raw_item)
            collated_targets = torch.stack(tensors_to_stack, dim=0).to(device)

            moved_images = []
            for img_pair_or_img in images:
                if isinstance(img_pair_or_img, list):
                    moved_images.append([img.to(device) for img in img_pair_or_img])
                else:
                    moved_images.append(img_pair_or_img.to(device))
            collated_images = DualTensor.collate(moved_images)
            collated_images.to(device) # If necessary

            output = model(collated_images)

            if output.shape != collated_targets.shape:
                print(
                    f"Eval Warning: Mismatch output ({output.shape}) vs target ({collated_targets.shape}). Adjusting.")
                if output.ndim == 1: output = output.unsqueeze(1)
                if collated_targets.ndim == 1: collated_targets = collated_targets.unsqueeze(1)
                if output.shape[0] != collated_targets.shape[0]:
                    print("Critical Eval: Batch size mismatch. Skipping batch.")
                    continue

            loss = criterion(output, collated_targets)
            total_loss += loss.item()
            batches_processed += 1

            all_outputs_list.append(output.cpu())
            all_targets_list.append(collated_targets.cpu())

    if not all_outputs_list:
        return 0.0, torch.empty(0), torch.empty(0)

    final_outputs = torch.cat(all_outputs_list)
    final_targets = torch.cat(all_targets_list)
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0

    return avg_loss, final_outputs, final_targets


def train_one_epoch(model, device, optimizer, dataloader, criterion):
    model.train()
    total_epoch_loss = 0.0
    batches_processed = 0
    all_outputs_list = []
    all_targets_list = []

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        optimizer.zero_grad()

        # need to do this cause my dataset code SUCKS BALLS
        collated_images = DualTensor.collate(images)
        collated_images.to(device)
        tensors_to_stack = []
        for entry in targets:
            item = entry["phenotypes"][0]
            tensors_to_stack.append(item)
        collated_targets = torch.stack(tensors_to_stack, dim=0)

        output = model(collated_images)

        if output.shape != collated_targets.shape:
            print(
                f"Warning: Mismatch between output shape ({output.shape}) and target shape ({collated_targets.shape}) in batch {batch_idx}.")

        loss = criterion(output, collated_targets)
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()
        batches_processed += 1

        all_outputs_list.append(output.detach().cpu())
        all_targets_list.append(collated_targets.detach().cpu())

    if not all_outputs_list:  # Handle case where no batches were processed
        return 0.0, 0, torch.empty(0), torch.empty(0)

    epoch_outputs = torch.cat(all_outputs_list)
    epoch_targets = torch.cat(all_targets_list)

    avg_epoch_loss = total_epoch_loss / batches_processed if batches_processed > 0 else 0

    return avg_epoch_loss, batches_processed, epoch_outputs, epoch_targets


def main(args):
    device = torch.device(args.device)

    kf = KFold(n_splits=args.k_folds, shuffle=True)

    transform = transforms.Compose([
        transforms.CenterCrop(1024),
        transforms.Resize((256, 256)),
        transforms.RandomPhotometricDistort(),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToImage(),
        transforms.ToDtype(torch.float, scale=True),
        transforms.ToPureTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = get_rgbd_data(args.root, "train", transform)

    criterion = nn.MSELoss()

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"----- Starting Fold {fold + 1}/{args.k_folds} -----")
        fold_dir = os.path.join(args.output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            collate_fn = utils.collate_fn,
            num_workers = args.num_workers,
        )
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

        # Initialize the model and optimizer
        model = lettuce_regressor_model(dual_branch=args.dual_branch)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

        fold_metrics_summary = {
            "train_loss": [], "r2_train": [], "rmse_train": [], "nrmse_train": [], "mape_train": []
        }

        # Train the model on the current fold
        for epoch in range(args.epochs):
            avg_train_loss, _, epoch_train_outputs, epoch_train_targets = train_one_epoch(
                model, device, optimizer, train_loader, criterion
            )

            # Ensure outputs and targets are 1D for metric calculation
            r2_tr, rmse_tr, nrmse_tr, mape_tr = calculate_metrics(
                epoch_train_targets.squeeze(), epoch_train_outputs.squeeze()
            )

            fold_metrics_summary["train_loss"].append(avg_train_loss)
            fold_metrics_summary["r2_train"].append(r2_tr)
            fold_metrics_summary["rmse_train"].append(rmse_tr)
            fold_metrics_summary["nrmse_train"].append(nrmse_tr)
            fold_metrics_summary["mape_train"].append(mape_tr)

            print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{args.epochs}]: "
                  f"Avg Train Loss: {avg_train_loss:.4f}, R²: {r2_tr:.4f}, RMSE: {rmse_tr:.4f}, "
                  f"NRMSE: {nrmse_tr:.4f}, MAPE: {mape_tr:.2f}%")

            # Save model checkpoint
            checkpoint_path = os.path.join(fold_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'r2': r2_tr,
                'rmse': rmse_tr,
                'fold': fold + 1
            }, checkpoint_path)
            # print(f"Saved checkpoint: {checkpoint_path}")

            lr_scheduler.step()

        # Evaluate the model on the test set for the current fold
        print(f"\nEvaluating Fold {fold + 1} on Test Set...")
        avg_test_loss, test_outputs, test_targets = evaluate_model(model, device, test_loader, criterion)

        r2_test, rmse_test, nrmse_test, mape_test = calculate_metrics(
            test_targets.squeeze(), test_outputs.squeeze()
        )

        print(f"----- Fold {fold + 1} Test Set Summary -----")
        print(f"  Average Loss: {avg_test_loss:.4f}")
        print(f"  R-squared (R²): {r2_test:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse_test:.4f}")
        print(f"  Normalized RMSE (NRMSE by mean): {nrmse_test:.4f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape_test:.2f}%")
        print("---------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="lettuce")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--dual-branch", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
    )
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    main(args)
