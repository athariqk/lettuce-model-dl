import argparse

import torch
from torch import nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from dataset import get_rgbd_data
from neural_networks import lettuce_regressor_model
from sklearn.model_selection import KFold

import my_utils as utils
from neural_networks.types import DualTensor


def train_one_epoch(model, device, optimizer, dataloader, criterion):
    model.train()
    total_epoch_loss = 0.0
    batches_processed = 0
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

    return total_epoch_loss, batches_processed


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
        print(f"Start training for fold {fold + 1}")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            collate_fn = utils.collate_fn
        )
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
            collate_fn=utils.collate_fn
        )

        # Initialize the model and optimizer
        model = lettuce_regressor_model(dual_branch=args.dual_branch)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

        # Train the model on the current fold
        for epoch in range(args.epoch):
            total_loss, batches_processed = train_one_epoch(model, device, optimizer, train_loader, criterion)
            lr_scheduler.step()

            avg_epoch_loss = total_loss / batches_processed
            print(f"Epoch [{epoch}/{args.epoch}]:  Average Training Loss: {avg_epoch_loss:.4f}")

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / len(test_loader)

        # Print the results for the current fold
        print(
            f"Fold {fold + 1}: Test set: average loss: {test_loss:.4f}, accuracy: {correct}/{len(test_loader)} ({accuracy:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="lettuce")
    parser.add_argument("--dual-branch", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=30)
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
    args = parser.parse_args()

    main(args)
