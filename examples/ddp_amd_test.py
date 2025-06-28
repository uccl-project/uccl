import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size per GPU"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training for AMD GPUs"""
    # Check if ROCm is available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm/CUDA not available. Please install ROCm for AMD GPUs."
        )

    # Initialize process group
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Multi-node setup
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single node setup
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = local_rank
        world_size = torch.cuda.device_count()

    print(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    return rank, local_rank, world_size


def main():
    args = parse_args()

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    # Initialize process group (RCCL is compatible with NCCL API for AMD GPUs)
    dist.init_process_group(backend="nccl")

    # Set device for this process (works with both CUDA and ROCm)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"Process {rank}/{world_size} using device: {device}")
    print(f"Device name: {torch.cuda.get_device_name(device)}")

    # Build the model and move it to the corresponding device
    model = torchvision.models.resnet18(num_classes=10)
    model.to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Prepare the dataset (using CIFAR10 as an example)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # Use DistributedSampler for data parallelism
    train_sampler = data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    test_sampler = data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)

        # === Training phase ===
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print progress every 100 batches for rank 0
            if rank == 0 and batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Gather metrics from all processes
        total_samples = torch.tensor(total, dtype=torch.float, device=device)
        total_correct = torch.tensor(correct, dtype=torch.float, device=device)
        total_loss = torch.tensor(
            running_loss, dtype=torch.float, device=device
        )

        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        train_loss = total_loss.item() / total_samples.item()
        train_acc = 100.0 * total_correct.item() / total_samples.item()

        # === Testing phase ===
        model.eval()
        test_loss = 0.0
        test_total = 0
        test_correct = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(
                    device, non_blocking=True
                ), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        # Gather test metrics from all processes
        test_total_samples = torch.tensor(
            test_total, dtype=torch.float, device=device
        )
        test_total_correct = torch.tensor(
            test_correct, dtype=torch.float, device=device
        )
        test_total_loss = torch.tensor(
            test_loss, dtype=torch.float, device=device
        )

        dist.all_reduce(test_total_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_total_loss, op=dist.ReduceOp.SUM)

        test_loss = test_total_loss.item() / test_total_samples.item()
        test_acc = 100.0 * test_total_correct.item() / test_total_samples.item()

        lr_scheduler.step()

        # Only rank 0 prints epoch results
        if rank == 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )

    # Cleanup
    dist.destroy_process_group()
    print(f"Process {rank} finished training.")


if __name__ == "__main__":
    main()
