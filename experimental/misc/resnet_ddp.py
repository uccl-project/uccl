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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Initialize process group
    dist.init_process_group(backend="nccl")
    
    # 2. Get local rank from environment variable
    local_rank = int(os.environ["LOCAL_RANK"])  # This is automatically set by torchrun
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 3. 构建模型并移动到对应设备
    model = torchvision.models.resnet18(num_classes=10)
    model.to(device)

    # 4. 使用 DDP 包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 5. 准备数据集（示例使用 CIFAR10）
    #    注意这里使用了 DistributedSampler 实现分布式并行读取数据
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    train_sampler = data.distributed.DistributedSampler(train_dataset)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    test_sampler = data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # 6. 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # 学习率调整策略（可按需选择）
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 7. 开始训练
    for epoch in range(args.epochs):
        # 分布式 sampler 设置 epoch，确保 shuffle 随机数可复现
        train_sampler.set_epoch(epoch)

        # === 训练阶段 ===
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # === 测试阶段 ===
        model.eval()
        test_sampler.set_epoch(epoch)
        test_loss = 0.0
        test_total = 0
        test_correct = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / test_total
        test_acc = 100. * test_correct / test_total

        lr_scheduler.step()

        # 只让 rank=0 的进程打印结果，避免多卡打印重复信息
        if dist.get_rank() == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # 8. 训练结束后，销毁进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
