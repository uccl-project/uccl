import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import argparse
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    return parser.parse_args()

class TimerContext:
    def __init__(self, name, enable=True):
        self.name = name
        self.enable = enable
        
    def __enter__(self):
        if self.enable:
            torch.cuda.synchronize()
            self.start = time.time()
        return self
        
    def __exit__(self, *args):
        if self.enable:
            torch.cuda.synchronize()
            self.end = time.time()
            self.elapsed = self.end - self.start

class CustomDistributedResNet(nn.Module):
    def __init__(self, local_rank):
        super().__init__()
        self.model = torchvision.models.resnet152(num_classes=10)
        self.model.to(f'cuda:{local_rank}')
        
        # Track layers and their reduction times
        self.layer_times = OrderedDict()
        self.local_rank = local_rank
        
        # Group parameters by layers for reduction
        self.param_groups = self._group_parameters()
        
    def _group_parameters(self):
        """Group parameters by layers"""
        param_groups = []
        for name, module in self.model.named_children():
            layer_params = list(module.parameters())
            if layer_params:  # Only add if layer has parameters
                param_groups.append((name, layer_params))
        return param_groups
    
    def forward(self, x):
        return self.model(x)
    
    def reduce_gradients(self):
        """Perform layer-wise gradient reduction"""
        total_reduction_time = 0
        
        for layer_name, params in self.param_groups:
            with TimerContext(f"reduce_{layer_name}") as t:
                for param in params:
                    if param.grad is not None:
                        # All-reduce for this layer's gradients
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                
            self.layer_times[layer_name] = t.elapsed
            total_reduction_time += t.elapsed
            
        return total_reduction_time

def main():
    args = parse_args()
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Get rank and world size
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    # Create model with custom distributed implementation
    model = CustomDistributedResNet(local_rank)
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    # Datasets and loaders
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train)
    
    train_sampler = data.distributed.DistributedSampler(train_dataset)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(f'cuda:{local_rank}')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        running_loss = 0.0
        total = 0
        correct = 0

        # Epoch timing
        total_epoch_time = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            with TimerContext("batch") as batch_timer:
                inputs = inputs.to(f'cuda:{local_rank}')
                targets = targets.to(f'cuda:{local_rank}')
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass with timing
                optimizer.zero_grad()
                with TimerContext("backward") as backward_timer:
                    loss.backward()
                
                # Layer-wise gradient reduction
                with TimerContext("reduction") as reduction_timer:
                    total_reduction_time = model.reduce_gradients()
                
                # Update weights
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            total_epoch_time += batch_timer.elapsed

            # Print statistics every 20 batches
            if batch_idx % 20 == 0 and local_rank == 0:
                print(f'\nBatch {batch_idx}:')
                print(f'Total Batch Time: {batch_timer.elapsed*1000:.2f}ms')
                print(f'Backward Time: {backward_timer.elapsed*1000:.2f}ms')
                print(f'Total Reduction Time: {reduction_timer.elapsed*1000:.2f}ms')
                print(f'Layer-wise Reduction Times:')
                for layer_name, reduce_time in model.layer_times.items():
                    print(f'  {layer_name}: {reduce_time*1000:.2f}ms')
                print(f'Loss: {loss.item():.4f}')
        
        # Print epoch results
        if local_rank == 0:
            train_loss = running_loss / total
            train_acc = 100. * correct / total
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Total Epoch Time: {total_epoch_time:.2f}s")
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()