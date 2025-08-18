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
import torch.autograd.profiler as profiler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    return parser.parse_args()

class AsyncCustomDistributedResNet(nn.Module):
    def __init__(self, local_rank):
        super().__init__()
        self.model = torchvision.models.resnet152(num_classes=10)
        # self.model = torchvision.models.resnet101(num_classes=10)
        # self.model = torchvision.models.resnet50(num_classes=10)
        self.model = self.model.to(torch.float32)
        self.model.to(f'cuda:{local_rank}')
        
        self.local_rank = local_rank
        
        # Register backward hooks on each layer
        self.hook_handles = []
        self.reduction_handles = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks for each layer to trigger async all-reduce"""
        # Reverse the layers to match backward pass order
        for name, module in reversed(list(self.model.named_children())):
            layer_params = list(module.parameters())
            if not layer_params:  # Skip if no parameters
                continue
                
            # Register hooks for all parameters in the module
            for param in layer_params:
                handle = param.register_hook(
                    lambda grad, param_ref=param, layer_name=name: 
                    self._grad_hook(grad, param_ref, layer_name)
                )
                self.hook_handles.append(handle)
                
    def _grad_hook(self, grad, param, layer_name):
        """Hook called during backward pass when grad is computed"""
        # Launch async all-reduce
        handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
        
        # Store the handle for this reduction
        if layer_name not in self.reduction_handles:
            self.reduction_handles[layer_name] = []
        self.reduction_handles[layer_name].append(handle)
        
        # Return the gradient - will be modified in-place by all-reduce
        return grad
    
    def forward(self, x):
        # Clear any previous reduction handles
        self.reduction_handles = {}
        return self.model(x)
    
    def wait_for_reductions(self):
        """Wait for all async reductions to complete"""
        for layer_name, handles in self.reduction_handles.items():
            for handle in handles:
                handle.wait()

def main():
    args = parse_args()
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Get rank and world size
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    
    # Create model with async distributed implementation
    model = AsyncCustomDistributedResNet(local_rank)
    
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
        num_workers=0,
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
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(f'cuda:{local_rank}')
            targets = targets.to(f'cuda:{local_rank}')
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass - will trigger async all-reduce via hooks
            optimizer.zero_grad()
            loss.backward()
            
            # Wait for all async gradient reductions to complete
            model.wait_for_reductions()
            
            # Update weights
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print statistics every 20 batches
            if batch_idx % 20 == 0 and rank == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        if rank == 0:
            train_loss = running_loss / total
            train_acc = 100. * correct / total
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Total Epoch Time: {epoch_time:.2f}s")
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()