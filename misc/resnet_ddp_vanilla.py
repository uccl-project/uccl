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
        self.model = torchvision.models.resnet18(num_classes=10)
        self.model.to(f'cuda:{local_rank}')
        
        # Get total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        # Calculate size for gradient buffer (float16 for memory efficiency)
        self.gradient_buffer = torch.zeros(total_params, 
                                         device=f'cuda:{local_rank}',
                                         dtype=torch.float16)
        
        # Initialize timing statistics
        self.allreduce_time = 0.0
        self.total_time = 0.0
        self.num_steps = 0
        
    def forward(self, x):
        return self.model(x)
    
    def all_reduce_gradients(self):
        with TimerContext("allreduce") as t:
            # Flatten all gradients into our pre-allocated buffer
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_data = param.grad.data.half()  # Convert to float16
                    numel = param.grad.numel()
                    self.gradient_buffer[offset:offset + numel].copy_(grad_data.view(-1))
                    offset += numel
            
            # Perform all-reduce on the buffer
            dist.all_reduce(self.gradient_buffer, op=dist.ReduceOp.SUM)
            
            # Copy reduced gradients back
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    numel = param.grad.numel()
                    param.grad.data.copy_(self.gradient_buffer[offset:offset + numel]
                                        .view_as(param.grad.data).float())
                    offset += numel
        
        self.allreduce_time += t.elapsed
        self.num_steps += 1

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

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)

    # Distributed samplers
    train_sampler = data.distributed.DistributedSampler(train_dataset)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    test_sampler = data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(f'cuda:{local_rank}')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        running_loss = 0.0
        total = 0
        correct = 0
        epoch_start = time.time()
        
        # Reset timing stats for this epoch
        model.allreduce_time = 0.0
        model.num_steps = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            with TimerContext("batch") as batch_timer:
                inputs = inputs.to(f'cuda:{local_rank}')
                targets = targets.to(f'cuda:{local_rank}')
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Custom all-reduce
                model.all_reduce_gradients()
                
                # Update weights
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Print timing statistics every 20 batches
            if batch_idx % 20 == 0 and local_rank == 0:
                avg_allreduce_time = model.allreduce_time / max(1, model.num_steps)
                allreduce_percentage = (avg_allreduce_time / batch_timer.elapsed) * 100
                
                print(f'Epoch: {epoch} | Batch: {batch_idx} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Batch Time: {batch_timer.elapsed*1000:.2f}ms | '
                      f'AllReduce Time: {avg_allreduce_time*1000:.2f}ms '
                      f'({allreduce_percentage:.1f}%)')

        epoch_time = time.time() - epoch_start
        
        # Calculate metrics
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        
        # Calculate average timing statistics for the epoch
        avg_allreduce_time = model.allreduce_time / max(1, model.num_steps)
        allreduce_percentage = (model.allreduce_time / epoch_time) * 100
        
        # Evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(f'cuda:{local_rank}')
                targets = targets.to(f'cuda:{local_rank}')
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / test_total
        test_acc = 100. * test_correct / test_total
        
        # Update learning rate
        lr_scheduler.step()
        
        # Print results (only from rank 0)
        if local_rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Average AllReduce Time: {avg_allreduce_time*1000:.2f}ms")
            print(f"AllReduce Time Percentage: {allreduce_percentage:.1f}%\n")
            
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()