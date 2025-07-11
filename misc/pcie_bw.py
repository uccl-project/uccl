import torch
import time


def measure_pcie_bandwidth(buffer_size_bytes, num_iters=100, device_index=1):
    """Measure PCIe bandwidth (H2D, D2H, D2D) for a specific GPU device."""

    device = torch.device(f"cuda:{device_index}")

    # Create pinned memory buffer (Host memory)
    host_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8).pin_memory()

    # Create device buffers (GPU memory)
    device_buffer = torch.empty(
        buffer_size_bytes, dtype=torch.uint8, device=device
    )

    # Measure H2D (Host to Device) bandwidth
    torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(num_iters):
        device_buffer.copy_(host_buffer, non_blocking=True)
    torch.cuda.synchronize(device)
    end = time.time()
    h2d_bandwidth = (
        (buffer_size_bytes * num_iters) / (end - start) / 1e9
    )  # GB/s

    # Measure D2H (Device to Host) bandwidth
    torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(num_iters):
        host_buffer.copy_(device_buffer, non_blocking=True)
    torch.cuda.synchronize(device)
    end = time.time()
    d2h_bandwidth = (
        (buffer_size_bytes * num_iters) / (end - start) / 1e9
    )  # GB/s

    # Measure D2D (Device to Device) bandwidth
    device_buffer2 = torch.empty(
        buffer_size_bytes, dtype=torch.uint8, device=device
    )
    torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(num_iters):
        device_buffer2.copy_(device_buffer, non_blocking=True)
    torch.cuda.synchronize(device)
    end = time.time()
    d2d_bandwidth = (
        (buffer_size_bytes * num_iters) / (end - start) / 1e9
    )  # GB/s

    # Measure Device to Device bandwidth over NVLink between two GPUs
    peer_bandwidth = 0.0
    num_gpus = torch.cuda.device_count()
    
    if device_index + 1 < num_gpus:
        device2 = torch.device(f"cuda:{device_index + 1}")  # Next GPU
        
        # Check if peer access is supported
        if torch.cuda.can_device_access_peer(device.index, device2.index):
            peer_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device=device2)
            
            # Measure D2D bandwidth between different GPUs
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                peer_buffer.copy_(device_buffer, non_blocking=True)
            torch.cuda.synchronize()
            end = time.time()
            peer_bandwidth = ((buffer_size_bytes * num_iters) / (end - start) / 1e9)  # GB/s
        else:
            print(f"Warning: Peer access not supported between GPU {device_index} and GPU {device_index + 1}")
    else:
        print(f"Warning: Only {num_gpus} GPU(s) available, cannot measure peer bandwidth")
    
    return h2d_bandwidth, d2h_bandwidth, d2d_bandwidth, peer_bandwidth


if __name__ == "__main__":
    buffer_size = 64 * 1024 * 1024  # 64 MB buffer
    num_iters = 100
    device_index = 0  # Second GPU (index 0 in PyTorch)

    h2d, d2h, d2d, peer = measure_pcie_bandwidth(buffer_size, num_iters, device_index)
    print(f"Buffer size: {buffer_size / (1024 * 1024)} MB")
    print(f"H2D Bandwidth (Host to Device): {h2d:.2f} GB/s")
    print(f"D2H Bandwidth (Device to Host): {d2h:.2f} GB/s")
    print(f"D2D Bandwidth (Device to Device): {d2d:.2f} GB/s")
    print(f"D2D Bandwidth (Device to Device over NVLink): {peer:.2f} GB/s")