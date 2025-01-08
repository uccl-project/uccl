import torch
import time


def measure_bandwidth(buffer_size_bytes, num_iters=100, device="cuda"):
    """Measure cudaMemcpy bandwidth for H2D, D2H, and D2D."""

    # Create buffers
    host_buffer = torch.empty(
        buffer_size_bytes, dtype=torch.uint8
    ).pin_memory()  # Pinned memory for H2D and D2H
    device_buffer1 = torch.empty(
        buffer_size_bytes, dtype=torch.uint8, device=device
    )
    device_buffer2 = torch.empty(
        buffer_size_bytes, dtype=torch.uint8, device=device
    )

    # H2D (Host to Device)
    start = time.time()
    for _ in range(num_iters):
        device_buffer1.copy_(host_buffer, non_blocking=True)
        torch.cuda.synchronize()  # Ensure copy completion
    end = time.time()
    h2d_bandwidth = (
        (buffer_size_bytes * num_iters) / (end - start) / 1e9
    )  # GB/s

    # D2H (Device to Host)
    start = time.time()
    for _ in range(num_iters):
        host_buffer.copy_(device_buffer1, non_blocking=True)
        torch.cuda.synchronize()  # Ensure copy completion
    end = time.time()
    d2h_bandwidth = (
        (buffer_size_bytes * num_iters) / (end - start) / 1e9
    )  # GB/s

    # D2D (Device to Device)
    start = time.time()
    for _ in range(num_iters):
        device_buffer2.copy_(device_buffer1, non_blocking=True)
        torch.cuda.synchronize()  # Ensure copy completion
    end = time.time()
    d2d_bandwidth = (
        (buffer_size_bytes * num_iters) / (end - start) / 1e9
    )  # GB/s

    return h2d_bandwidth, d2h_bandwidth, d2d_bandwidth


if __name__ == "__main__":
    buffer_size = 64 * 1024 * 1024  # 64 MB buffer
    num_iters = 100

    h2d, d2h, d2d = measure_bandwidth(buffer_size, num_iters)
    print(f"Buffer size: {buffer_size / (1024 * 1024)} MB")
    print(f"H2D Bandwidth: {h2d:.2f} GB/s")
    print(f"D2H Bandwidth: {d2h:.2f} GB/s")
    print(f"D2D Bandwidth: {d2d:.2f} GB/s")
