# KVTrans Engine - High-Performance RDMA KV Cache Transfer

KVTrans Engine is a high-performance, RDMA-based KV cache transfer system designed for distributed machine learning and high-throughput data processing applications. It provides a Python API for seamless integration with NumPy arrays and other data structures while leveraging the performance of InfiniBand RDMA for ultra-low latency communication.

## Project Structure

```
kvtrans/
├── engine.h          # C++ Engine class header with RDMA functionality
├── engine.cc         # C++ Engine implementation
├── pybind_engine.cc  # pybind11 wrapper for Python integration
├── Makefile          # Build configuration
├── test_engine.py    # Comprehensive test suite
├── demo.py           # Usage demonstration
└── README.md         # This file
```

## Prerequisites

### System Requirements
- Linux with InfiniBand support (optional for development)
- Python 3.7+ with development headers
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- pybind11 library
- NumPy (for array operations)

### Optional Dependencies
- InfiniBand drivers and libraries (`libibverbs-dev`)
- RDMA-capable network hardware

## Installation

1. **Install Python dependencies:**
   ```bash
   make install-deps
   ```

2. **Build the module:**
   ```bash
   make
   ```

3. **Run tests:**
   ```bash
   make test
   ```

## Usage Examples

### Basic Engine Setup

```python
import kvtrans_engine
import numpy as np

# Create engine with network interface, CPUs, connections per CPU, and listen port
engine = kvtrans_engine.Engine("eth0", ncpus=4, nconn_per_cpu=8, listen_port=12345)
```

### Client-Side Connection

```python
# Connect to remote server
success, conn_id = engine.connect("192.168.1.100", 12345)
if success:
    print(f"Connected with connection ID: {conn_id}")
```

### Server-Side Connection

```python
# Accept incoming connection
success, client_ip, client_port, conn_id = engine.accept()
if success:
    print(f"Accepted connection from {client_ip}:{client_port}")
```

### Key-Value Operations

```python
# Create and register data
data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
success, kv_id = engine.reg_kv(conn_id, data)

# Send data
if success:
    engine.send_kv(kv_id, data)
    
# Receive data
success, received_data = engine.recv_kv(kv_id, max_size=1024)
if success:
    print(f"Received: {received_data}")
```

### NumPy Array Transfer

```python
# Create large NumPy arrays
large_array = np.random.rand(1000, 1000).astype(np.float32)
weights = np.random.rand(256, 256).astype(np.float64)

# Register arrays for RDMA transfer
success1, kv_id1 = engine.reg_kv(conn_id, large_array)
success2, kv_id2 = engine.reg_kv(conn_id, weights)

# High-speed transfer
engine.send_kv(kv_id1, large_array)
engine.send_kv(kv_id2, weights)
```

## Development and Testing

### Build Targets
```bash
make all          # Build the module
make clean        # Clean build artifacts
make test         # Run test suite
make install-deps # Install dependencies
make help         # Show help
```