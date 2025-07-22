# UCCL P2P Engine - High-Performance RDMA P2P Transfer

UCCL P2P Engine is a high-performance, RDMA-based P2P transfer system designed for distributed machine learning and high-throughput data processing applications. It provides a Python API for seamless integration with PyTorch tensors, NumPy arrays, and other data structures while leveraging the performance of InfiniBand/RoCE RDMA for ultra-low latency communication.

UCCL has an experimental GPU-driven P2P engine, see [gpu_driven](../gpu_driven/) folder.

## Project Structure

```
p2p/
├── engine.h          # C++ Endpoint class header with RDMA functionality
├── engine.cc         # C++ Endpoint implementation
├── pybind_engine.cc  # pybind11 wrapper for Python integration
├── Makefile          # Build configuration
├── test_engine.py    # Comprehensive test suite
├── demo.py           # Usage demonstration
└── README.md         # This file
```

## Prerequisites

### System Requirements
- Linux with RDMA support (optional for development)
- Python 3.7+ with development headers
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- pybind11 library
- PyTorch or NumPy (for tensor/array operations)

### Optional Dependencies
- RDMA drivers and libraries (`libibverbs-dev`)
- RDMA-capable network hardware (InfiniBand, RoCE)
- CUDA (for GPU tensor operations)
- Install Redis 

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libhiredis-dev \
    libuv1-dev \
    pkg-config
```
and
```bash
git clone https://github.com/redis/hiredis.git
cd hiredis
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
sudo make install
cd ../..

git clone https://github.com/sewenew/redis-plus-plus.git
cd redis-plus-plus
mkdir build && cd build
cmake                                  \
  -DCMAKE_BUILD_TYPE=Release           \
  -DCMAKE_INSTALL_PREFIX=/usr/local    \
  -DREDIS_PLUS_PLUS_CXX_STANDARD=17    \
  -DREDIS_PLUS_PLUS_BUILD_ASYNC=libuv  \
  ..
make -j
sudo make install
```

## Installation

1. **Install Python dependencies:**
   ```bash
   make install-deps
   ```

2. **Build the UCCL P2P module:**
   ```bash
   make
   ```

3. **Run tests:**
   ```bash
   make test
   ```

4. **Install the UCCL P2P module:**
   ```bash
   make install
   ```
## Performance Benchmarks

Navigate to benchmarks directory:

```bash
cd benchmarks
```

### Running UCCL P2P

On server: 
```bash
python benchmark_uccl.py --role server --device gpu --local-gpu-idx 0 --num-cpus 4
```

On client:
```bash
# Note: if running atop GH200 with unified memory, use `--device cpu` or use `cudaMallocManaged`. 
python benchmark_uccl.py --role client --device gpu --local-gpu-idx 0 --num-cpus 4 --remote-ip <Server IP>
```

### Running NIXL

If you have not installed nixl with RDMA support, you can follow: 
<details><summary>Click me</summary>

```bash
sudo apt install build-essential cmake pkg-config 
pip3 install meson
pip3 install pybind11

git clone git@github.com:NVIDIA/gdrcopy.git
cd gdrcopy
sudo make prefix=/usr/local CUDA=/usr/local/cuda all install
cd ..

# Run these if you find there is no libcuda.so under /usr/local/cuda. Using GH200 as an example.
sudo ln -s /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so

wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
tar xzf ucx-1.18.0.tar.gz
cd ucx-1.18.0
./configure --prefix=/usr/local/ucx --enable-shared --disable-static \
            --disable-doxygen-doc --enable-optimizations --enable-cma \
            --enable-devel-headers --with-cuda=/usr/local/cuda \
            --with-gdrcopy=/usr/local --with-verbs --with-dm --enable-mt
make -j
sudo make -j install-strip
sudo ldconfig
cd ..

git clone https://github.com/ai-dynamo/nixl.git
cd nixl
meson setup build -Ducx_path=/usr/local/ucx
cd build
ninja
yes | ninja install
cd ..
pip install .
cd ..

UCX_LIB_PATH="/usr/local/ucx/lib"
export LD_LIBRARY_PATH="$UCX_LIB_PATH:$CONDA_PREFIX/lib/python3.13/site-packages/.nixl.mesonpy.libs/plugins:$LD_LIBRARY_PATH"
```
</details>

On Server:
```bash
UCX_MAX_RMA_LANES=4 UCX_IB_PCI_RELAXED_ORDERING=on UCX_NET_DEVICES=mlx5_2:1 UCX_TLS=rc \
    python benchmark_nixl.py --role server --device gpu --local-gpu-idx 0
```

On Client:
```bash
UCX_MAX_RMA_LANES=4 UCX_IB_PCI_RELAXED_ORDERING=on UCX_NET_DEVICES=mlx5_2:1 UCX_TLS=rc \
    python benchmark_nixl.py --role client --device gpu --local-gpu-idx 0 --remote-ip <Server IP>
```


### Running NCCL

On Server:
```bash
NCCL_NCHANNELS_PER_NET_PEER=4 \
python benchmark_nccl.py --role server --device gpu --local-gpu-idx 0
```

On Client:
```bash
NCCL_NCHANNELS_PER_NET_PEER=4 \
python benchmark_nccl.py --role client --device gpu --local-gpu-idx 0 --remote-ip <Server IP>
```


## Usage Examples

### Basic Endpoint Setup

```python
from uccl import p2p
import torch

# Create endpoint with local GPU index and number of CPUs
endpoint = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
```

### Client-Server Communication

```python
# Server side - accept incoming connections
success, remote_ip_addr, remote_gpu_idx, conn_id = endpoint.accept()
if success:
    print(f"Connected to {remote_ip_addr}, GPU {remote_gpu_idx}, conn_id={conn_id}")

# Client side - connect to remote server  
success, conn_id = endpoint.connect("192.168.1.100", remote_gpu_idx=1)
if success:
    print(f"Connected with conn_id={conn_id}")
```

### PyTorch Tensor Transfer

```python
# Sender side
send_tensor = torch.ones(1024, dtype=torch.float32)
assert send_tensor.is_contiguous()  # Ensure tensor is contiguous

# Register tensor for RDMA
success, mr_id = endpoint.reg(send_tensor.data_ptr(), send_tensor.numel() * 4)
assert success

# Send the tensor
success = endpoint.send(conn_id, mr_id, send_tensor.data_ptr(), send_tensor.numel() * 4)
assert success

# Receiver side
recv_tensor = torch.zeros(1024, dtype=torch.float32)
assert recv_tensor.is_contiguous()

# Register receive buffer
success, mr_id = endpoint.reg(recv_tensor.data_ptr(), recv_tensor.numel() * 4)
assert success

# Receive the tensor
success, recv_size = endpoint.recv(conn_id, mr_id, recv_tensor.data_ptr(), recv_tensor.numel() * 4)
assert success and recv_size == recv_tensor.numel() * 4
```

### NumPy Array Transfer

```python
import numpy as np

# Create and prepare NumPy array
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
assert data.flags['C_CONTIGUOUS']  # Ensure array is contiguous

# Register for RDMA
ptr = data.ctypes.data
size = data.nbytes
success, mr_id = endpoint.reg(ptr, size)

# Send array
if success:
    success = endpoint.send(conn_id, mr_id, ptr, size)
    
# Receive array
recv_data = np.zeros_like(data)
recv_ptr = recv_data.ctypes.data
success, recv_mr_id = endpoint.reg(recv_ptr, recv_data.nbytes)
success, recv_size = endpoint.recv(conn_id, recv_mr_id, recv_ptr, recv_data.nbytes)
```

### Vectorized Multi-Tensor Transfer

```python
# Sender side - send multiple tensors at once
tensors = [
    torch.ones(512, dtype=torch.float32),
    torch.ones(1024, dtype=torch.float32),
    torch.ones(256, dtype=torch.float32)
]

# Ensure all tensors are contiguous
for tensor in tensors:
    assert tensor.is_contiguous()

# Register all tensors
mr_ids = []
for tensor in tensors:
    success, mr_id = endpoint.reg(tensor.data_ptr(), tensor.numel() * 4)
    assert success
    mr_ids.append(mr_id)

# Prepare data for vectorized send
ptr_list = [tensor.data_ptr() for tensor in tensors]
size_list = [tensor.numel() * 4 for tensor in tensors]
num_iovs = len(tensors)

# Send all tensors in one operation
success = endpoint.sendv(conn_id, mr_ids, ptr_list, size_list, num_iovs)
assert success

# Receiver side - receive multiple tensors at once
recv_tensors = [
    torch.zeros(512, dtype=torch.float32),
    torch.zeros(1024, dtype=torch.float32),
    torch.zeros(256, dtype=torch.float32)
]

# Register receive buffers
recv_mr_ids = []
for tensor in recv_tensors:
    success, mr_id = endpoint.reg(tensor.data_ptr(), tensor.numel() * 4)
    assert success
    recv_mr_ids.append(mr_id)

# Prepare data for vectorized receive
recv_ptr_list = [tensor.data_ptr() for tensor in recv_tensors]
max_size_list = [tensor.numel() * 4 for tensor in recv_tensors]

# Receive all tensors in one operation
success, recv_size_list = endpoint.recvv(conn_id, recv_mr_ids, recv_ptr_list, max_size_list, num_iovs)
assert success
print(f"Received sizes: {recv_size_list}")
```


## API Reference

### Endpoint Class

#### Constructor
```python
Endpoint(local_gpu_idx, num_cpus)
```
Create a new RDMA endpoint instance.

**Parameters:**
- `local_gpu_idx` (int): GPU index for this endpoint
- `num_cpus` (int): Number of CPU threads to use for RDMA operations

#### Connection Management

```python
connect(remote_ip_addr, remote_gpu_idx) -> (success, conn_id)
```
Connect to a remote endpoint.

**Parameters:**
- `remote_ip_addr` (str): IP address of remote server
- `remote_gpu_idx` (int): GPU index of remote endpoint

**Returns:**
- `success` (bool): Whether connection succeeded
- `conn_id` (int): Connection ID for subsequent operations

```python
accept() -> (success, remote_ip_addr, remote_gpu_idx, conn_id)
```
Accept an incoming connection (blocking).

**Returns:**
- `success` (bool): Whether connection was accepted
- `remote_ip_addr` (str): IP address of connecting client
- `remote_gpu_idx` (int): GPU index of connecting client
- `conn_id` (int): Connection ID for subsequent operations

#### Memory Registration

```python
reg(ptr, size) -> (success, mr_id)
```
Register a memory region for RDMA operations.

**Parameters:**
- `ptr` (int): Memory pointer (use `tensor.data_ptr()` for PyTorch)
- `size` (int): Size in bytes

**Returns:**
- `success` (bool): Whether registration succeeded
- `mr_id` (int): Memory region ID for transfer operations

#### Data Transfer

```python
send(conn_id, mr_id, ptr, size) -> success
```
Send data to remote endpoint (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID from register
- `ptr` (int): Pointer to data to send
- `size` (int): Number of bytes to send

**Returns:**
- `success` (bool): Whether send completed successfully

```python
recv(conn_id, mr_id, ptr, max_size) -> (success, recv_size)
```
Receive data from remote endpoint (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID from register
- `ptr` (int): Pointer to buffer for received data
- `max_size` (int): Maximum number of bytes to receive

**Returns:**
- `success` (bool): Whether receive completed successfully
- `recv_size` (int): Number of bytes actually received

```python
sendv(conn_id, mr_id_list, ptr_list, size_list, num_iovs) -> success
```
Send multiple memory regions to remote endpoint in a single operation (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id_list` (list[int]): List of memory region IDs from register
- `ptr_list` (list[int]): List of pointers to data to send
- `size_list` (list[int]): List of sizes in bytes for each memory region
- `num_iovs` (int): Number of I/O vectors (length of the lists)

**Returns:**
- `success` (bool): Whether send completed successfully

```python
recvv(conn_id, mr_id_list, ptr_list, max_size_list, num_iovs) -> (success, recv_size_list)
```
Receive multiple memory regions from remote endpoint in a single operation (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id_list` (list[int]): List of memory region IDs from register
- `ptr_list` (list[int]): List of pointers to buffers for received data
- `max_size_list` (list[int]): List of maximum sizes in bytes for each memory region
- `num_iovs` (int): Number of I/O vectors (length of the lists)

**Returns:**
- `success` (bool): Whether receive completed successfully
- `recv_size_list` (list[int]): List of actual sizes received for each memory region

#### Asynchronous Transfer Operations

```python
send_async(conn_id, mr_id, ptr, size) -> (success, transfer_id)
```
Send data to remote endpoint asynchronously (non-blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID from register
- `ptr` (int): Pointer to data to send
- `size` (int): Number of bytes to send

**Returns:**
- `success` (bool): Whether send was initiated successfully
- `transfer_id` (int): Transfer ID for polling completion

```python
recv_async(conn_id, mr_id, ptr, max_size) -> (success, transfer_id)
```
Receive data from remote endpoint asynchronously (non-blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID from register
- `ptr` (int): Pointer to buffer for received data
- `size` (int): Exact number of bytes to receive

**Returns:**
- `success` (bool): Whether receive was initiated successfully
- `transfer_id` (int): Transfer ID for polling completion

```python
poll_async(transfer_id) -> (success, is_done)
```
Poll the status of an asynchronous transfer operation.

**Parameters:**
- `transfer_id` (int): Transfer ID returned by send_async or recv_async

**Returns:**
- `success` (bool): Whether polling succeeded
- `is_done` (bool): Whether the transfer has completed


## Development and Testing

### Build Targets
```bash
make all          # Build the module
make install      # Installs the module in python package path
make clean        # Clean build artifacts  
make test         # Run test suite
make install-deps # Install Python dependencies
make help         # Show available targets
```

### Testing Your Setup
```bash
# Run the included test suite
python3 test_engine.py

# Check if RDMA hardware is available
# (This will work even without RDMA hardware for testing)
```