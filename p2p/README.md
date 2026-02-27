# UCCL P2P Engine - High-Performance RDMA P2P Transfer

UCCL P2P Engine is a high-performance, RDMA-based P2P transfer system designed for distributed machine learning and high-throughput data processing applications. It provides a Python API for seamless integration with PyTorch tensors, NumPy arrays, and other data structures while leveraging the performance of InfiniBand/RoCE RDMA for ultra-low latency communication.

UCCL has an experimental GPU-driven P2P engine, see [ep](../ep/) folder.

## Project Structure

```
p2p/
├── engine.h          # C++ Endpoint class header with RDMA functionality
├── engine.cc         # C++ Endpoint implementation
├── engine_pybind.cc  # pybind11 wrapper for Python integration
├── Makefile          # Build configuration
├── tests/            # Comprehensive test suite
├── benchmarks/       # Comprehensive benchmark suite
└── README.md         # This file
```

## Prerequisites

The easiest way is to: 
```bash
git clone https://github.com/uccl-project/uccl.git --recursive
cd uccl && bash build.sh [cuda|rocm] p2p [py_version] --install
```

Alternatively, you can setup your local dev environment by: 

<details><summary>Click me</summary>

### System Requirements
- Linux with RDMA support
- Python 3.7+ with development headers
- C++17 compatible compiler
- pybind11 library
- PyTorch (for tensor/array operations)

```bash
sudo apt install build-essential net-tools libelf-dev libibverbs-dev \
                 libgoogle-glog-dev libgtest-dev libgflags-dev -y
```

### Installation

1. **Install Pybind11 dependency:**
   ```bash
   make install-deps
   ```

2. **Build the UCCL P2P module:**
   ```bash
   make -j
   ```

3. **Install the UCCL P2P module:**
   ```bash
   make install
   ```

</details>

To build AWS EFA support, you can: 
```bash
USE_EFA=1 bash build.sh cuda p2p --install
# or
make -j USE_EFA=1 install
```

To build GCP TCPX support, you can refer to [NIXL_plugin_readme.md](./NIXL_plugin_readme.md). 

## Performance Benchmarks

### Running UCCL P2P

On client: 
```bash
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=0 --master_addr=<IP addr> benchmarks/benchmark_uccl.py
```

On server:
```bash
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=<IP addr> benchmarks/benchmark_uccl.py
```

Notes: 
* You may consider exporting `GLOO_SOCKET_IFNAME=xxx` if triggering Gloo connectFullMesh failure.
* You may consider exporting `UCCL_P2P_RDMA_GID_INDEX` if your cluster requires it for NCCL to run (usually 1, or 3 in some testbed).
* **You must first import `torch` before importing `uccl.p2p` for AMD GPUs**, otherwise, `RuntimeError: No HIP GPUs are available` will occur. We guess this is because torch does some extra init for AMD GPUs, in order for Pybind-C++ code to work. 
* To benchmark dual direction transfer, `benchmark_uccl.py --dual`.
* To benchmark intra-node transfer via CUDA/HIP IPC, `torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --ipc`.
* To benchmark one-sided IPC write (GPU-to-GPU or CPU-to-GPU), `torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --write-ipc`. Use `--device cpu --pinned` for CPU source buffers.
* To benchmark one-sided IPC read (GPU-to-GPU or GPU-to-CPU), `torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --read-ipc`. Use `--device cpu --pinned` for CPU destination buffers.
* To benchmark one-sided READ/WRITE transfer, `benchmark_uccl_readwrite.py`.
* To benchmark UCCL copy-only collectives (eg, sendrecv, allgather), `benchmark_uccl_collective.py`. You can also run ring-like communication pattern with `--ring`.
* From CollectiveContext, the default parameter `use_copy_engine_for_intra` is `False`, which means it will use NCCL/RCCL via `torch.distributed` for intra-node communication; if setting to `True`, it will use GPU copy engine (eg, `cudaMemcpy`) via UCCL for intranode communication. 

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| UCCL_P2P_LOG_LEVEL | Logging level | WARNING (others: INFO, ERROR, FATAL) |
| UCCL_P2P_RDMA_GID_INDEX | GID index in RDMA network | 0/3 (EFA/IB) |
| UCCL_P2P_RDMA_SL | Service level in RDMA network | 8/3 (EFA/IB) |
| UCCL_P2P_RDMA_TC | Traffic class in RDMA network | 104 (IB) |

### Running NCCL

On Client:
```bash
NCCL_NCHANNELS_PER_NET_PEER=4 torchrun --nnodes=2 --nproc_per_node=1 --node-rank=0 --master_addr=<IP addr> benchmarks/benchmark_nccl.py
```

On Server:
```bash
NCCL_NCHANNELS_PER_NET_PEER=4 torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=<IP addr> benchmarks/benchmark_nccl.py
```

Notes: 
* You can specify `NCCL_IB_HCA=mlx5_2:1` to control which NIC and port to use. 
* If you see errors like `message size truncated`, it is likely caused by NCCL version mismatch. We suggest specifying `LD_PRELOAD=<path to libnccl.so.2>`. 
* To benchmark dual direction transfer, you can add `--dual`. 
* Consider tune `NCCL_IB_GID_INDEX=3` if NCCL triggers errors.
* This also works for AMD GPUs.

### Running NIXL with UCCL backend

If you have not installed nixl, you can follow:
<details><summary>Click me</summary>

```bash
sudo apt install build-essential cmake pkg-config autoconf automake libtool -y
pip3 install meson pybind11

git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
sudo make prefix=/usr/local/gdrcopy CUDA=/usr/local/cuda all install
cd ..

# Run these if you find there is no libcuda.so under /usr/local/cuda. Using GH200 as an example.
sudo ln -s /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so

git clone https://github.com/ai-dynamo/nixl.git && cd nixl && git checkout 0.5.0
meson setup build --prefix=/usr/local/nixl -Ducx_path=/usr/local/ucx -Ddisable_gds_backend=true
cd build
ninja
yes | ninja install
cd ..
pip install .
cd ..

export LD_LIBRARY_PATH="/usr/local/nixl/lib/`uname -m`-linux-gnu/plugins"
```
</details>

On Server:
```bash
python benchmark_nixl.py --role server --backend uccl
```

On Client:
```bash
python benchmark_nixl.py --role client --remote-ip <Server IP> --backend uccl
```

### Running NIXL with UCX backend

If you have not installed nixl with UCX backend, you can follow: 
<details><summary>Click me</summary>

```bash
sudo apt install build-essential cmake pkg-config autoconf automake libtool -y
pip3 install meson pybind11

git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
sudo make prefix=/usr/local/gdrcopy CUDA=/usr/local/cuda all install
cd ..

# Run these if you find there is no libcuda.so under /usr/local/cuda. Using GH200 as an example.
sudo ln -s /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so

# Install UCX
git clone https://github.com/openucx/ucx.git && cd ucx && git checkout v1.19.x
./autogen.sh
./configure --prefix=/usr/local/ucx --enable-shared --disable-static \
            --disable-doxygen-doc --enable-optimizations --enable-cma \
            --enable-devel-headers --with-cuda=/usr/local/cuda \
            --with-gdrcopy=/usr/local/gdrcopy --with-verbs --with-dm --enable-mt
make -j
sudo make -j install-strip
sudo ldconfig
cd ..

git clone https://github.com/ai-dynamo/nixl.git && cd nixl && git checkout 0.5.0
meson setup build --prefix=/usr/local/nixl -Ducx_path=/usr/local/ucx -Ddisable_gds_backend=true 
cd build
ninja
yes | ninja install
cd ..
pip install .
cd ..

export LD_LIBRARY_PATH="/usr/local/nixl/lib/`uname -m`-linux-gnu/plugins:/usr/local/ucx/lib:$LD_LIBRARY_PATH"
```
</details>

On Server:
```bash
UCX_MAX_RMA_LANES=4 UCX_IB_PCI_RELAXED_ORDERING=on UCX_NET_DEVICES=mlx5_2:1 UCX_TLS=cuda,rc \
python benchmark_nixl.py --role server
```

On Client:
```bash
UCX_MAX_RMA_LANES=4 UCX_IB_PCI_RELAXED_ORDERING=on UCX_NET_DEVICES=mlx5_2:1 UCX_TLS=cuda,rc \
python benchmark_nixl.py --role client --remote-ip <Server IP>
```

Notes: 
* You can specify `--op-type read` to benchmark one-sided READ transfer in NIXL. On GH200, we find NIXL READ over GPU memory is extremely slow with 1GB/s out of 25, while NIXL READ over CPU memory is better but only max at 9GB/s. 

### Running NIXL on AMD+Broadcom

Run `./run_container.sh` to launch containers on two servers; then inside the container, run the following: 
```bash
# On server
UCX_MAX_RMA_LANES=4 UCX_NET_DEVICES=rdma3:1 UCX_TLS=rocm,rc python benchmark_nixl.py --role server

# On client
UCX_MAX_RMA_LANES=4 UCX_NET_DEVICES=rdma3:1 UCX_TLS=rocm,rc python benchmark_nixl.py --role client --remote-ip <Server IP>
```

Notes: 
* You can specify `--dual --remote-ip <Remote IP>` to benchmark dual-direction NIXL transfer. 

### Running NIXL with Mooncake backend

If you have not installed nixl with Mooncake backend, you can follow:
<details><summary>Click me</summary>

```bash
sudo apt install build-essential cmake pkg-config autoconf automake libtool -y
pip3 install meson pybind11

git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
sudo make prefix=/usr/local/gdrcopy CUDA=/usr/local/cuda all install
cd ..

# Run these if you find there is no libcuda.so under /usr/local/cuda. Using GH200 as an example.
sudo ln -s /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so

# Install Mooncake
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
sudo bash dependencies.sh
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j
sudo make install
cd ../..

git clone https://github.com/ai-dynamo/nixl.git && cd nixl && git checkout 0.5.0
meson setup build --prefix=/usr/local/nixl
cd build
ninja
yes | ninja install
cd ..
pip install .
cd ..

export LD_LIBRARY_PATH="/usr/local/nixl/lib/`uname -m`-linux-gnu/plugins:$LD_LIBRARY_PATH"
```
</details>

On Server:
```bash
python benchmark_nixl.py --role server --backend mooncake
```

On Client:
```bash
python benchmark_nixl.py --role client --remote-ip <Server IP> --backend mooncake
```

### Running Mooncake Transfer Engine Bench

Run `./run_container.sh` to launch containers on three servers; then inside the container, run the following: 
```bash
# Metadata server: 
cd /tmp/Mooncake/mooncake-transfer-engine/example/http-metadata-server
go get github.com/kvcache-ai/Mooncake/mooncake-transfer-engine/example/http-metadata-server
go run . --addr=:8080

# Target server: 
cd /tmp/Mooncake/build/mooncake-transfer-engine/example/
./transfer_engine_bench --mode=target --metadata_server=http://<metadata server ip>:8080/metadata \
    --local_server_name=my_target --device_name=rdma3

# Initiator server: 
cd /tmp/Mooncake/build/mooncake-transfer-engine/example/
echo "" > /io/p2p/mooncake.txt
for bs in 256 1024 4096 16384 65536 262144 1048576 10485760 16777216 104857600; do
    ./transfer_engine_bench --metadata_server=http://<metadata server ip>:8080/metadata \
        --segment_id=my_target --local_server_name=my_initiator --device_name=rdma3 \
        --batch_size=1 --threads=1 --operation=write --block_size=$bs >> /io/p2p/mooncake.txt 2>&1
done
```


## Usage Examples

<details><summary>Click me</summary>

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
success = endpoint.recv(conn_id, mr_id, recv_tensor.data_ptr(), recv_tensor.numel() * 4)
assert success
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
success = endpoint.recv(conn_id, recv_mr_id, recv_ptr, recv_data.nbytes)
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
size_list = [tensor.numel() * 4 for tensor in recv_tensors]

# Receive all tensors in one operation
success = endpoint.recvv(conn_id, recv_mr_ids, recv_ptr_list, size_list, num_iovs)
assert success
```

</details>


## API Reference

<details><summary>Click me</summary>

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
Note that a connection is one direction, only allowing the client (that calls `connect()`) to send data to the server (that calls `accept()`). 
If you want bi-directional communication, you should create two connections. 

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

```python
add_remote_endpoint(metadata_bytes) -> (success, conn_id)
```
Add a remote endpoint using serialized metadata. Connects only once per remote endpoint — if a connection to the same remote endpoint already exists, the cached `conn_id` is returned directly.

**Parameters:**
- `metadata_bytes` (bytes): Serialized endpoint metadata (obtained from `get_metadata()`) containing IP address, port, and GPU index

**Returns:**
- `success` (bool): Whether connection succeeded (or was already cached)
- `conn_id` (int): Connection ID for subsequent operations

```python
start_passive_accept() -> success
```
Start a background thread that continuously accepts incoming connections. This is useful when you don't want to block the main thread on `accept()` calls — the background thread will automatically accept all incoming connections.

**Returns:**
- `success` (bool): Whether the background accept thread was started

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
recv(conn_id, mr_id, ptr, size) -> success
```
Receive data from remote endpoint (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID from register
- `ptr` (int): Pointer to buffer for received data
- `size` (int): Number of bytes to receive

**Returns:**
- `success` (bool): Whether receive completed successfully

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
recvv(conn_id, mr_id_list, ptr_list, size_list, num_iovs) -> success
```
Receive multiple memory regions from remote endpoint in a single operation (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id_list` (list[int]): List of memory region IDs from register
- `ptr_list` (list[int]): List of pointers to buffers for received data
- `size_list` (list[int]): List of sizes in bytes for each memory region
- `num_iovs` (int): Number of I/O vectors (length of the lists)

**Returns:**
- `success` (bool): Whether receive completed successfully

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
recv_async(conn_id, mr_id, ptr, size) -> (success, transfer_id)
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

#### One-Sided RDMA Operations

```python
read(conn_id, mr_id, dst, size, slot_item) -> success
```
Read data from remote endpoint using one-sided RDMA READ operation (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID of remote data to read
- `dst` (int): Pointer to local destination buffer
- `size` (int): Number of bytes to read
- `slot_item` (FifoItem): Slot item for RDMA operation coordination (contains the remote address to read from)

**Returns:**
- `success` (bool): Whether read completed successfully

```python
read_async(conn_id, mr_id, dst, size, slot_item) -> (success, transfer_id)
```
Read data from remote endpoint using one-sided RDMA READ operation asynchronously (non-blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID of remote data to read
- `dst` (int): Pointer to local destination buffer
- `size` (int): Number of bytes to read
- `slot_item` (FifoItem): Slot item for RDMA operation coordination (contains the remote address to read from)

**Returns:**
- `success` (bool): Whether read was initiated successfully
- `transfer_id` (int): Transfer ID for polling completion

```python
readv(conn_id, mr_id_list, dst_list, size_list, slot_item_list, num_iovs) -> success
```
Read multiple memory regions from remote endpoint using one-sided RDMA READ operations in a single operation (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id_list` (list[int]): List of memory region IDs of remote data to read
- `dst_list` (list[int]): List of pointers to local destination buffers
- `size_list` (list[int]): List of sizes in bytes for each memory region
- `slot_item_list` (list[FifoItem]): List of slot items for RDMA operation coordination (contains the remote address to read from)
- `num_iovs` (int): Number of I/O vectors (length of the lists)

**Returns:**
- `success` (bool): Whether read completed successfully

```python
advertise(conn_id, mr_id, addr, len, out_buf) -> success
```
Advertise memory region information to remote endpoint for one-sided RDMA operations.

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID to advertise
- `addr` (int): Pointer to the memory region
- `len` (int): Size of the memory region in bytes
- `out_buf` (str): Output buffer to store advertisement metadata

**Returns:**
- `success` (bool): Whether advertisement completed successfully

```python
advertisev(conn_id, mr_id_list, addr_list, len_list, out_buf_list, num_iovs) -> success
```
Advertise multiple memory regions to remote endpoint for one-sided RDMA operations in a single operation.

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id_list` (list[int]): List of memory region IDs to advertise
- `addr_list` (list[int]): List of pointers to memory regions
- `len_list` (list[int]): List of sizes in bytes for each memory region
- `out_buf_list` (list[str]): List of output buffers to store advertisement metadata
- `num_iovs` (int): Number of I/O vectors (length of the lists)

**Returns:**
- `success` (bool): Whether advertisement completed successfully

```python
write(conn_id, mr_id, src, size, slot_item) -> success
```
Write data to remote endpoint using one-sided RDMA WRITE operation (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID of remote destination
- `src` (int): Pointer to local buffer
- `size` (int): Number of bytes to write
- `slot_item` (FifoItem): Slot item for RDMA operation coordination (contains the remote address to write to)

**Returns:**
- `success` (bool): Whether write completed successfully

```python
write_async(conn_id, mr_id, src, size, slot_item) -> (success, transfer_id)
```
Write data to remote endpoint using one-sided RDMA WRITE operation asynchronously (non-blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID of remote destination
- `src` (int): Pointer to local buffer
- `size` (int): Number of bytes to write
- `slot_item` (FifoItem): Slot item for RDMA operation coordination (contains the remote address to write to)

**Returns:**
- `success` (bool): Whether write was initiated successfully
- `transfer_id` (int): Transfer ID for polling completion

```python
writev(conn_id, mr_id_list, src_list, size_list, slot_item_list, num_iovs) -> success
```
Write multiple memory regions to remote endpoint using one-sided RDMA WRITE operations in a single operation (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id_list` (list[int]): List of memory region IDs of remote destinations
- `src_list` (list[int]): List of pointers to local buffers
- `size_list` (list[int]): List of sizes in bytes for each memory region
- `slot_item_list` (list[FifoItem]): List of slot items for RDMA operation coordination (contains the remote address to write to)
- `num_iovs` (int): Number of I/O vectors (length of the lists)

**Returns:**
- `success` (bool): Whether write completed successfully

</details>


## Testing

```bash
python tests/test_engine_send.py
python tests/test_engine_read.py
python tests/test_engine_write.py
python tests/test_engine_metadata.py
torchrun --nnodes=1 --nproc_per_node=2 tests/test_engine_nvlink.py

# One-sided IPC correctness tests (write_ipc, read_ipc, writev_ipc, readv_ipc — sync and async)
# Verifies that each API correctly copies data from source to destination buffers.
torchrun --nnodes=1 --nproc_per_node=2 tests/test_engine_onesided_ipc.py

# One-sided IPC benchmarks (write_ipc / read_ipc)
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --write-ipc
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --read-ipc
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --write-ipc --device cpu --pinned
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --read-ipc --device cpu --pinned
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --write-ipc --async-api
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --read-ipc --async-api

# Vectorized one-sided IPC benchmarks (writev_ipc / readv_ipc), e.g. 4 buffers per call
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --write-ipc --num-kvblocks 4
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --read-ipc --num-kvblocks 4
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --write-ipc --num-kvblocks 4 --async-api
torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --read-ipc --num-kvblocks 4 --async-api
```
