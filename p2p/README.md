# UCCL P2P Engine - High-Performance RDMA P2P Transfer

UCCL P2P Engine is a high-performance, RDMA-based P2P transfer system designed for distributed machine learning and high-throughput data processing applications. It provides a Python API for seamless integration with PyTorch tensors, NumPy arrays, and other data structures while leveraging the performance of InfiniBand/RoCE RDMA for ultra-low latency communication.

UCCL has a GPU-driven P2P engine, see [ep](../ep/) folder.

## Project Structure

```
p2p/
├── engine.h          # C++ Endpoint class header with RDMA functionality
├── engine.cc         # C++ Endpoint implementation
├── engine_api.cc     # nanobind wrapper for Python integration
├── Makefile          # Build configuration
├── tests/            # Comprehensive test suite
├── benchmarks/       # Comprehensive benchmark suite
└── README.md         # This file
```

## Prerequisites

The easiest way is to: 
```bash
git clone https://github.com/uccl-project/uccl.git && cd uccl
bash build.sh [cu12|cu13|roc7|roc6] p2p --install
```

Alternatively, you can setup your local dev environment by: 

<details><summary>Click me</summary>

### System Requirements
- Linux with RDMA support
- Python 3.8+ with development headers
- C++17 compatible compiler
- nanobind library
- PyTorch (for tensor/array operations)

```bash
sudo apt install build-essential net-tools libelf-dev libibverbs-dev \
                 libgtest-dev libgflags-dev -y
```

### Installation

1. **Install nanobind dependency:**
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

To enable AWS EFA support, you can do the same as above, and specify `UCCL_P2P_TRANSPORT=efa` during runtime. 

To enable GCP TCPX support, you can refer to [NIXL_plugin_readme.md](./NIXL_plugin_readme.md).

To enable HPE Slingshot/CXI support, set `UCCL_P2P_TRANSPORT=cxi` at runtime:
```bash
sudo apt install libfabric-dev libhwloc-dev -y
make -j install
UCCL_P2P_TRANSPORT=cxi UCCL_P2P_DISABLE_IPC=1 torchrun ...
```

To build with DietGPU float compression support, you can:
```bash
USE_DIETGPU=1 make -j install
# or
USE_DIETGPU=1 bash build.sh cu12 p2p --install
```

DietGPU provides lossless GPU-side compression for float16/bfloat16/float32 tensors. It only activates for transfers larger than 2 MB. At runtime, control compression behavior via the `UCCL_P2P_COMPRESS_STRATEGY` environment variable (see the environment variable table below).

Compression also applies to one-sided `write`/`writev` when `UCCL_P2P_COMPRESS_STRATEGY=split_only`. Only the `split_only` strategy is supported for the write path (not `encode`/`full`). The mechanism is transparent: the sender compresses float data in two GPU kernel phases and writes the result directly into a pre-allocated GPU buffer on the receiver side; once all data arrives, the receiver decompresses into the advertised destination address and sends a small RDMA ack to release the sender's buffer slot. Both sides must be built with `USE_DIETGPU=1`.

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
* You may consider exporting `GLOO_SOCKET_IFNAME=xxx NCCL_SOCKET_IFNAME=xxx` if triggering Gloo connectFullMesh failure.
* You may consider exporting `UCCL_P2P_RDMA_GID_INDEX` if your cluster requires it for NCCL to run (usually 1, or 3 in some testbed).
* You can specify `UCCL_P2P_TRANSPORT=ib|efa|nccl|tcp|tcpx|cxi` at runtime to choose different network backends. The default is `ib` that works for NVIDIA, Broadcom, AMD, and Intel RDMA NICs.
* **You must first import `torch` before importing `uccl.p2p` for AMD GPUs**, otherwise, `RuntimeError: No HIP GPUs are available` will occur. We guess this is because torch does some extra init for AMD GPUs, in order for Pybind-C++ code to work. 
* One-sided network write is the default in `benchmark_uccl.py`; use `--mode read` for RDMA read.
* To benchmark one-sided IPC write (GPU-to-GPU or CPU-to-GPU), `torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --write-ipc`. Use `--device cpu --pinned` for CPU source buffers.
* To benchmark one-sided IPC read (GPU-to-GPU or GPU-to-CPU), `torchrun --nproc_per_node=2 benchmarks/benchmark_uccl.py --read-ipc`. Use `--device cpu --pinned` for CPU destination buffers.

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| UCCL_P2P_LOG_LEVEL | Logging level | WARNING (others: INFO, ERROR, FATAL) |
| UCCL_P2P_RDMA_GID_INDEX | GID index in RDMA network | 0/3 (EFA/IB) |
| UCCL_P2P_RDMA_MAX_RD_ATOMIC | Maximum outstanding RDMA READ requests initiated per QP | 16 |
| UCCL_P2P_RDMA_MAX_DEST_RD_ATOMIC | Maximum outstanding RDMA READ requests accepted per QP from the peer | 16 |
| UCCL_P2P_RDMA_SL | Service level in RDMA network | 8/3 (EFA/IB) |
| UCCL_P2P_RDMA_TC | Traffic class in RDMA network | 104 (IB) |
| UCCL_P2P_RDMA_DEV | RDMA devices forced to use (instead of auto-selecting based on PCIe affinity) | none (eg, `irdma-mkp0,irdma-mkp1`) |
| UCCL_P2P_TRANSPORT | Network backend to use at runtime | ib (others: efa/nccl/tcp/tcpx/cxi) |
| UCCL_CXI_DOMAIN | CXI/libfabric domain to use when `UCCL_P2P_TRANSPORT=cxi` | auto from GPU index, eg `cxi0` |
| UCCL_CXI_DEVICE_INDEX | CXI device index used for automatic domain selection | GPU index modulo 4 |
| UCCL_CXI_THREADING | libfabric threading hint for the CXI domain | endpoint |
| UCCL_CXI_TX_QUEUE_SIZE | CXI transmit queue size | 4096 |
| UCCL_CXI_RX_QUEUE_SIZE | CXI receive queue size | 4096 |
| UCCL_CXI_CQ_SIZE | CXI completion queue size | 8192 |
| UCCL_P2P_MAX_INFLIGHT_OPS | Maximum one-sided in-flight operations; CXI defaults lower than RDMA | 32 for CXI, otherwise internal maximum |
| UCCL_LIBFABRIC_SO | Override libfabric shared-library path for the CXI dlsym wrapper | auto-detect `libfabric.so` / `libfabric.so.1` |
| UCCL_P2P_COMPRESS_STRATEGY | DietGPU compression strategy (requires `USE_DIETGPU=1` build) | none |
| UCCL_RDMA_ADAPTIVE_SLEEP | Enable adaptive sleeping on proxy threads, by putting the proxy threads into a sleeping state if there have been no new work requests / RDMA completion events after 120s. | null |

`UCCL_P2P_COMPRESS_STRATEGY` accepted values:
* `none` / `off` / `0` — no compression
* `split` / `split_only` — pipelined: transfer the uncompressed portion of the float data immediately, then ANS-encode and transfer the remainder
* `encode` / `split_encode` / `full` / `1` — blocking: ANS-encode all float data first, then transfer everything once encoding is complete

Example:
```bash
UCCL_P2P_COMPRESS_STRATEGY=encode torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=0 --master_addr=<IP addr> benchmarks/benchmark_uccl.py
```

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

# Create endpoint with local GPU index
endpoint = p2p.Endpoint(local_gpu_idx=0)
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

The active transfer API is one-sided RDMA READ/WRITE. See
`benchmark_uccl.py` for end-to-end examples using registered buffers and FIFO
metadata.

</details>


## API Reference

<details><summary>Click me</summary>

### Endpoint Class

#### Constructor
```python
Endpoint(local_gpu_idx)
```
Create a new RDMA endpoint instance.

**Parameters:**
- `local_gpu_idx` (int): GPU index for this endpoint

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
advertise(mr_id, addr, len) -> (success, meta_blob)
```
Advertise memory region information for one-sided RDMA operations.

**Parameters:**
- `mr_id` (int): Memory region ID to advertise
- `addr` (int): Pointer to the memory region
- `len` (int): Size of the memory region in bytes

**Returns:**
- `success` (bool): Whether advertisement completed successfully
- `meta_blob` (bytes): 64-byte serialized FifoItem metadata

```python
advertisev(mr_id_list, addr_list, len_list, num_iovs) -> (success, meta_blob_list)
```
Advertise multiple memory regions for one-sided RDMA operations in a single operation.

**Parameters:**
- `mr_id_list` (list[int]): List of memory region IDs to advertise
- `addr_list` (list[int]): List of pointers to memory regions
- `len_list` (list[int]): List of sizes in bytes for each memory region
- `num_iovs` (int): Number of I/O vectors (length of the lists)

**Returns:**
- `success` (bool): Whether advertisement completed successfully
- `meta_blob_list` (list[bytes]): Serialized FifoItem metadata for each buffer

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
python tests/test_engine_read.py
python tests/test_engine_write.py
torchrun --nnodes=1 --nproc_per_node=2 tests/test_engine_nvlink.py

# One-sided IPC correctness tests (write_ipc, read_ipc, writev_ipc, readv_ipc — sync and async)
# Verifies that each API correctly copies data from source to destination buffers.
torchrun --nnodes=1 --nproc_per_node=2 tests/test_engine_onesided_ipc.py
```
