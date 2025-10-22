# TCPX P2P Library

High-performance GPU-to-GPU communication library using Google's TCPX (GPUDirect over TCP).

**Status**: ✅ Phase 1 Complete - API layer working, benchmark validated (~9 GB/s)

---

## Quick Start

### Step 1: Install TCPX Plugin

The TCPX plugin is typically installed at `/var/lib/tcpx/lib64`. We need to create a symlink at `/usr/local/tcpx/lib64` for compatibility:

```bash
# 1. Create the target directory
sudo mkdir -p /usr/local/tcpx/lib64

# 2. Copy TCPX libraries from the default installation path
sudo cp -r /var/lib/tcpx/lib64/* /usr/local/tcpx/lib64/

# 3. Create the NCCL plugin symlink
cd /usr/local/tcpx/lib64
sudo ln -sf libnccl-net.so libnccl-net-tcpx.so
```

### Step 2: Set Environment Variables

```bash
# Set UCCL home directory
export UCCL_HOME=/your/uccl

# Navigate to TCPX directory
cd $UCCL_HOME/p2p/tcpx
```

### Step 3: Build
```bash
make clean
make all
```

### Step 4: Run
```bash
# Server (node 1)
cd scripts
./run_p2p_fullmesh.sh server

# Client (node 2)
cd scripts
./run_p2p_fullmesh.sh client <server_ip>
```

### Expected Output
```
[PERF] Bandwidth: 9.XX GB/s (per GPU)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UCCL_TCPX_NUM_CHANNELS` | 2 | Number of TCPX channels per GPU |
| `UCCL_TCPX_PERF_SIZE` | 64MB | Transfer size per iteration |
| `UCCL_TCPX_PERF_ITERS` | 10 | Number of iterations |
| `UCCL_TCPX_CHUNK_BYTES` | 512KB | Chunk size for pipelining |
| `NCCL_NSOCKS_PERTHREAD` | 2 | Sockets per TCPX channel |
| `NCCL_SOCKET_NTHREADS` | 1 | Threads per TCPX comm |

---

## Architecture

### Components

**Phase 1 API Layer** (✅ Complete):
- **TcpxSession**: Session management (listen, accept, connect, memory registration)
- **TcpxTransfer**: Data plane operations (send, recv, completion tracking)
- **TcpxHelpers**: Core logic (completion polling, event management)

**Foundation Layer**:
- **ChannelManager**: Multi-channel TCPX connection management
- **Bootstrap**: Connection handle exchange between nodes
- **Transfer Flow Control**: Managed within `TcpxTransfer` via per-channel windows
- **UnpackKernel**: GPU kernel for scattered-to-contiguous memory copy

### Data Flow

**Server (Receiver)**:
1. `TcpxSession::listen()` → Get connection info
2. Bootstrap: Send connection info to client
3. `TcpxSession::accept()` → Accept connections
4. `TcpxSession::registerMemory()` → Register GPU buffer
5. `TcpxTransfer::createTransfer()` → Create transfer object
6. Loop:
   - `transfer->postRecv()` → Post receive requests
   - `transfer->wait()` → Wait for completion
   - `transfer->release()` → Release resources

**Client (Sender)**:
1. Bootstrap: Receive connection info from server
2. `TcpxSession::loadRemoteConnInfo()` → Load server handles
3. `TcpxSession::connect()` → Connect to server
4. `TcpxSession::registerMemory()` → Register GPU buffer
5. `TcpxTransfer::createTransfer()` → Create transfer object
6. Loop:
   - `transfer->postSend()` → Post send requests
   - `transfer->wait()` → Wait for completion
   - `transfer->release()` → Release resources

---

## File Structure

```
p2p/tcpx/
├── README.md                     # This file
├── Makefile                      # Build system
├── docs/
│   ├── roadmap.md                # Long-term development plan
│   └── history.md                # Development history and fixes
│
├── include/                      # Public API headers
│   ├── session_manager.h         # Session management API
│   ├── transfer_manager.h        # Transfer API
│   ├── transfer_flow.h           # Core flow types + helper declarations
│   ├── tcpx_logging.h            # Logging utilities
│   ├── unpack_descriptor.h       # Unpack descriptor helpers + plugin structs
│   ├── channel_manager.h         # Channel management facade
│   └── bootstrap.h               # Bootstrap protocol + handle definition
│
├── src/                          # Implementation
│   ├── session_manager.cc        # Session implementation
│   ├── transfer_manager.cc       # Transfer implementation
│   ├── transfer_flow.cc          # Helper implementation (stage 2 flow)
│   ├── channel_manager.cc        # Channel management
│   └── bootstrap.cc              # Bootstrap protocol
│
├── device/                       # GPU kernels
│   ├── unpack_kernels.cu         # Unpack kernel
│   ├── unpack_launch.cu          # Kernel launcher
│   └── unpack_launch.h           # Launcher API
│
├── tests/                        # Tests and benchmarks
│   ├── test_tcpx_perf_multi.cc   # Main benchmark (Phase 1 API)
│   ├── tcpx_perf_runner.h        # Benchmark runner API
│   └── tcpx_perf_runner.cc       # Benchmark runner implementation
│
├── archive/                      # Historical reference
│   └── test_tcpx_perf_multi.cc.original  # Original working implementation
│
└── tcpx_impl.cc                  # TCPX plugin wrapper
```

---

## API Usage Example

### Server

```cpp
#include "session_manager.h"
#include "transfer_manager.h"

// 1. Create session
TcpxSession session(gpu_id, num_channels);

// 2. Listen and get connection info
std::string conn_info = session.listen();
// ... send conn_info to client via bootstrap ...

// 3. Accept connection
session.accept("client");

// 4. Register memory
void* gpu_buffer = allocate_gpu_buffer(size);
uint64_t mem_id = session.registerMemory(gpu_buffer, size, NCCL_PTR_CUDA, true);

// 5. Create transfer
TcpxTransfer* transfer = session.createTransfer("client");

// 6. Receive data
for (int i = 0; i < num_chunks; i++) {
  transfer->postRecv(mem_id, offset, chunk_size);
}
transfer->wait();
transfer->release();

delete transfer;
```

### Client

```cpp
  #include "session_manager.h"
  #include "transfer_manager.h"

// 1. Create session
TcpxSession session(gpu_id, num_channels);

// 2. Load server connection info
// ... receive conn_info from server via bootstrap ...
session.loadRemoteConnInfo("server", conn_info);

// 3. Connect to server
session.connect("server");

// 4. Register memory
void* gpu_buffer = allocate_gpu_buffer(size);
uint64_t mem_id = session.registerMemory(gpu_buffer, size, NCCL_PTR_CUDA, false);

// 5. Create transfer
TcpxTransfer* transfer = session.createTransfer("server");

// 6. Send data
for (int i = 0; i < num_chunks; i++) {
  transfer->postSend(mem_id, offset, chunk_size);
}
transfer->wait();
transfer->release();

delete transfer;
```

---

## Performance

- **Hardware**: 2 nodes, 8x H100 GPUs, 4x gVNIC (200 Gbps each)
- **Configuration**: 4 channels × 2 sockets per GPU
- **Bandwidth**: ~9 GB/s per GPU (validated)
- **Latency**: ~8 ms for 64MB transfer

---


## Debugging

```bash
# Enable TCPX debug logs
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE

# Enable internal debug logs
export TCPX_DEBUG=1
export TCPX_PERF=1

```

---

## Documentation

- **docs/roadmap.md**: Long-term development plan (Phase 2: NIXL plugin, Phase 3: Integration)
- **docs/history.md**: Development history, critical fixes, and lessons learned

---
