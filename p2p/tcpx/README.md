# TCPX GPU-to-GPU Transfer

TCPX-based GPU-to-GPU data transfer implementation using [nccl-plugin-gpudirecttcpx](https://github.com/google/nccl-plugin-gpudirecttcpx).

## Features

- TCPX connection management (listen, accept, connect)
- CUDA device buffer registration
- Async send/receive operations
- RX metadata parsing and descriptor construction
- Multiple unpack implementations:
  - **D2D**: Device-to-device memcpy (default, recommended)
  - **Host**: Host-mediated gather (fallback for debugging)
  - **Kernel**: CUDA kernel-based unpack (experimental, not in this PR)

---

## Setup Guide

### Prerequisites

- CUDA Toolkit 11.0+
- H100 or compatible GPU with devmem-tcp support
- Linux kernel with devmem-tcp support
- C++17 compatible compiler
- NCCL GPUDirect TCPX plugin installed

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
export UCCL_HOME=/mnt/user_storage/uccl

# Navigate to TCPX directory
cd $UCCL_HOME/p2p/tcpx
```

### Step 3: Build

```bash
# Clean previous builds
make clean

# Build the test executable
make test_tcpx_transfer
```

### Step 4: Configure Unpack Mode

Choose the unpack implementation:

```bash
# Option 1: D2D mode (recommended for production)
export UCCL_TCPX_UNPACK_IMPL=d2d

# Option 2: Host mode (for debugging)
export UCCL_TCPX_UNPACK_IMPL=host
```

### Step 5: Run Tests

**Using the test script (recommended)**:

```bash
# Make script executable
chmod +x run_tcpx_test.sh

# Server (Node 1)
./run_tcpx_test.sh transfer server

# Client (Node 2)
./run_tcpx_test.sh transfer <server_ip>
```

**Manual execution**:

```bash
# Server (Node 1)
./tests/test_tcpx_transfer server

# Client (Node 2)
./tests/test_tcpx_transfer client <server_ip>
```

---

## Quick Start (TL;DR)

```bash
# One-time setup
sudo mkdir -p /usr/local/tcpx/lib64
sudo cp -r /var/lib/tcpx/lib64/* /usr/local/tcpx/lib64/
cd /usr/local/tcpx/lib64
sudo ln -sf libnccl-net.so libnccl-net-tcpx.so

# Build and run
export UCCL_HOME=/mnt/user_storage/uccl
cd $UCCL_HOME/p2p/tcpx
make clean && make test_tcpx_transfer
chmod +x run_tcpx_test.sh

# Choose mode
export UCCL_TCPX_UNPACK_IMPL=d2d  # or 'host'

# Run test
./run_tcpx_test.sh transfer server  # Node 1
./run_tcpx_test.sh transfer <ip>    # Node 2
```

---

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `UCCL_TCPX_UNPACK_IMPL` | `d2d` (default) | Device-to-device memcpy |
| | `host` | Host-mediated transfer |
| | `kernel` | GPU kernel (experimental) |
| `UCCL_TCPX_DEBUG` | `0` or `1` | Enable verbose logging |

---

## Expected Output

**Successful test**:
```
[DEBUG] === TCPX GPU-to-GPU transfer test ===
[DEBUG] Received data (23 bytes): Hello from TCPX client!
[DEBUG] ✓ Data validation PASSED
[DEBUG] Server test completed successfully
```

---

## Architecture

```
Client                          Server
  │                               │
  ├─ cudaMalloc(send_buf)        ├─ cudaMalloc(recv_buf)
  ├─ tcpx_reg_mr(send_buf)       ├─ tcpx_reg_mr(recv_buf)
  ├─ Write payload to GPU        │
  ├─ tcpx_isend() ──────────────>├─ tcpx_irecv()
  │                               ├─ Poll tcpx_test()
  │                               ├─ Parse RX metadata
  │                               ├─ buildDescriptorBlock()
  │                               ├─ Execute unpack (D2D/host)
  │                               ├─ Validate received data
  └─ Cleanup                      └─ Cleanup
```

---
## Components

### TCPX Interface Layer
- `tcpx_interface.h`: C API definitions
- `tcpx_impl.cc`: TCPX plugin wrapper implementation

### RX Descriptor Module
- `rx_descriptor.h`: Header-only descriptor construction utilities
  - Uses `tcpx::plugin::loadMeta` as descriptor type (avoids duplication)
  - Provides `buildDescriptorBlock()` inline function

### Device Unpack (`device/`)
- `unpack_kernels.cu`: CUDA kernels for GPU-side unpack (experimental)
- `unpack_launch.{h,cu}`: Kernel launcher and configuration (experimental)
- **Note**: Device unpack is not included in this PR

### Tests (`tests/`)
- `test_tcpx_transfer.cc`: End-to-end GPU-to-GPU transfer validation

---

## Troubleshooting

### Plugin Not Found

**Error**: `Failed to load TCPX plugin`

**Solution**:
```bash
# Verify plugin exists
ls -la /usr/local/tcpx/lib64/libnccl-net-tcpx.so

# If not, run setup again
sudo mkdir -p /usr/local/tcpx/lib64
sudo cp -r /var/lib/tcpx/lib64/* /usr/local/tcpx/lib64/
cd /usr/local/tcpx/lib64
sudo ln -sf libnccl-net.so libnccl-net-tcpx.so
```

### Build Errors

**Error**: `cuda_runtime.h: No such file or directory`

**Solution**:
```bash
# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Connection Timeout

**Error**: `Bootstrap connection timeout`

**Solution**:
- Verify server IP is correct
- Check firewall allows TCP port 12345
- Ensure both nodes can ping each other
- Verify TCPX plugin is loaded on both nodes

### Data Validation Failed

**Error**: `Data validation FAILED`

**Solution**:
- Try host mode: `export UCCL_TCPX_UNPACK_IMPL=host`
- Enable debug logging: `export UCCL_TCPX_DEBUG=1`
- Check GPU memory is not corrupted
- Verify CUDA driver version compatibility


---

## Implementation Details

### TCPX Plugin Integration

Uses NCCL plugin v7 APIs:
- `ncclNetPlugin_v7`: Main plugin interface
- `ncclNet_v7_t`: Network operations (init, listen, accept, connect)
- Device handle management for GPU memory registration

### RX Metadata Format

TCPX delivers fragments with metadata:
```c
struct loadMeta {
  uint32_t src_off;  // Offset in bounce buffer
  uint32_t len;      // Fragment length
  uint64_t dst_off;  // Offset in destination buffer
};
```

### Memory Management

- **Bounce buffers**: TCPX plugin managed (devmem-tcp mapped GPU memory)
- **Destination buffers**: Application allocated via `cudaMalloc`
- **Descriptor blocks**: Host-side structures for unpack operations

---


## Known Limitations

1. **Kernel mode**: Direct GPU kernel access to devmem-tcp bounce buffers requires staging buffer workaround (not in this PR)

2. **Multi-fragment optimization**: Current D2D implementation issues one copy per fragment

---


## References

- [nccl-plugin-gpudirecttcpx](https://github.com/google/nccl-plugin-gpudirecttcpx)
- [Linux devmem-tcp](https://lwn.net/Articles/945687/)

