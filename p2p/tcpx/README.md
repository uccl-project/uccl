# TCPX P2P Performance Benchmark

Multi-channel TCPX GPU-to-GPU performance test using Google's nccl-plugin-gpudirecttcpx.

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
./run_p2p_fullmesh.sh server

# Client (node 2)
./run_p2p_fullmesh.sh client <server_ip>
```

### Expected Output in each server's log
```
[PERF] Avg (10 iter): 22.42 ms, BW: 2.85 GB/s
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UCCL_TCPX_NUM_CHANNELS` | 2 | Number of TCPX channels |
| `UCCL_TCPX_PERF_SIZE` | 4MB | Transfer size per iteration |
| `UCCL_TCPX_PERF_ITERS` | 10 | Number of iterations |
| `UCCL_TCPX_CHUNK_BYTES` | 512KB | Chunk size |
| `UCCL_TCPX_UNPACK_IMPL` | kernel | Unpack mode: kernel/d2d/host |
| `NCCL_NSOCKS_PERTHREAD` | 2 | Sockets per thread (TCPX plugin) |
| `NCCL_SOCKET_NTHREADS` | 1 | Threads per comm (TCPX plugin) |

---

## Architecture

### Components

- **ChannelManager**: Manages multiple TCPX connections per GPU
- **Bootstrap**: Exchanges connection handles between server/client
- **SlidingWindow**: Prevents exhausting TCPX request pool (MAX_REQUESTS=16)
- **UnpackKernel**: GPU kernel to copy scattered 4KB fragments to contiguous memory
- **UnpackLauncher**: Manages CUDA streams and events

### Data Flow

**Server (Receiver)**:
1. Listen on all channels → Bootstrap send handles → Accept connections
2. CUDA init → Register GPU receive buffer
3. Main loop:
   - Post `tcpx_irecv` (round-robin across channels)
   - Wait for completion via `tcpx_test`
   - Launch GPU unpack kernel
   - Call `tcpx_irecv_consumed` after kernel completes

**Client (Sender)**:
1. Bootstrap receive handles → Connect to all channels
2. CUDA init → Register GPU send buffer
3. Main loop:
   - Post `tcpx_isend` (round-robin across channels)
   - Wait for completion via `tcpx_test`

---

## File Structure

```
p2p/tcpx/
├── Makefile
├── tcpx_impl.cc                      # TCPX plugin wrapper
├── include/                          # Headers (8 files)
│   ├── tcpx_interface.h
│   ├── tcpx_structs.h
│   ├── channel_manager.h
│   ├── bootstrap.h
│   ├── sliding_window.h
│   └── rx_descriptor.h
├── src/                              # Implementation (3 files)
│   ├── channel_manager.cc
│   ├── bootstrap.cc
│   └── sliding_window.cc
├── device/                           # GPU code (3 files)
│   ├── unpack_kernels.cu
│   ├── unpack_launch.cu
│   └── unpack_launch.h
└── tests/
    └── test_tcpx_perf_multi.cc       # Main program
```

---

## Advanced Usage

### Multi-Channel Configuration

```bash
# 1 channel × 8 sockets (maximize single channel bandwidth)
UCCL_TCPX_NUM_CHANNELS=1 NCCL_NSOCKS_PERTHREAD=8 ./tests/test_tcpx_perf_multi server 0

# 2 channels × 4 sockets (balanced)
UCCL_TCPX_NUM_CHANNELS=2 NCCL_NSOCKS_PERTHREAD=4 ./tests/test_tcpx_perf_multi server 0

# 4 channels × 2 sockets (current default)
UCCL_TCPX_NUM_CHANNELS=4 NCCL_NSOCKS_PERTHREAD=2 ./tests/test_tcpx_perf_multi server 0
```

**Note**: `total_sockets = UCCL_TCPX_NUM_CHANNELS × NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS`

