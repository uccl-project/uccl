# NIXL with UCCL_P2P (TCPX)

Quick setup and run guide for running the NCCL backend with TCPX runtime.

## Git submodules

From the repo root:

```bash
git submodule update --init thirdparty/nccl thirdparty/nccl-tests
```

## Prerequisites

```bash
export UCCL_HOME=/path/to/your/uccl

# Install dependencies
sudo apt update
sudo apt install -y build-essential net-tools libelf-dev libibverbs-dev \
                    libgtest-dev libgflags-dev libaio-dev \
                    python3-dev pybind11-dev python3-pip python3-pybind11
python3 -m pip install --user nanobind pyzmq intervaltree paramiko pybind11

cd $UCCL_HOME/thirdparty/nccl
git checkout v2.28.9-1
make src.build -j NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

cd $UCCL_HOME/thirdparty/nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$UCCL_HOME/thirdparty/nccl/build -j
```

## Build UCCL P2P engine

```bash
cd $UCCL_HOME/p2p
make clean
make -j
sudo make install
sudo ldconfig
```

## Stage TCPX runtime

Many images mount `/var/lib/tcpx` with `noexec`, and some nodes already have
`/usr/local/lib/libnccl-net.so -> libnccl-net-ofi.so`. For TCPX, stage the
runtime libraries into an executable directory and put that directory first in
`LD_LIBRARY_PATH`.

Run this once on every node before launching the benchmark. The
`collective/rdma/run_nccl_test_tcpx.sh` helper does the same staging
automatically.

```bash
export TCPX_RUNTIME_LIBDIR=/usr/local/tcpx/lib64
sudo install -d -m 755 "${TCPX_RUNTIME_LIBDIR}"
sudo cp -af /var/lib/tcpx/lib64/libnccl.so* \
            /var/lib/tcpx/lib64/libnccl-net.so \
            "${TCPX_RUNTIME_LIBDIR}/"
```

## Runtime env (TCPX)

```bash
# CUDA paths
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/tcpx/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/lib:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_MODULE_LOADING=EAGER

# NCCL TCPX bindings
export NCCL_GPUDIRECTTCPX_CTRL_DEV="eth0"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=8388608
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="eth1,eth2,eth3,eth4"
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=/run/tcpx
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0

# NCCL general config
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8
export NCCL_DEBUG=WARN # INFO
export NCCL_DEBUG_SUBSYS=ENV

# Nixl plugin config
export NIXL_PLUGIN_DIR=$UCCL_HOME/thirdparty/nixl/build/src/plugins/uccl
export UCCL_P2P_TRANSPORT=tcpx
# export NIXL_LOG_LEVEL=debug
```

## Build NIXL + TCPX plugin

```bash
cd $UCCL_HOME/thirdparty/
git clone https://github.com/ai-dynamo/nixl.git
cd nixl

meson setup build # nixl auto-detect the backend's library.
cd build
ninja
ninja install
sudo ldconfig

pip uninstall -y nixl
python -m pip install --no-cache-dir "nixl[cu12]"
```


## Run benchmark

```bash
cd $UCCL_HOME/p2p

# Server (node A)
python benchmarks/benchmark_nixl.py --backend uccl --role server --sizes 67108864 --iters 10 --op-type read

# Client (node B, set server IP)
python benchmarks/benchmark_nixl.py --backend uccl --role client --sizes 67108864 --iters 10 --remote-ip=10.65.27.236 --op-type read

# 8 GPUs Server (node A)
python benchmarks/benchmark_nixl_8gpu.py --role server

# 8 GPUs Client (node B, set server IP)
python benchmarks/benchmark_nixl_8gpu.py --role client --server-ip 10.65.27.236
```

## Performance

- **Hardware**: 2 nodes, 8x H100 GPUs, 4x gVNIC
- **Bandwidth**: ~18 GB/s per gVNIC
