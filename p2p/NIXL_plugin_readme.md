# NIXL with TCPX Backend

Quick setup and run guide for the TCPX backend.

---

## Prerequisites

```bash
export UCCL_HOME=/path/to/your/uccl

# Install dependencies
sudo apt update
sudo apt install -y build-essential net-tools libelf-dev libibverbs-dev \
                    libgoogle-glog-dev libgtest-dev libgflags-dev \
                    python3-dev pybind11-dev python3-pip python3-pybind11

cd $UCCL_HOME/thirdparty/nccl
git checkout v2.18.5-1
make src.build -j NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

cd $UCCL_HOME/thirdparty/nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$UCCL_HOME/thirdparty/nccl/build -j
```

## Build UCCL P2P engine

```bash
cd $UCCL_HOME/p2p
make clean
make USE_TCPX=1 -j
sudo install -m 0755 libuccl_engine.so /usr/local/lib/
sudo ldconfig
sudo make USE_TCPX=1 install
```

## Environment variables

```bash
# CUDA paths
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/var/lib/tcpx/lib64:$LD_LIBRARY_PATH"

# NCCL TCPX bindings
export NCCL_GPUDIRECTTCPX_CTRL_DEV="eth0"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP=100

# NCCL general config
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ENV
```

## Build NIXL + TCPX plugin

```bash
cd $UCCL_HOME/thirdparty/
git clone https://github.com/uccl-project/nixl.git
cd $UCCL_HOME/thirdparty/nixl
git checkout pr-895-tcpx
pip uninstall -y nixl
meson setup build -Ddisable_tcpx_backend=false
ninja -C build src/plugins/tcpx/libplugin_TCPX.so
pip install . --no-cache-dir --force-reinstall
export LD_LIBRARY_PATH=$(pwd)/build/src/core:$(pwd)/build/src:$(pwd)/build/src/infra:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=$(pwd)/build/src/plugins/tcpx
export NIXL_LOG_LEVEL=debug
```

### (Optional) Remove legacy system NIXL

```bash
sudo mv /usr/local/nixl /usr/local/nixl.bak
sudo ldconfig
```

```bash
cd $UCCL_HOME/thirdparty/nixl
pip uninstall -y nixl
pip install . --no-cache-dir --force-reinstall
```

## Run benchmark

```bash
cd $UCCL_HOME/p2p

# Server (node 1)
python benchmarks/benchmark_nixl.py --backend tcpx --role server --sizes 67108864 --iters 10

# Client (node 2, replace with actual IP)
python benchmarks/benchmark_nixl.py --backend tcpx --role client --sizes 67108864 --iters 10 --remote-ip=<server_ip>
```

## Performance

- **Hardware**: 2 nodes, 8x H100 GPUs, 4x gVNIC
- **Bandwidth**: ~18 GB/s per GPU
