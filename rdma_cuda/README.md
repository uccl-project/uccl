# UCCL-RDMA-NCCL

UCCL RDMA plugin for NCCL.

1. UCCL supports two network fabrics: RoCE, Infiniband.
2. UCCL supports two modes: Unreliable Connection (UC) and Reliable Connection (RC).

## Configuration
### transport_config.h:
Modify the below constants based on the environment.

1. Network
```
ROCE_NET:               true (RoCE) or false (Infiniband)

SINGLE_CTRL_NIC:        The device name of control NIC. Set to empty string if each RDMA NIC has its own IP address. UCCL will detect them automatically.
```

2. NIC
```
NUM_DEVICES:            The number of physical NICs (use ibv_devices).

IB_DEVICE_NAME_PREFIX:  The prefix of the device name (e.g. mlx5_).

DEVNAME_SUFFIX_LIST:    The suffix of the device name (use ibv_devices).

LINK_BANDWIDTH:         The bandwidth (bytes per second) of each NIC (use ibstat).
```

### run_nccl_test.sh:
```
ROOT:                   The root directory of the workspace.
NODES:                  IP list of all nodes.
CTRL_NIC:               The name of control NIC.
NCCL_PATH:              The path of libnccl.so
PLUGIN_PATH:            The path of libnccl-net.so

Usage: ./run_nccl_test.sh [NCCL/UCCL: 0/1, default:1] [# of Nodes, default:2] [# of GPUs per node, default:8] [allreduce/alltoall: 0/1]
```

## Building and running UCCL

Build `nccl` and `nccl-tests`: 

```shell
# Eg, /home/yangz/uccl
export UCCL_HOME=<the absolute path of uccl>

# Build nccl (taking ~3min); assume H100 GPUs
cd $UCCL_HOME/thirdparty/nccl
make src.build -j NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
cp src/include/nccl_common.h build/include/

# Build nccl-tests; consider "conda deactivate" when hitting dependency errors
cd $UCCL_HOME/thirdparty/nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$UCCL_HOME/thirdparty/nccl-sg/build -j
```

Build `libnccl-net.so`

```shell
cd $UCCL_HOME/rdma_cuda
make -j
```

Running `nccl-tests`:

```shell
./run_nccl_test.sh 1 2 8 1
```
