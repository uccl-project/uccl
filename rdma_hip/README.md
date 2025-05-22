# UCCL-RDMA-RCCL

UCCL-RDMA plugin for RCCL.

1. UCCL supports two network fabrics: RoCE, Infiniband.
2. UCCL supports two modes: Unreliable Connection (UC) and Reliable Connection (RC).

This guide assumes under the [AMD HPC Fund cluster](https://amdresearch.github.io/hpcfund/hardware.html), without any root access. 

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

## Prepare dependency
Install and activate recent Anaconda to prepare necessary libraries such as `-lglog -lgflags -lgtest`. Consider installing it into `$WORK` directory as Anaconda is large. Then: 
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

## Building and running UCCL

Build `rccl` and `rccl-tests`: 

```shell
# Eg, /home1/yangzhou/uccl
export UCCL_HOME=<the absolute path of uccl>
# Eg, /work1/yzhou/yangzhou/anaconda3/lib
export CONDA_LIB_HOME=<the absolute path of anaconda lib>

# Avoiding gfx950 as the HPC Fund cluster clang does not support it yet. Note this takes ~20min. 
cd $UCCL_HOME/thirdparty/rccl
./install.sh --amdgpu_targets="gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" -j 16

cd $UCCL_HOME/thirdparty/rccl-tests
make MPI=1 MPI_HOME=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5 HIP_HOME=/opt/rocm-6.3.1 NCCL_HOME=/opt/rocm-6.3.1/include/rccl CUSTOM_RCCL_LIB=/opt/rocm-6.3.1/lib/librccl.so -j
```

Build `librccl-net-uccl.so`

```shell
cd $UCCL_HOME/rdma_hip
make -j
```

Running `rccl-tests`:

```shell
# Using slurm to allocate two AMD nodes
salloc -N 2 -n 2 -p mi2104x -t 00:30:00

# Usage: ./run_rccl_test.sh [rccl/uccl, default: uccl]
./run_rccl_test.sh rccl
```
