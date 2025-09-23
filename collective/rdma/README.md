# UCCL RDMA NCCL/RCCL

UCCL RDMA plugin for NCCL/RCCL.

1. UCCL supports two network fabrics: RoCE, Infiniband.
2. UCCL supports two modes: Unreliable Connection (UC) and Reliable Connection (RC).
3. UCCL supports both Nvidia and AMD GPUs.

## Configuration
### Env Vars
By default, UCCL auto-detects all necessary parameters, similar to NCCL/RCCL. 
But you can overwrite these parameters, eg:
```
UCCL_IB_HCA:          The names of IB devices you want to use.
UCCL_SOCKET_IFNAME:   The control NIC name you want to use for bootstrapping.
```

For convenience, the NCCL versions, i.e., NCCL_IB_HCA and NCCL_SOCKET_IFNAME, are also available.

### run_nccl_test.sh:
```
HOSTFILE:               The MPI host file (e.g., ${UCCL_HOME}/scripts/node_ips/default.txt)

HCA_NAMES:              For UCCL_IB_HCA.
CTRL_NIC:               For UCCL_SOCKET_IFNAME.

Usage: ./run_nccl_test.sh [NCCL/UCCL: 0/1, default:1] [# of total processes, default:2] [# of GPUs per process, default:8] [allreduce/alltoall: 0/1] [# of processes per node, default:1]
```

### run_rccl_test.sh: 
```
HOSTFILE:               The MPI host file

HCA_NAMES:              For UCCL_IB_HCA.
CTRL_NIC:               For UCCL_SOCKET_IFNAME.

Usage: ./run_rccl_test.sh [RCCL/UCCLL: rccl/uccl, default:uccl]
```

## Building and running UCCL for Nvidia GPUs

### Build `nccl` and `nccl-tests`: 

```bash
export UCCL_HOME=<the path of uccl> # Eg, /home/yangz/uccl

# Build nccl ~3min; assume H100 GPUs
cd $UCCL_HOME/thirdparty/nccl
make src.build -j NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
cp src/include/nccl_common.h build/include/

# Build nccl-tests; consider "conda deactivate" when hitting dependency errors
cd $UCCL_HOME/thirdparty/nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$UCCL_HOME/thirdparty/nccl/build -j
```

### Build `libnccl-net-uccl.so`

The easiest way is to use docker, which packs all needed external libraries into a python wheel and install into your local python env: 
```bash
cd $UCCL_HOME && bash build_and_install.sh cuda rdma
```

The following alternative is best for development where you have installed all needed external libraries: 
<details><summary>Click me</summary>

```bash
cd $UCCL_HOME/rdma
make -j
```
</details>

### Running `nccl-tests`:

```bash
cd $UCCL_HOME/scripts
python rsync.py -n node_ips/h100_6.txt

cd $UCCL_HOME/rdma

# 1) UCCL, 2 processes, 8 GPUs per process, alltoall, 1 process per node
./run_nccl_test.sh 1 2 8 1 1
# 2) UCCL, 16 processes, 1 GPU per process, alltoall, 8 processes per node
./run_nccl_test.sh 1 16 1 1 8
```


## Building and running UCCL for AMD GPUs

This guide assumes under the [AMD HPC Fund cluster](https://amdresearch.github.io/hpcfund/hardware.html), without any root access. 

### Build `rccl` and `rccl-tests`: 

```bash
export UCCL_HOME="/home1/yangzhou/uccl"
export CONDA_LIB_HOME="/work1/yzhou/yangzhou/anaconda3/lib"

# Avoiding gfx950 as the HPC Fund cluster clang does not support it yet. Note this takes ~20min. 
cd $UCCL_HOME/thirdparty/rccl
./install.sh --amdgpu_targets="gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" --disable-mscclpp -j 16

cd $UCCL_HOME/thirdparty/rccl-tests
make MPI=1 MPI_HOME=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5 HIP_HOME=/opt/rocm-6.3.1 NCCL_HOME=/opt/rocm-6.3.1/include/rccl CUSTOM_RCCL_LIB=/opt/rocm-6.3.1/lib/librccl.so -j
```

### Build `librccl-net-uccl.so`

The easiest way is to use docker: 
```bash
cd $UCCL_HOME && bash build_and_install.sh rocm rdma
```

The following alternative is best for development where you have installed all needed external libraries:
<details><summary>Click me</summary>

Install and activate recent Anaconda to prepare necessary libraries. Consider installing it into `$WORK` directory as Anaconda is large. 

Inside the conda env, install libs that contains libglog, libgflags, and libgtest: 
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Then build: 
```bash
cd $UCCL_HOME/rdma
make -f Makefile.rocm -j
```
</details>

### Run `rccl-tests`

```bash
# Using slurm to allocate two AMD nodes
salloc -N 2 -n 2 -p mi2104x -t 00:30:00

# Usage: ./run_rccl_test_hpcfund.sh [rccl/uccl, default: uccl]
./run_rccl_test_hpcfund.sh rccl
```

### For Broadcom NICs

Using the following to run `rccl-tests`:

```bash
# Edit ${UCCL_HOME}/scripts/node_ips/amd.txt to fill up the node addresses. 

# Usage: ./run_rccl_test.sh [rccl/uccl, default: uccl]
./run_rccl_test.sh rccl
```

## Environment Variables in UCCL

UCCL supports the following environment variables to configure.

For example, one can enlarge the chunk size to 128KB by setting `UCCL_CHUNK_SIZE_KB=128`. 

Use `UCCL_PARAM()` to introduce new environment variables.

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| UCCL_IB_HCA | The names of IB devices used | (null) |
| UCCL_SOCKET_IFNAME | The control NIC name used for bootstrapping | (null) |
| UCCL_PIN_TO_NUMA | Pin threads to the NUMA node | 1 |
| UCCL_ROCE_TRAFFIC_CLASS | Traffic class for RoCE | 3 |
| UCCL_ROCE_SERVICE_LEVEL | Service level for RoCE | 135 |
| UCCL_IB_SERVICE_LEVEL | Service level for IB | 0 |
| UCCL_RCMODE | Use RC for data transfer | 1 (Broadcom NIC), 0 (others) |
| UCCL_BYPASS_PACING | Bypass the pacing stage | true |
| UCCL_NUM_ENGINES | Number of engines per device | 4 (NVIDIA), 1 (AMD) |
| UCCL_PORT_ENTROPY | Path/QP per engine | 32 (NVIDIA), 256 (AMD) |
| UCCL_CHUNK_SIZE_KB | Maximum chunk size for each WQE | 64 (NVIDIA), 128 (AMD) |
