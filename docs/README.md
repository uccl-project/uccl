# UCCL Dev Guide

First clone the UCCL repo and init submodules: 
```bash
git clone https://github.com/uccl-project/uccl.git --recursive
export UCCL_HOME=$(pwd)/uccl
```

To build UCCL for development, you need to install some common dependencies: 
<details><summary>Click me</summary>

```bash
# Note if you are using docker+wheel build, there is no need to install the following dependencies. 
sudo apt update
sudo apt install linux-tools-$(uname -r) clang llvm cmake m4 build-essential \
                 net-tools libgoogle-glog-dev libgtest-dev libgflags-dev \
                 libelf-dev libpcap-dev libc6-dev-i386 libpci-dev \
                 libopenmpi-dev libibverbs-dev clang-format -y

# Install and activate Anaconda (you can choose any recent versions)
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
bash ./Anaconda3-2025.06-1-Linux-x86_64.sh -b
source ~/anaconda3/bin/activate
source ~/.bashrc # or .zshrc and others
conda init

# Install python ssh lib into conda-default base env
pip install paramiko pybind11
```
</details>

For quick installation with docker, you can directly dive into: 
* [`UCCL-Collective RDMA`](../collective/rdma/README.md): Collectives for Nvidia/AMD GPUs + IB/RoCE RDMA NICs (currently support Nvidia and Broadcom NICs)
* [`UCCL-Collective EFA`](../collective/efa/README.md): Collectives for AWS EFA NIC (currently support p4d.24xlarge)
* [`UCCL-Collective AFXDP`](../collective/afxdp/README.md): Collectives for Non-RDMA NICs (currently support AWS ENA NICs and IBM VirtIO NICs)
* [`UCCL-P2P`](../p2p/README.md): P2P for RDMA NICs and GPU IPCs (currently support Nvidia/AMD GPUs and Nvidia/Broadcom NICs)

### Python Wheel Build

Run the following to build Python wheels: 
```bash
cd $UCCL_HOME
./build.sh [cuda|rocm]
```

Run the following to install the wheels locally: 
```bash
cd $UCCL_HOME
pip install wheelhouse-[cuda/rocm]/uccl-*.whl
```

The cross-compilation matrix is as follows:

| Platform/Feature   | rdma-cuda | rdma-rocm | rdma-arm | p2p-cuda | p2p-rocm | p2p-arm | efa |
|--------------------|-----------|-----------|----------|----------|----------|---------|-----|
| cuda + x86         | ✓         | ✓         | x        | ✓        | ✓        | x       | ✓   |
| cuda + arm (gh200) | ✓         | x         | x        | ✓        | x        | x       | x   |
| rocm + x86         | ✓         | ✓         | ✓        | ✓        | ✓        | ✓       | x   |
| aws p4d/p4de       | ✓         | ✓         | x        | ✓        | x        | x       | ✓   |

Note that you need ARM hosts to build ARM wheels, as cross-compilation tool `qemu-user-static` cannot emulate CUDA or ROCm. 


### On Cloudlab CPU Machines

If you want to build nccl and nccl-tests on cloudlab ubuntu22, you need to install cuda and openmpi: 

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit -y
sudo apt install nvidia-driver-550 nvidia-utils-550 -y

sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev -y
```