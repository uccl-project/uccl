# UCCL Dev Guide

To get started, let's first clone the UCCL repo and init submodules. 
```bash
git clone https://github.com/uccl-project/uccl.git --recursive
export UCCL_HOME=$(pwd)/uccl
```

Then install some common dependencies: 
```bash
sudo apt update
sudo apt install linux-tools-$(uname -r) clang llvm cmake m4 build-essential \
                 net-tools libgoogle-glog-dev libgtest-dev libgflags-dev \
                 libelf-dev libpcap-dev libc6-dev-i386 \
                 libopenmpi-dev libibverbs-dev libpci-dev -y

# Install and activate Anaconda (you can choose any recent versions)
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash ./Anaconda3-2024.10-1-Linux-x86_64.sh -b
source ~/anaconda3/bin/activate
source ~/.bashrc # or .zshrc and others
conda init

# Install python ssh lib into conda-default base env
conda install paramiko -y
```

You can then dive into: 
* [`dev_rdma.md`](dev_rdma.md): Nvidia/AMD GPUs + IB/RoCE RDMA NICs (currently support Nvidia and Broadcom NICs)
* [`dev_efa.md`](dev_efa.md): AWS EFA NIC (currently support p4d.24xlarge)
* [`dev_afxdp.md`](dev_afxdp.md): Non-RDMA NICs (currently support AWS ENA NICs and IBM VirtIO NICs)

### Python Wheel Build

Run the following to build Python wheels (you can replace `all` with `cuda`, `rocm`, `efa`, and more): 
```bash
cd $UCCL_HOME
./docker_build.sh all
```

Run the following to install the wheels locally: 
```bash
cd $UCCL_HOME
pip install wheelhouse-all/uccl-*.whl
```

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