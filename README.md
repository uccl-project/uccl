<div align="center">

# UCCL

<p align="center">
    <a href="#about"><b>About</b></a> | 
    <a href="#road-map"><b>Road Map</b></a> | 
    <a href="#getting-started"><b>Getting Started</b></a> | 
    <a href="#documentation"><b>Documentation</b></a> | 
    <a href="#acknowledgement"><b>Acknowledgement</b></a> |
    <a href="#contact"><b>Contact</b></a>
</p>

</div>

## About 

UCCL is an efficient collective communication library for GPUs. 
UCCL aims to 
* rearchitect the CCL layer (while keeping NCCL APIs) to unleash the full potential of network hardware
* rearchitect the network transport layer to be fast and extensible
* support heterogeneous GPU and networking vendors such as Nvidia, AMD, and Broadcom
* become an open and collaborative platform for GPU communication research

UCCL has built a fast and extensible transport layer in software, with code released in this repo. 
This transport layer has created many benefits. 
For example, existing network transports under NCCL (i.e., kernel TCP and RDMA) leverage one or few network paths to stream huge data volumes, thus prone to congestion happening in datacenter networks. 
Instead, UCCL employs packet spraying in software to leverage abundant network paths to avoid "single-path-of-congestion". 
More benefits include: 1) packet spraying with 256 paths, 2) advanced congestion control such as latency-based and receiver-driven ones, 3) efficient loss recovery by selective repeat, and 4) widely usable in public clouds with legacy NICs and Ethernet. 

On two AWS `g4dn.8xlarge` instances with 1x50G ENA NICs and 1xT4 GPUs under the same cluster placement group, UCCL outperforms NCCL by up to **3.7x** for AllReduce: 

<p align="center"> <img src="./doc/images/allreduce_2_g4dn.png" alt="" width="700"> </p>

On four AWS `p4d.24xlarge` instances with 4x100G EFA NICs and 8xA100 GPUs, UCCL outperforms NCCL by up to **3.3x** for AlltoAll: 

<p align="center"> <img src="./doc/images/alltoall_4_p4d.png" alt="" width="700"> </p>

On six HGX servers (across two racks) with 8x400G CX-7 RoCE NICs and 8xH100 GPUs, UCCL outperforms NCCL by up to **2.5x** for AllReduce:

<p align="center"> <img src="./doc/images/allreduce_6_hgx.png" alt="" width="700"> </p>

Feel free to check out our full [technical report](https://arxiv.org/pdf/2504.17307) and [slides](https://drive.google.com/file/d/1YsgMNPeCV797sYPiCWAT0AMfc0WgIhP0/view?usp=sharing).

## Road Map

More UCCL features are under development in this repo, currently including: 
- [ ] Dynamic membership with GPU servers joining and exiting
- [ ] More efficient KV cache transfer engine (e.g., better Mooncake)
- [ ] Generic and SM-free GPU-initiated P2P (e.g., better DeepEP for MoE)
  - [ ] Supporting all NIC vendors including Nvidia, AWS EFA, and Broadcom
  - [ ] Avoiding burning precious GPU SMs
- [ ] Re-architecting NCCL to unleash network hardware performance
  - [ ] Scalable and efficient CPU proxy
  - [ ] Fast async collectives with compute-communication ordering guarantee
  - [ ] Device kernels in vendor-agnostic Triton language


## Getting Started

UCCL provides a drop-in replacement for any NCCL/RCCL application without code modification or compilation. 

To get started, let's first clone the UCCL repo and init submodules. 
```shell
git clone https://github.com/uccl-project/uccl.git --recursive
export UCCL_HOME=$(pwd)/uccl
```

Then install some common dependencies: 
```shell
sudo apt update
sudo apt install linux-tools-$(uname -r) clang llvm cmake m4 build-essential \
                 net-tools libgoogle-glog-dev libgtest-dev libgtest-dev \
                 libelf-dev libpcap-dev libc6-dev-i386 \
                 libopenmpi-dev libibverbs-dev libpci-dev -y

# Install and activate Anaconda (you can choose any recent versions)
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash ./Anaconda3-2024.10-1-Linux-x86_64.sh -b
source ~/anaconda3/bin/activate
source ~/.bashrc
conda init

# Install python ssh lib into conda-default base env
conda install paramiko -y
```

Next, you can dive into individual folders for various supports: 
* [`afxdp/`](./afxdp/README.md): Non-RDMA NICs (currently support AWS ENA NICs and IBM VirtIO NICs)
* [`efa/`](./efa/README.md): AWS EFA NIC (currently support p4d.24xlarge)
* [`rdma/`](./rdma/README.md): Nvidia/AMD GPUs + IB/RoCE RDMA NICs (currently support Nvidia NICs)

## Documentation

Please refer to [doc/README.md](./doc/README.md) for full documentation.

## Citation
The code in this repository is mostly described in the paper below. Please consider citing this work if you find the repository helpful. 

```bibtex
@article{zhou2025extensible,
  title={An Extensible Software Transport Layer for GPU Networking},
  author={Zhou, Yang and Chen, Zhongjie and Mao, Ziming and Lao, ChonLam and Yang, Shuo and Kannan, Pravein Govindan and Gao, Jiaqi and Zhao, Yilong and Wu, Yongji and You, Kaichao and others},
  journal={arXiv preprint arXiv:2504.17307},
  year={2025}
}
```

## Acknowledgement

UCCL is being actively developed at [UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/). We welcome open-source developers. 

## Contact
Feel free to raise issues or contact us if you have any questions or suggestions. You can reach us at: 
* Yang Zhou (yangzhou.rpc@gmail.com)
* Zhongjie Chen (chenzhjthu@gmail.com)
* Ziming Mao (ziming.mao@berkeley.edu)
