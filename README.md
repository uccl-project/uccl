<div align="center">

<p align="center"> <img src="./docs/images/uccl_logo.png" alt="" width="300"> </p>

[![🌐 UCCL](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://uccl-project.github.io/) [![Github](https://img.shields.io/badge/UCCL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/uccl-project/uccl) [![Twitter](https://img.shields.io/badge/UCCL-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/uccl_proj)
<p align="center">
    <a href="#about"><b>About</b></a> | 
    <a href="#road-map"><b>Road Map</b></a> | 
    <a href="#quick-start"><b>Quick Start</b></a> | 
    <a href="#dev-guide"><b>Dev Guide</b></a> | 
    <a href="#acknowledgement"><b>Acknowledgement</b></a> |
    <a href="#contact"><b>Contact</b></a>
</p>

</div>

## About 

UCCL is an efficient communication library for GPUs, covering collectives, P2P (e.g., KV cache transfer, RL weight transfer), and EP (e.g., IBGDA), with two key focuses: 
* **Flexibility** for high performance in fast-evolving ML workloads
* **Portability** for connecting heterogeneous GPUs in ML workloads

An UCCL overview can be found in this [slide deck](https://docs.google.com/presentation/d/1LQxZzxghRmua4FkfQjWu69wXy9hrs9V_tXrXt_DT-F4/edit?usp=sharing) with the following components: 

* **[UCCL-collective](collective/)** serves as a drop-in replacement for NCCL/RCCL (e.g., requiring no changes to application code), and significantly outperforms them in both latency and throughput across various settings. 

  <details>
  <summary>UCCL-collective performance comparison</summary>

  * On six HGX servers (across two racks) with 8x400G CX-7 RoCE NICs and 8xH100 GPUs, UCCL-collective outperforms NCCL by up to **2.5x** for AllReduce:
    <p align="left"> <img src="./docs/images/allreduce_6_hgx.png" alt="" width="600"> </p>

  * On two AWS `g4dn.8xlarge` instances with 1x50G ENA NICs and 1xT4 GPUs within the same cluster placement group, UCCL-collective outperforms NCCL by up to **3.7x** for AllReduce: 
    <p align="left"> <img src="./docs/images/allreduce_2_g4dn.png" alt="" width="600"> </p>

  More specifically, UCCL-collective aims to: 
  * rearchitect the CCL layer (while keeping NCCL APIs) to unleash the full potential of network hardware
  * rearchitect the network transport layer to be fast and extensible
  * support heterogeneous GPU and networking vendors such as Nvidia, AMD, and Broadcom
  * become an open and collaborative platform for GPU communication research
  <br>

  UCCL-collective has built a fast and extensible transport layer in software, which has created many benefits. 
  For example, existing network transports under NCCL (i.e., kernel TCP and RDMA) leverage one or few network paths to stream huge data volumes, thus prone to congestion happening in datacenter networks. 
  Instead, UCCL-collective employs packet spraying in software to leverage abundant network paths to avoid "single-path-of-congestion". 
  More benefits include: 1) packet spraying with 256 paths, 2) advanced congestion control such as latency-based and receiver-driven ones, 3) efficient loss recovery by selective repeat, and 4) widely usable in public clouds with legacy NICs and Ethernet. Feel free to check out our full [technical report](https://arxiv.org/pdf/2504.17307).
  </details>

* **[UCCL-P2P](p2p/)** provides both NIXL-style initiator-target tranfer APIs and NCCL-style collective APIs, with the same or better performance than both. UCCL-P2P is purposely designed for the next-gen 800Gbps NICs with efficient multi-threaded transfer engines. 

* **[UCCL-EP](ep/)** allows running DeepEP atop of heterogeneous hardware platforms, including AMD and Nvidia GPUs, and any RDMA NICs such as AWS EFA NICs and Broadcom NICs, while achieving IBGDA-level performance. UCCL-EP also makes DeepEP SM-free, devoting all GPU SMs to compute. 

UCCL has been adopted as part of the AMD [TheRock](https://github.com/ROCm/TheRock) ecosystem.

## Road Map

More UCCL features are under development in this repo, currently including: 
- ✅ More efficient KV cache transfer engine (e.g., better Mooncake)
- 🚧 Generic and SM-free GPU-initiated P2P (e.g., better DeepEP for MoE)
  - 🚧 Supporting all NIC vendors including Nvidia, AWS EFA, and Broadcom
  - 🚧 Avoiding burning precious GPU SMs
- 🚧 Re-architecting NCCL to unleash network hardware performance
  - 🚧 Scalable and efficient CPU proxy
  - ☐ Fast async collectives with compute-communication ordering guarantee
  - ☐ Device kernels in vendor-agnostic Triton language
- ☐ Dynamic membership with GPU servers joining and exiting


## Quick Start

The easiest way to use UCCL is to first build based on your platform. The build script will automatically detect the `py_version` of your current environment. If you need to compile UCCL for a specific python version, please specify the `py_version`, such as `3.10`. 

```bash
git clone https://github.com/uccl-project/uccl.git --recursive
cd uccl && bash build_and_install.sh [cuda|rocm|therock] [all|ccl_rdma|ccl_p2p|efa|ep] [py_version] [rocm_index_url]
```
> Note: 
> - when building for ROCm with python packaging through TheRock, please specify your ROCm index url; the default is `https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu` and it may not be what you want. When installing UCCL wheels for TheRock, please provide pip with the index url and add the optional extra `[rocm]` to the wheel, e.g., `pip install --extra-index-url https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu wheelhouse-therock/uccl-0.0.1.post4-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl[rocm]`.
> - you can build with different CUDA or ROCm versions by specifying tags such as cuda13 or rocm6. The default versions are CUDA 12.x for the "cuda" tag and ROCm 7.x for the "rocm" tag.

Then, when running your PyTorch applications, set the environment variable accordingly: 
```bash
# NCCL over IB/RoCE on x86 or GH200 ARM hosts
NCCL_NET_PLUGIN=`python -c "import uccl; print(uccl.nccl_plugin_path())"`

# RCCL over IB/RoCE on x86 hosts
NCCL_NET_PLUGIN=`python -c "import uccl; print(uccl.rccl_plugin_path())"`

# NCCL over AWS EFA NICs (p4d and p4de only)
LD_PRELOAD=`python -c "import uccl; print(uccl.efa_nccl_path())"`
NCCL_NET_PLUGIN=`python -c "import uccl; print(uccl.efa_plugin_path())"`
```

Now, you can just run your PyTorch applications and enjoy UCCL performance benefits! 

## Dev Guide

Please refer to [docs/README.md](docs/README.md) for full development guide of UCCL.

## Citation
The code in this repository is mostly described in the paper below. Please consider citing this work if you find the repository helpful. 

```bibtex
@article{uccl_transport,
  title={An Extensible Software Transport Layer for GPU Networking},
  author={Zhou, Yang and Chen, Zhongjie and Mao, Ziming and Lao, ChonLam and Yang, Shuo and Kannan, Pravein Govindan and Gao, Jiaqi and Zhao, Yilong and Wu, Yongji and You, Kaichao and Ren, Fengyuan and Xu, Zhiying and Raiciu, Costin and Stoica, Ion},
  journal={arXiv preprint arXiv:2504.17307},
  year={2025}
}
```

## Acknowledgement

UCCL is being actively developed at [UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/) and [UC Davis ArtSy lab](https://github.com/artsy-lab). We enthusiastically welcome open-source developers joining us! 

UCCL is generously supported by (in alphabetical order): 
[AMD](https://www.amd.com/en.html), 
[AWS](https://aws.amazon.com/), 
[Broadcom](https://www.broadcom.com/), 
[CloudLab](https://www.cloudlab.us/), 
[Google Cloud](https://cloud.google.com/), 
[IBM](https://www.ibm.com/), 
[Lambda](https://lambda.ai/),
[Mibura](https://www.mibura.com/).

## Contact
Feel free to raise GitHub issues if you have any questions or suggestions. 
