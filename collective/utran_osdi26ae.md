# UCCL-Tran (OSDI '26) — Artifact Evaluation README

This document is the artifact-evaluation README for the OSDI '26 paper:

> **UCCL-Tran: An Extensible Software Transport Layer for GPU Networking**
> Yang Zhou, Zhongjie Chen, Ziming Mao, ChonLam Lao, Shuo Yang, Pravein Govindan Kannan, Jiaqi Gao, Yilong Zhao, Yongji Wu, Kaichao You, Fengyuan Ren, Zhiying Xu, Costin Raiciu, Ion Stoica.
> *USENIX OSDI 2026.*

We are applying for the **Artifacts Available** badge. The full source code, build scripts, run scripts, and per-figure reproduction recipes are all hosted permanently and publicly in this repository: <https://github.com/uccl-project/uccl>.

---

## Important notice to reviewers (testbed accessibility)

The evaluation in the paper uses four GPU + RDMA testbeds. **Unfortunately, due to the long period between the original experiment evaluation date and now, and the high expenses of RDMA-connected GPU testbeds, we are no longer able to access any of the testbeds used in the paper.** As a result, we cannot offer reviewers SSH access to a pre-configured cluster.

To make reproduction possible, we provide:

1. The **exact testbed specifications** used in the paper (Table 1, reproduced below) so reviewers can rent equivalent instances from public clouds.
2. **Per-figure / per-table** pointers to the build and run scripts in this repository, and the exact knobs (cluster size, message size, chunk size, engines, etc.) that need to be swept.
3. Step-by-step, modality-specific build instructions for RoCE/IB, AWS EFA, and AF\_XDP NICs.

We very much appreciate the AEC's understanding given the cost of multi-rack 400 G-class RDMA fleets.

### Testbeds used in the paper

| Name    | # of Servers | Network    | Topology                  | MTU | NIC                            | GPU                       | CPU       | Where to rent (suggested)                              |
| ------- | ------------ | ---------- | ------------------------- | --- | ------------------------------ | ------------------------- | --------- | ------------------------------------------------------ |
| CX_ETH  | 6            | Ethernet   | Cross racks, fat-tree     | 4KB | NVIDIA ConnectX-7 400G x 8     | NVIDIA H100-80G x 8       | 160 cores | TensorWave / Lambda / CoreWeave / Crusoe (H100 + RoCE) |
| AMD     | 4            | Ethernet   | Cross racks, rail-optimized | 4KB | Broadcom Thor-2 400G x 8       | AMD MI300X-192G x 8       | 128 cores | IBM Cloud `gx3d-160x1792x8mi300x` instances            |
| EFA     | 4            | Ethernet   | Cross racks, fat-tree     | 9KB | AWS EFA 100G x 4               | NVIDIA A100-40G x 8       | 96  cores | AWS `p4d.24xlarge` (us-east-1, capacity reservation)   |
| CX_IB   | 2            | InfiniBand | Same rack                 | 4KB | NVIDIA ConnectX-7 400G x 8     | NVIDIA H100-80G x 8       | 128 cores | A bare-metal H100 + IB host (e.g., CoreWeave / Lambda) |

A representative subset of results (in particular Figures 7, 9, 11(b), and 14) can also be reproduced on **smaller clusters** of the same GPU/NIC family — the absolute numbers will differ but the qualitative trends should hold.

---

## Repository layout

The artifact lives in <https://github.com/uccl-project/uccl>. The directories most relevant for OSDI '26 reproduction are:

| Path                                       | What it contains                                                                          |
| ------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `collective/rdma/`                         | UCCL-Tran NCCL/RCCL plugin for RoCE & InfiniBand NICs (CX_ETH, AMD, CX_IB testbeds)       |
| `collective/efa/`                          | UCCL-Tran NCCL plugin for AWS EFA NICs (EFA testbed)                                      |
| `collective/afxdp/`                        | UCCL-Tran AF\_XDP backend for legacy non-RDMA NICs (additional reference path)            |
| `collective/rdma/incast/`                  | Incast micro-benchmark (Figure 12)                                                        |
| `experimental/misc/`                       | DeepEP-style benchmarks used for Figure 11(b)                                             |
| `scripts/`                                 | Hostfile templates and `rsync.py` helper to fan out builds / binaries                     |
| `build.sh`                                 | One-shot Docker-based build that produces a `pip`-installable `uccl` wheel                |

---

## Getting Started Instructions (≈ 30 minutes)

The goal of this section is to let an evaluator confirm, on a single small RDMA-capable host pair, that the artifact builds, installs, and runs end-to-end. This is the "Hello, world" sanity check.

We recommend two AWS `p4d.24xlarge` instances (the **EFA** testbed). If RDMA hardware is not available, even one node alone is enough to confirm that the wheel builds and that `import uccl` exposes the plugin paths.

### Step 1 — Clone and initialize submodules

```bash
git clone https://github.com/uccl-project/uccl.git
cd uccl
export UCCL_HOME=$(pwd)
git submodule update --init thirdparty/nccl thirdparty/nccl-tests
```

### Step 2 — Build & install UCCL via Docker (recommended)

This produces and `pip install`s a `uccl` Python wheel containing the plugin shared libraries:

```bash
# RoCE / IB hosts (CX_ETH, CX_IB):
bash build.sh cu12 ccl_rdma --install

# AMD MI300X + Broadcom Thor-2 (AMD testbed):
bash build.sh roc7 ccl_rdma --install

# AWS p4d EFA hosts (EFA testbed):
bash build.sh cu12 ccl_efa --install
```

After installation, verify the plugin path is exposed:

```bash
python -c "import uccl; print(uccl.nccl_plugin_path())"
# Expected: an absolute path ending in libnccl-net-uccl.so (or libnccl-net-efa.so for EFA)
```

### Step 3 — Run a 30-minute "hello world" `nccl-tests` job

Pick the recipe matching your hardware. Each one launches `all_reduce_perf` over 1 KB → 1 GB messages.

* **RoCE / IB** (any 2 hosts with RDMA NICs, e.g., CX_ETH or CX_IB):
  ```bash
  cd $UCCL_HOME/scripts
  # Put the IPs of both nodes (one per line) into node_ips/h100_6.txt
  python rsync.py -n node_ips/h100_6.txt

  cd $UCCL_HOME/collective/rdma
  # Args: [nccl|uccl] [#procs] [GPUs/proc] [allreduce(0)/alltoall(1)] [procs/node]
  ./run_nccl_test.sh uccl 2 8 0 1
  ```
  Detailed walk-through: [`collective/rdma/README.md`](rdma/README.md).

* **AWS EFA** (2 × `p4d.24xlarge`):
  ```bash
  cd $UCCL_HOME/scripts
  # Fill node_ips/p4d.txt with the two p4d IPs
  python rsync.py -n node_ips/p4d.txt

  cd $UCCL_HOME/collective/efa
  ./run_nccl_test.sh ud 16   # ud = UCCL over EFA's UD; 16 GPUs total
  ```
  Detailed walk-through: [`collective/efa/README.md`](efa/README.md).

* **AMD MI300X + Broadcom Thor-2** (2 nodes):
  ```bash
  # Edit $UCCL_HOME/scripts/node_ips/amd.txt to list the two nodes.
  cd $UCCL_HOME/collective/rdma
  ./run_rccl_test.sh uccl
  ```
  Detailed walk-through: [`collective/rdma/README.md`](rdma/README.md) (sections "Building and running UCCL for AMD GPUs" and "For Broadcom NICs").

A successful run prints a standard `nccl-tests` table; the UCCL bandwidth column for the largest sizes should be on par with or above the corresponding NCCL/RCCL run (UCCL's wins grow with cluster size, so on 2 nodes the gap is smaller than the headline numbers in the paper).

If you only want to verify that the artifact is **available and builds**, Step 2 alone is sufficient.

---

## Detailed Instructions (per-figure / per-table reproduction)

Each entry below states (a) which testbed from Table 1 the result was collected on, (b) the script(s) to run, and (c) the parameters that were swept to produce the curves/bars in the paper.

### Figure 7 — Allreduce performance on CX_ETH

* **Testbed:** CX_ETH (6 × HGX H100 + 8 × CX-7 400G RoCE).
* **Script:** [`collective/rdma/run_nccl_test.sh`](rdma/run_nccl_test.sh).
* **How to sweep:**
  * Change the **cluster size** by varying the `# of total processes` argument (and the `node_ips/*.txt` hostfile passed via `HOSTFILE`).
  * The **message size sweep** is built into `nccl-tests` (`-b 1K -e 1G -f 2`). Edit the line at the bottom of `run_nccl_test.sh` if you want a different range/step.
  * Compare `./run_nccl_test.sh uccl ...` against `./run_nccl_test.sh nccl ...`.
* **Detailed setup:** [`collective/rdma/README.md`](rdma/README.md).

### Figure 8 — Allreduce performance on AMD (Broadcom Thor-2)

* **Testbed:** AMD (4 × MI300X + 8 × Broadcom Thor-2 400G).
* **Script:** [`collective/rdma/run_rccl_test.sh`](rdma/run_rccl_test.sh).
* **How to sweep:**
  * Cluster size is controlled by `np` and the `amd.txt` hostfile inside the script (`-np 4 -N 1`); edit and re-run for 2/4/8 nodes.
  * For RCCL vs. UCCL comparison: `./run_rccl_test.sh rccl` and `./run_rccl_test.sh uccl`.
* **Detailed setup:** [`collective/rdma/README.md`](rdma/README.md) (section "Building and running UCCL for AMD GPUs").

### Figure 9 — Allreduce / alltoall performance on EFA

* **Testbed:** EFA (4 × `p4d.24xlarge`, 4 × EFA 100 G per node).
* **Script:** [`collective/efa/run_nccl_test.sh`](efa/run_nccl_test.sh).
* **How to sweep:**
  * Cluster size: change the second positional arg (total #GPUs). E.g., `./run_nccl_test.sh ud 8`, `./run_nccl_test.sh ud 16`, `./run_nccl_test.sh ud 32`.
  * For NCCL baseline, replace `ud` with `srd` (the upstream EFA NCCL plugin path) or compare against the official `aws-ofi-nccl` plugin.
* **Detailed setup:** [`collective/efa/README.md`](efa/README.md).

### Figure 10 — Cross-rack allreduce on CX_ETH

* **Testbed:** CX_ETH, but with the 6 servers spread across both racks (cross-rack flows).
* **Script:** [`collective/rdma/run_nccl_test.sh`](rdma/run_nccl_test.sh).
* **How to sweep:** Same as Figure 7. Use a hostfile that places nodes on different racks, then sweep cluster size and message size.
* **Detailed setup:** [`collective/rdma/README.md`](rdma/README.md).

### Figure 11(a) — DeepSeek-V2-Lite training on AMD

* **Testbed:** AMD (4 × MI300X + 8 × Broadcom Thor-2 400G).
* **Procedure**
  * Please clone and install [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus/tree/main) by following its guidelines.
  * Please use [deepseek_v2_lite-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml) as the model config and set `tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=32`.
  * For RCCL vs. UCCL comparison, set the environment variable `NCCL_NET_PLUGIN = libnccl-net-uccl.so` to use UCCL, and leave it unset by default to use RCCL.
* **How to sweep**
  * Add `num_experts` to [deepseek_v2_lite-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml) and sweep it over 32/64/128.
* **Detailed setup:** [`collective/rdma/README.md`](rdma/README.md) (section "Building and running UCCL for AMD GPUs").

### Figure 11(b) — Expert-parallel benchmark on EFA

* **Testbed:** EFA (4 × `p4d.24xlarge`).
* **Scripts:**
  * [`experimental/misc/run_ep_benchmark_nccl.sh`](../experimental/misc/run_ep_benchmark_nccl.sh) — NCCL + official EFA plugin baseline.
  * [`experimental/misc/run_ep_benchmark_uccl.sh`](../experimental/misc/run_ep_benchmark_uccl.sh) — UCCL-Tran EFA plugin.
* **How to run:**
  ```bash
  # On all p4d nodes, ensure UCCL is built & installed (build.sh cu12 ccl_efa --install).
  # Edit $UCCL_HOME/scripts/node_ips/p4d.txt to list all nodes.
  cd $UCCL_HOME/experimental/misc
  bash run_ep_benchmark_uccl.sh   # UCCL-Tran
  bash run_ep_benchmark_nccl.sh   # NCCL baseline
  ```
* **What to compare:** dispatch / combine latency reported by `deepseek_ep.py` for the configured `--hidden-size 7168 --num-experts 256 --top-k 8`.

### Figure 12 — Incast vs. permutation interference

* **Testbed:** CX_IB (2 × HGX H100 + 8 × CX-7 400G IB, same rack).
* **Procedure:** Follow [`collective/rdma/incast/README.md`](rdma/incast/README.md) verbatim.
  ```bash
  # On a master node, with $UCCL_HOME/rdma already built:
  cd $UCCL_HOME/collective/rdma/incast
  python gen_permutation_full_bisection.py matrix.txt 16 4
  ./sync_repo.sh
  ./run.sh
  ```
* **What to vary:** number of incast senders and permutation matrix size (regenerate `matrix.txt` with different arguments).

### Figure 13 — *(reserved for co-author Zhongjie)*

> @zhongjie: please fill in the script path and exact sweep parameters for Figure 13.

### Figure 14 — Sensitivity to chunk size and engine count on CX_ETH

* **Testbed:** CX_ETH.
* **Script:** [`collective/rdma/run_nccl_test.sh`](rdma/run_nccl_test.sh).
* **How to sweep:** Add the following environment variables to the `mpirun -x ...` list inside the script, then re-run for each combination:
  ```bash
  -x UCCL_CHUNK_SIZE_KB=<8 | 16 | 32 | 64 | 128 | 256> \
  -x UCCL_NUM_ENGINES=<1 | 2 | 4 | 8>
  ```
  `UCCL_CHUNK_SIZE_KB` controls the maximum chunk size carried by each WQE; `UCCL_NUM_ENGINES` controls how many CPU engine threads UCCL-Tran spawns per device. The full list of UCCL-Tran tunables is documented in [`collective/rdma/README.md`](rdma/README.md#environment-variables-in-uccl).

### Table 2 — *(reserved for co-author Zhongjie)*

> @zhongjie: please fill in the script path and exact sweep parameters for Table 2.

---

## Reusing the artifact for future research

Beyond the figures in the paper, the artifact is structured so that downstream researchers can:

* **Add a new transport flavor** by extending `collective/rdma/` (RoCE/IB), `collective/efa/` (EFA), or `collective/afxdp/` (AF\_XDP) — each is a self-contained NCCL/RCCL net plugin.
* **Tune the data plane** via the `UCCL_*` environment variables documented in [`collective/rdma/README.md`](rdma/README.md#environment-variables-in-uccl) (chunk size, engine count, RC vs. UC mode, port entropy, traffic class, etc.).
* **Drop UCCL-Tran into existing PyTorch / DeepSpeed / vLLM / Megatron-LM workloads** by setting only `NCCL_NET_PLUGIN` (and `LD_PRELOAD` for EFA). See [`examples/README.md`](../examples/README.md) for a worked DDP example on CIFAR-10 / ResNet-18.

---

## Contact

For artifact questions during the AEC review window, please contact the authors via HotCRP. For long-term follow-ups, please open a GitHub issue at <https://github.com/uccl-project/uccl/issues>.
