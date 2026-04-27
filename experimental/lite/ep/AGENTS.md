# DeepEP-Lite

`experimental/lite/ep` is now based on DeepEPv2 PR605's NCCL GIN
`ElasticBuffer` path rather than the old `uccl.ep`/DeepEPv1-lite path.

Target environment:

- NVIDIA L4 / `sm_89`
- no NVLink
- no GPUDirect RDMA
- no IBGDA or UCCL-EP dependency in the migrated path
- NCCL GIN proxy mode (`NCCL_GIN_TYPE=2`)

Detailed migration notes and the latest benchmark matrix are in
`doc/deepep-v2-no-nvlink-no-gdr.md`.

## Required no-GDR runtime mode

```bash
EP_FORCE_NO_NVLINK=1
NCCL_NET_GDR_LEVEL=0
NCCL_GIN_TYPE=2
DISABLE_SM90_FEATURES=1
EP_TEST_DISABLE_FP8=1
EP_NCCL_ROOT_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
LD_PRELOAD=$EP_NCCL_ROOT_DIR/lib/libnccl.so.2:/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so
```

Notes:

- `NCCL_GIN_TYPE=2` selects NCCL GIN proxy mode. The default GDAKI-style path
  timed out or wedged on l40/l41 without GPUDirect RDMA.
- `EP_FORCE_NO_NVLINK=1` also forces a conservative correctness policy:
  single GIN QP and no multiple-reduction combine.
- `DISABLE_SM90_FEATURES=1` selects SM89 fallbacks for L4.

## Build and test

Activate the node-specific Python environment before rebuilding:

```bash
# l40
conda activate uccl

# l41
source ~/zhongjie/zj_py/bin/activate
```

```bash
make -j SM=89
make -j install
```

Benchmark workflows are packaged as project skills:

- `bench-lite-ep-intra`: single-node benchmark, default 2 GPUs.
- `bench-lite-ep-inter`: multi-node benchmark across `l40` and `l41`, default
  2 total GPUs.

Constraint: `num_experts` must be divisible by total GPU count.

## Benchmark (L4, 128 tok x 7168 hid x top-8 x 64 exp)

The table below uses DeepEPv2 NCCL GIN proxy mode, BF16 tests, no NVLink, no
GPUDirect RDMA, and one GIN QP. Values are averages across ranks.

| Setup | Physical GPUs | Dispatch BW | Dispatch latency | Combine BW | Combine latency | Reduced combine BW | Reduced combine latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1n x 2g | l40: GPU2,3 | 6.00 GB/s | 616.961 us | 6.00 GB/s | 580.603 us | 6.00 GB/s | 2507.500 us |
| 1n x 4g | l40: GPU0,1,2,3 | 7.75 GB/s | 863.959 us | 6.25 GB/s | 1027.750 us | 4.00 GB/s | 3510.250 us |
| 2n x 2g | l40/l41: GPU2,3 | 7.00 GB/s | 926.900 us | 6.00 GB/s | 1048.000 us | 4.00 GB/s | 3469.750 us |
| 2n x 4g | l40/l41: GPU0,1,2,3 | 5.00 GB/s | 1904.750 us | 5.00 GB/s | 2074.875 us | 5.25 GB/s | 2680.375 us |

Raw logs:

- `/tmp/deepep_matrix_1n2g.log`
- `/tmp/deepep_matrix_1n4g.log`
- `/tmp/deepep_matrix_2n2g.log`
- `/tmp/deepep_matrix_2n4g.log`

## Modification guidelines

- Keep no-NVLink behavior gated by `EP_FORCE_NO_NVLINK=1`.
- Do not re-enable multiple-reduction combine or multiple QPs in this mode
  without rerunning correctness on l40+l41 no-GDR proxy GIN.
- Guard SM90-only instructions/features behind `DISABLE_SM90_FEATURES` or
  equivalent architecture checks.
- Keep NCCL 2.30.4+ available at build and runtime for GIN host/device APIs.
