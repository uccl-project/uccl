# UCCL-EP intra-node + inter-node gap vs NCCL — open work

Latest baseline (commit `d74a9b5b`, no EP path changes), 4 rounds A/B
(warmup discard) on L4 PCIe-only / CX7 ×2 / dual-channel DDR4-3200,
no-GDR, validation shape `128 tok × 7168 hid × top-8`:

| Topology | Stage | EP latency | NCCL a2a ref | Gap |
| --- | --- | ---: | ---: | ---: |
| 1n × 4g | dispatch | 1183 µs | 868 µs |  ~1.4× |
| 1n × 4g | combine  |  520 µs | (≈ a2a) |  — |
| 2n × 4g | dispatch | 2148 µs | 998 µs |  ~2.2× |
| 2n × 4g | combine  | 1007 µs | (≈ a2a) |  — |

## Dense collective reference: NCCL allgather / reduce-scatter

For the cleanest payload-level comparison against dense collectives, reran
2n × 4g with `num_experts=8`, `num_topk=8`, `num_tokens=128`,
`hidden=7168`, BF16, `LITE_EP_NVLINK=0`, `do_handle_copy=0`,
`expert_alignment=128`, and `fp8_dispatch=0`.  With 8 total EP ranks,
this is the dense-routing case: one expert per rank and top-8 covers all
EP ranks.

The NCCL reference used the same per-rank dense payload size,
`128 * 7168 * 2 * 8 = 14,680,064` bytes, and measured NCCL
`allgather` for dispatch-like traffic plus NCCL `reducescatter` for
combine-like traffic.  Bandwidths below are normalized to that fixed
dense payload.  `reduced combine` is the better semantic comparison to
NCCL `reducescatter`; plain `combine` first applies local top-k
accumulation in the benchmark harness and is therefore closer to the
actual Lite-EP pipeline than to a pure reduce-scatter primitive.

| Mode | Lite-EP dispatch | NCCL allgather | Gap |
| --- | ---: | ---: | ---: |
| no-GDR | 2540 µs / 5.78 GB/s | 825 µs / 17.79 GB/s | 3.08× slower |
| GDR | 2579 µs / 5.69 GB/s | 971 µs / 15.12 GB/s | 2.66× slower |

| Mode | Lite-EP reduced combine | NCCL reducescatter | Gap |
| --- | ---: | ---: | ---: |
| no-GDR | 1150 µs / 12.77 GB/s | 834 µs / 17.61 GB/s | 1.38× slower |
| GDR | 1204 µs / 12.19 GB/s | 980 µs / 14.98 GB/s | 1.23× slower |

Takeaway: `reduced combine` is already close to NCCL reduce-scatter
for this dense case, while dispatch remains the dominant gap.  Even in
the dense-routing configuration, dispatch still runs the dynamic MoE
routing path: top-k count exchange, slot assignment, per-token UCCL
WRITE commands, CPU-proxy verbs posting, channel tail/finish signals,
forwarding, and copy epilogue.  NCCL `allgather` avoids those steps
because source/destination offsets are static and payloads are moved as
bulk contiguous collective transfers.

### Reproducing on l40/l41

Testbed aliases and common settings used for the dense 2n × 4g runs:

| Role | Host | IB address | GPUs |
| --- | --- | --- | --- |
| l40 / node 0 | `mibura-sky-test-01` | `10.10.55.1` | `0,1,2,3` |
| l41 / node 1 | `mibura-sky-test-02` | `10.10.55.2` | `0,1,2,3` |

Use the Python 3.12 `uccl` environment and the NCCL root that the
Lite-EP extension was built against.  The `/home/yangz/nfs/zhongjie/nccl`
tree is newer/older depending on local rebuilds and can trigger NCCL ABI
checks in `deep_ep._C`.

```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/lite-ep

PY=/home/yangz/nfs/miniconda3/envs/uccl/bin/python
NCCL_ROOT=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
EP_DIR=$PWD
```

Lite-EP dense dispatch / reduced-combine run (`num_experts=8`, `topk=8`):

```bash
run_lite_ep_dense() {
  mode="$1"                    # nogdr or gdr
  port="$2"
  transport=uccl-no-gdr
  gdr_level=0
  if [[ "$mode" == gdr ]]; then
    transport=uccl-gdr
    gdr_level=SYS
  fi

  common_env="WORLD_SIZE=2 LOCAL_WORLD_SIZE=4 MASTER_ADDR=10.10.55.1 MASTER_PORT=$port \
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$EP_DIR EP_NCCL_ROOT_DIR=$NCCL_ROOT \
LD_LIBRARY_PATH=$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-} LD_PRELOAD=$NCCL_ROOT/lib/libnccl.so.2 \
EP_SUPPRESS_NCCL_CHECK=1 LITE_EP_TRANSPORT=$transport LITE_EP_NVLINK=0 \
NCCL_SOCKET_IFNAME=ibp55s0f0 NCCL_IB_HCA=mlx5_0,mlx5_1 UCCL_IB_HCA=mlx5_0,mlx5_1 \
NCCL_NET_GDR_LEVEL=$gdr_level NCCL_GIN_TYPE=2 DISABLE_SM90_FEATURES=1 \
EP_JIT_CACHE_DIR=$EP_DIR/.jit-cache"

  args="tests/elastic/test_ep.py --num-processes 4 \
--num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=8 \
--test-first-only --skip-check --do-handle-copy-modes=0 \
--expert-alignment-modes=128 --fp8-dispatch-modes=0 --num-bench-tests=5"

  ssh 10.10.55.2 "cd $EP_DIR && env $common_env RANK=1 timeout 360s $PY $args" \
    > /tmp/lite_ep_topk8_exp8_${mode}_l41.log 2>&1 &
  remote_pid=$!

  env $common_env RANK=0 timeout 360s $PY $args \
    | tee /tmp/lite_ep_topk8_exp8_${mode}_l40.log
  local_status=${PIPESTATUS[0]}

  wait "$remote_pid"
  remote_status=$?
  echo "REMOTE_STATUS=$remote_status LOCAL_STATUS=$local_status"
  grep -E 'EP:|Config|Experts|legacy:' /tmp/lite_ep_topk8_exp8_${mode}_l41.log || true
  return $(( local_status != 0 ? local_status : remote_status ))
}

run_lite_ep_dense nogdr 29660
run_lite_ep_dense gdr   29661
```

For the NCCL dense collective reference, the numbers above were collected
with a small PyTorch/NCCL helper rather than MPI `nccl-tests`, because the
MPI `nccl-tests` allgather/reduce-scatter path built against the local
NCCL 2.29.7 tree segfaulted on this 2n × 4g BF16 run before printing the
first measurement.  The helper uses the same NCCL backend but directly
calls `dist.all_gather_into_tensor` and `dist.reduce_scatter_tensor`.
Create it on both nodes (or `scp` after creating it locally):

```bash
cat >/tmp/bench_nccl_collectives.py <<'PY'
import argparse
import os
import torch
import torch.distributed as dist

def init_dist(local_rank, num_local_ranks):
    node_rank = int(os.environ.get('RANK', 0))
    num_nodes = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank)
    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD

def bench(fn, warmup, iters, group):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters

def worker(local_rank, num_local_ranks, args):
    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    elems = args.num_tokens * args.hidden
    dense_bytes = elems * 2 * world_size
    ag_in = torch.empty(elems, dtype=torch.bfloat16, device='cuda').normal_()
    ag_out = torch.empty(world_size * elems, dtype=torch.bfloat16, device='cuda')
    rs_in = torch.empty(world_size * elems, dtype=torch.bfloat16, device='cuda').normal_()
    rs_out = torch.empty(elems, dtype=torch.bfloat16, device='cuda')
    for name, fn in (
        ('allgather', lambda: dist.all_gather_into_tensor(ag_out, ag_in, group=group)),
        ('reducescatter', lambda: dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.SUM, group=group))):
        t = bench(fn, args.warmup, args.iters, group)
        metric = torch.tensor([t], dtype=torch.float64, device='cuda')
        dist.all_reduce(metric, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            max_us = metric.item()
            print(f'BOTTLENECK op={name} time_us={max_us:.2f} '
                  f'dense_GBs={dense_bytes / (max_us * 1e-6) / 1e9:.2f}', flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-tokens', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
PY

scp /tmp/bench_nccl_collectives.py 10.10.55.2:/tmp/bench_nccl_collectives.py
```

Run the NCCL reference in no-GDR and GDR modes:

```bash
run_nccl_dense() {
  mode="$1"                    # nogdr or gdr
  port="$2"
  gdr_level=0
  if [[ "$mode" == gdr ]]; then
    gdr_level=SYS
  fi

  common_env="WORLD_SIZE=2 LOCAL_WORLD_SIZE=4 MASTER_ADDR=10.10.55.1 MASTER_PORT=$port \
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ibp55s0f0 \
NCCL_IB_HCA=mlx5_0,mlx5_1 NCCL_NET_GDR_LEVEL=$gdr_level NCCL_P2P_DISABLE=1"

  ssh 10.10.55.2 "cd $EP_DIR && env $common_env RANK=1 $PY /tmp/bench_nccl_collectives.py \
--num-processes 4 --num-tokens 128 --hidden 7168 --warmup 20 --iters 50" \
    > /tmp/nccl_collectives_${mode}_l41.log 2>&1 &
  remote_pid=$!

  env $common_env RANK=0 $PY /tmp/bench_nccl_collectives.py \
    --num-processes 4 --num-tokens 128 --hidden 7168 --warmup 20 --iters 50 \
    | tee /tmp/nccl_collectives_${mode}_l40.log
  wait "$remote_pid"
}

run_nccl_dense nogdr 29670
run_nccl_dense gdr   29671
```

The MPI `nccl-tests` commands are still useful for debugging once the
local NCCL 2.29.7 segfault is resolved.  Use `-b`/`-e 1835008` because
`nccl-tests` takes the per-rank chunk size, while the table above reports
bandwidth normalized to `8 * 1835008` bytes:

```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite

COMMON_PATHS="SDK_DIR=$PWD/lite-collective/.tmp/mint-nccl-sdk \
BUILD_DIR=$PWD/lite-collective/.tmp/mint-nccl-tests-mpi-build \
RUNTIME_ROOT=$PWD/lite-collective/.tmp/mint-nccl-tests-runtime"

env $COMMON_PATHS NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ibp55s0f0 \
  NCCL_IB_HCA=mlx5_0,mlx5_1 NCCL_NET_GDR_LEVEL=0 NCCL_P2P_DISABLE=1 \
  bash lite-collective/scripts/run-nccl-tests.sh \
  --test all_gather --backend nccl --topology inter \
  --hosts 10.10.55.1,10.10.55.2 --gpus 0,1,2,3 \
  --min-bytes 1835008 --max-bytes 1835008 \
  --iters 50 --warmup-iters 20 -- -d bfloat16 -a 3

env $COMMON_PATHS NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ibp55s0f0 \
  NCCL_IB_HCA=mlx5_0,mlx5_1 NCCL_NET_GDR_LEVEL=0 NCCL_P2P_DISABLE=1 \
  bash lite-collective/scripts/run-nccl-tests.sh \
  --test reduce_scatter --backend nccl --topology inter \
  --hosts 10.10.55.1,10.10.55.2 --gpus 0,1,2,3 \
  --min-bytes 1835008 --max-bytes 1835008 \
  --iters 50 --warmup-iters 20 -- -d bfloat16 -a 3
```

For the GDR variant of the MPI `nccl-tests` commands, keep all arguments
the same and change `NCCL_NET_GDR_LEVEL=0` to `NCCL_NET_GDR_LEVEL=SYS`.

## Gap source breakdown (revised after empirical test)

| Gap source | 1n × 4g | 2n × 4g |
| --- | ---: | ---: |
| ~~Extra host DDR R + W vs NCCL's 1× GPU↔host pass~~ **REFUTED — see below** | 0 µs | 0 µs |
| GPU kernel inherent cost (SHM write fan-out + setup + barriers) | ~1150 µs | ~1150 µs |
| Per-WR posting × ~400 × 1.5 µs (mlx5 doorbell + spinlock) | — | ~600 µs |
| CPU-proxy path latency vs GPU-driving NIC | — | ~150 µs |
| Channel-finish PUT_VALUEs | — | ~100 µs |

### Small-batch SM cap

For UCCL proxy + `LITE_EP_NVLINK=0` runs with
`num_max_tokens_per_rank <= 256`, the automatic SM selector now caps the EP
kernel at 24 SMs by default (`EP_UCCL_SMALL_BATCH_NUM_SMS=24`). The upstream
non-overlap floor of 64 SMs over-parallelizes the L4/PCIe validation shape and
adds setup/barrier overhead. Explicit `num_sms` arguments still take
precedence.

### Why "extra host DDR R + W" turned out to cost 0 µs

Initial analysis estimated the proxy memcpy (sender slice → receiver
slice) added ~300 µs of DDR work to the critical path on 1n × 4g.
Empirical test (6-round interleaved A/B with `EP_UCCL_INTRANODE_DIRECT`
toggle, std ≈ 30 µs):

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| Proxy memcpy (default) | 1152 ± 38 µs | 506 ± 24 µs |
| Direct sender-to-peer-slice write | 1152 ± 31 µs | 493 ± 27 µs |
| Δ | **0 µs (0%)** | -13 µs (-2.6%) |

So the proxy memcpy is **fully overlapped with concurrent GPU PCIe and
sender/receiver activity** — it's not on the critical path. The DDR
round-trip estimate ignored PCIe/DDR pipelining.

This rules out **all intra-node memcpy / DDR-bandwidth optimizations**
as a source of measurable improvement on this hardware:

- ~~NT-store memcpy~~: regresses 2n × 4g, 0 µs win on 1n × 4g.
- ~~THP madvise + page pre-touch~~: 0 µs win, kept opt-in only.
- ~~Sender direct-write to peer slice~~ (eliminates proxy memcpy):
  0 µs win on 1n × 4g (within 30 µs noise band).

**Real intra-node ceiling on 1n × 4g is the GPU dispatch kernel
itself**, not the host-side memcpy. NCCL beats us by ~285 µs because
its GPU-side ring code is leaner (no token-count all-to-all + layout
discovery + per-expert atomic counters).

## Open optimization ideas (ranked by expected payoff, post-refute)

1. **Coalesce per-token RDMA WRITEs into per-peer bulk WRITEs** (2n × 4g
   only).  EP currently posts ~400 WRs/iter on 2n × 4g (one per token).
   Real fix needs the kernel to lay out remote slots contiguously so
   multiple tokens share a single SGE — current layout interleaves
   slots by `(channel, slot, rank)` which makes proxy-side multi-SGE
   merging a no-op (already documented + tested).  Expected payoff:
   up to ~600 µs on 2n × 4g.  Significant kernel-layout change.

2. **Eliminate dispatch setup phase (token-count all-to-all + layout
   discovery)**.  Either run-ahead the count exchange a few iterations
   in advance, or fold it into the previous combine's barrier traffic.
   Expected payoff: ~100-150 µs on every config.  Touches
   `hybrid_dispatch.cuh` non-trivially.

3. **Inspect remaining SM89 single-lane sites in dispatch hot path**.
   The fan-out write loop for ~5.5 MB SHM payload should hit ~16 GB/s
   per GPU (PCIe upstream limit) ≈ 350 µs; if the kernel takes longer
   than that, vectorization is incomplete.  Diagnostic via kineto
   trace of `hybrid_dispatch_impl` then targeted PTX inspection.

## Refuted hypotheses (kept as opt-in env vars or reverted)

- **AVX2 NT-store memcpy + cross-iter prefetch**: regresses 2n × 4g
  combine +64% (cache-snoop loss on receiver-GPU PCIe DMA).  Reverted.
- **THP `madvise(MADV_HUGEPAGE)` + page pre-touch on shared window**:
  no measurable change, within run-to-run noise.  Reverted.
- **Heap-thrash elimination** (recycle proxy `std::vector` scratch
  across calls): no real win in steady-state interleaved A/B.  Reverted.
- **Proxy threads 4 → 8**: previously unstable.  Documented dead end.
- **Intra-node direct host-window write (eliminate proxy memcpy)**:
  0 µs improvement on 1n × 4g end-to-end latency under 6-round
  interleaved A/B (noise std ~30 µs).  The proxy memcpy is fully
  overlapped with concurrent GPU PCIe traffic and is not on the
  critical path.  However, it does eliminate ~22 MB/iter of host DDR
  traffic and frees the proxy thread's CPU time, so the path is
  enabled by default (`EP_UCCL_INTRANODE_DIRECT=1`) for the benefit of
  co-located workloads (e.g. KV cache prefetch, mixed RDMA traffic).
  Set `=0` to fall back to proxy memcpy for diagnostics.

## Validation harness

- `tests/elastic/bench_nccl_a2a.py` — NCCL a2a ref using same
  `legacy_bytes` as EP bench; comparable directly to EP `legacy GB/s`.
- `.bench/run_ab.sh <label> [env=val ...]` — runs 1n × 4g + 2n × 4g
  with given env, dumps logs to `.bench/<label>_{1n4g,2n4g}/`.
- `.bench/run_reps.sh <label> <reps> [env=val ...]` — N reps, drops
  rep 0 as warmup, prints per-rep + mean.
- `.bench/run_interleaved.sh <A_label> <A_envs> <B_label> <B_envs> <rounds>`
  — interleaved A/B, randomizes JIT/thermal effects; the only
  methodology that gives reproducible signal under run-to-run noise of
  ~10%.
