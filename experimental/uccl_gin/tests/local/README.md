# Native GIN and NCCL-EP validation

The native build uses the production mapped-host FIFO and real NCCL 2.30.4
device headers. No EFA, MPI, PyTorch headers or NCCL runtime are required by the
local executables. Set `NUMA_INCLUDE_DIR` and `NUMA_LIBRARY_DIR` for a private
libnuma prefix.

```sh
make -C experimental/uccl_gin/tests/local SM=120 CUDA_HOME=/usr/local/cuda \
  NCCL_INCLUDE_DIR=/path/to/nccl/include \
  EXTRA_DEVFLAGS=-DNCCL_GIN_GDAKI_ENABLE=0 \
  all coop-tests compile-fail adapter-tests adapter-compile-fail ht-tests \
  model-tests perf-controls -j2
```

`queue_smoke` compares every encoded command field, FIFO wrap/reuse and scalar
QUIET completion. `coop_flush` publishes one WRITE per member, calls the actual
standalone/adapter flush, and immediately overwrites the source. A separate host
consumer copies each generation from HBM before acknowledgement. Per-queue
counts, proxy-major routing, delayed acknowledgement and system-acquire status
checks detect premature source reuse. Uneven Q=3/33/64 and thread/private groups
exercise queue coverage. Unsupported CTA flush/signal groups must fail compilation.

```sh
BIN=experimental/uccl_gin/tests/local/build/sm120
"$BIN/coop_flush" --via adapter --group warp --warps 64 --queues 32 \
  --capacity 4096 --batch-size 2048 --hidden-size 2048 --num-lanes 4 \
  --channel-hint warp --iterations 3 --rounds 5
"$BIN/adapter_signal" --device 0
"$BIN/ht_single_rank-uccl" --batch 64 --concurrency 128 --rounds 7
"$BIN/ht_single_rank-nccl" --batch 64 --concurrency 128 --rounds 7
```

`coop_flush-scalar` and `coop_flush-elected` are test-only controls. All three
arms have entry/exit warp rendezvous and the same host/source oracle. Scalar
drains all queues on every member; elected drains on member zero; production
assigns queue r+k*warp_size to member r. Compare completed rounds in alternating
arm order. Capacity and routing remain identical across arms.

The single-rank HT fixture runs metadata preprocessing, dispatch, a deterministic
expert transform and combine; it compares every token/probability and routing
result. B denotes requests grouped into a batch, each with 128 tokens. C denotes
finite queued requests; the batches execute serially on each device. This is not
HTTP concurrency or a pretrained model forward. Both backend builds must produce
the same complete output hash. The tensor/model C ABI fixture is optional.

See [integration results](MAIN_INTEGRATION_RESULTS.md) and
[combined fix results](MAIN_FIX_RESULTS.md). Full distributed RDMA and model E2E
need their own actual-hardware campaign.
