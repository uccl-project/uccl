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

## Full-model comparison

Use a private CUDA PyTorch environment with Transformers **4.57.1** and the
public `ibm-granite/granite-3.1-1b-a400m-base` snapshot at
`408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`. `model_e2e.py` reads this snapshot
offline. Native NCCL 2.30.4 device headers are required to build its C ABI.

```sh
TESTS=experimental/uccl_gin/tests/local
make -C "$TESTS" SM=120 NCCL_INCLUDE_DIR=/path/to/nccl/include \
  EXTRA_DEVFLAGS=-DNCCL_GIN_GDAKI_ENABLE=0 model-tests -j2
make -C "$TESTS" SM=120 NCCL_INCLUDE_DIR=/path/to/nccl/include \
  BUILD_DIR=build/scalar-model-sm120 \
  EXTRA_DEVFLAGS='-DNCCL_GIN_GDAKI_ENABLE=0 -DMODEL_EXPECT_SCALAR_FLUSH=1' \
  model-tests -j2
python "$TESTS/model_e2e.py" --library "$TESTS/build/sm120/libmodel_transport.so" \
  --arm v2 --weights /path/to/verified-granite --device 0 --queues 32 \
  --batch-size 64 --concurrency 128 --prompt-tokens 64 --new-tokens 4 \
  --rounds 3 --warmup 1
```

For scalar, select `build/scalar-model-sm120/libmodel_transport.so` and
`--arm scalar`. Run both B32/B64 in fresh processes, alternating scalar/warp,
warp/scalar, scalar/warp across three trials. Retain both post-warmup waves per
process. Each process first checks seven tensor cases, runs an untimed native
model reference and then verifies every generated token, full logits hash,
payload byte and WRITE/signal/QUIET count. Scalar and warp use the same typed
put/signal prerequisite.

This exercises the pretrained 24-layer MoE model through a single-device test
receiver. All C128 requests enter a finite asynchronous queue before serial
B32/B64 model batches execute. Wave timing includes inference, staging and the
CPU payload/source-reuse oracle; the receiver supplies local CUDA copies.

See [integration results](MAIN_INTEGRATION_RESULTS.md) and
[combined fix results](MAIN_FIX_RESULTS.md). Distributed RDMA and model E2E on
additional device types need their own actual-hardware campaigns.
