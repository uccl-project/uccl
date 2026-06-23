# CCL

## Build

```bash
cd experimental/ukernel/src/ccl
make clean
make -j$(nproc)
```

Common override:

```bash
make -j$(nproc) CUDA_HOME=/usr/local/cuda CONDA_LIB_HOME=/usr/lib SM=80
```

## Test

Run unit tests:

```bash
make test-unit
```

Run the integration benchmark (p2p copy performance).

1. Build the benchmark binary:

```bash
make test_perf_p2p_copy
```

2. Pick two GPUs that support P2P. The benchmark requires P2P peer access between
the two GPUs, so check the topology first and choose a pair shown as `OK`:

```bash
nvidia-smi topo -p2p r
```

Use that pair as `CUDA_VISIBLE_DEVICES` (same value in both terminals below).
`--gpu` indexes into that list: `--gpu=0` is the first GPU, `--gpu=1` the second.
The examples use `6,7`; replace it with your own `OK` pair if needed.

3. Launch the server and client in two terminals.

Terminal 1 — server:

```bash
CUDA_VISIBLE_DEVICES=6,7 ./test_perf_p2p_copy --role=server --gpu=0 --exchanger-port=6979
```

Terminal 2 — client:

```bash
CUDA_VISIBLE_DEVICES=6,7 ./test_perf_p2p_copy --role=client --gpu=1 --exchanger-ip=127.0.0.1 --exchanger-port=6979
```

The two processes benchmark three same-node P2P copy paths side by side — ukernel
`DeviceBackend` (several `blocks_per_worker`), CUDA `cudaMemcpyPeerAsync`, and
`Communicator::send_put_async` (IPC put). The server terminal (rank 0) prints the
final latency (us) and throughput (GB/s) tables over sizes from 1KB to 1GB.

Troubleshooting:

- `Failed to connect to Exchanger`: a stale run is still holding the port. Clean
  up leftovers and/or retry on a fresh port (same value in both terminals):

  ```bash
  pkill -f test_perf_p2p_copy
  # then retry, e.g. add --exchanger-port=7100 to both commands
  ```

- `Peer access NOT supported` / `Cannot resolve remote IPC`: the two GPUs have no
  P2P path — pick a pair shown as `OK` in `nvidia-smi topo -p2p r`.
- Start the server first, then the client within ~3s (leader-ready timeout is
  3000ms; raise it with `UHM_OOB_LEADER_READY_TIMEOUT_MS=30000`).

Run everything:

```bash
make test
```

## Notes

- `test-unit` covers planner, lowering, simulator, and executor behavior.
- `test-integration` builds the p2p copy performance test; run it manually in two terminals (see above).
