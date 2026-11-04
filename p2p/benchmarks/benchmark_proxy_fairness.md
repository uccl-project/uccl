# Mixed-size proxy benchmark

`benchmark_proxy_fairness.py` measures two RDMA vector flows sharing **one
initiator Endpoint**. The large and small flows connect to **two independent
peer Endpoint processes**. Same-connection FIFO is preserved by the cooperative
proxy, so placing both flows on one connection would test a different condition.

**READ is the default and recommended first experiment.** Its destination buffers
are separate allocations in the initiator's address space. For WRITE, the current
adapter conservatively treats overlapping numeric remote address ranges as a
dependency, even across different peers: it has no trusted cross-process memory
identity. Two independent GPU processes can receive the same virtual addresses,
so two peers alone do not ensure that WRITE requests can overlap.

The script uses the native Python `reg`, `advertisev`, `writev_async` /
`readv_async`, and `poll_async` bindings. GPU allocations use the runtime selected
by `backend_info()` through ctypes; PyTorch is not loaded. `--help` needs only
Python and performs no GPU or network operations.

## Requirements

- A built and importable `uccl.p2p` containing the cooperative proxy change.
- Real IB/RoCE connectivity and GPU memory registration on all three processes.
  Two topologies are supported. **Single host:** all three processes run on one
  machine with one RDMA NIC and may share one GPU. The script connects with
  `Endpoint.connect`, which takes the RDMA path even between processes on the
  same host, so the transfers loop back through the NIC. Each Endpoint binds an
  OS-assigned out-of-band port; only the control-plane `--port` values must
  differ. **Two hosts:** the initiator runs on one machine and the two peers on
  another, on distinct GPUs, or on two further machines. Record which topology
  is used. Use the same topology, GPU/NIC placement, CPU affinity and runtime
  versions in both arms.
- Configure the site's `UCCL_SOCKET_IFNAME`, `UCCL_P2P_RDMA_DEV`, GID index and
  other required RDMA settings before launch. Metadata-advertised IPs must be
  reachable, in addition to the control-plane addresses given by `--peer`.
- The benchmark requires `UCCL_P2P_TRANSPORT=rdma`,
  `UCCL_P2P_DISABLE_IPC=1`, `UCCL_P2P_RDMA_CC=none`, and
  `UCCL_P2P_COMPRESS_STRATEGY=none`. It supplies missing values and rejects
  conflicting settings. Staged-network backends are rejected.
- Every IOV must fit the actual build's single-chunk limit to exercise raw
  vector submission. Verify `ChunkSplitStrategy::kMessageChunkSizeKB` in the
  source used to build the installed binary; pass that value in bytes through
  `--chunk-bytes`. The default declaration is 524288 bytes, matching the current
  source. This declaration is recorded, **not a runtime probe of raw-path use**.
  Reduce the IOV sizes if using a build with a smaller limit.

Running these commands uses GPU/RDMA resources. Obtain approval for the specific
machines and commands first when required by your environment; the script does
not launch remote jobs or request a benchmark through CI.

## Run one arm

Start a fresh peer process for each flow and substitute the actual build
identifier.

**Single host.** Run the three commands in three terminals on the same machine.
All three processes may use GPU 0. `--peer` carries the control-plane address;
the RDMA handshake uses the IP that each peer advertises in its metadata.

```bash
UCCL_P2P_COOPERATIVE_PROXY=0 python3 p2p/benchmarks/benchmark_proxy_fairness.py \
  --role peer --gpu 0 --port 29500 --build-label BUILD_ID

UCCL_P2P_COOPERATIVE_PROXY=0 python3 p2p/benchmarks/benchmark_proxy_fairness.py \
  --role peer --gpu 0 --port 29501 --build-label BUILD_ID

UCCL_P2P_COOPERATIVE_PROXY=0 python3 p2p/benchmarks/benchmark_proxy_fairness.py \
  --role initiator --gpu 0 --mode read \
  --peer 127.0.0.1:29500 --peer 127.0.0.1:29501 \
  --large-bytes 67108864 --large-iov-bytes 524288 --large-depth 2 \
  --small-bytes 65536 --small-iov-bytes 4096 --small-depth 1 \
  --chunk-bytes 524288 --warmup-seconds 2 --duration 10 \
  --build-label BUILD_ID --output default-read.json
```

For the cooperative arm, repeat the three commands in three new processes with
`UCCL_P2P_COOPERATIVE_PROXY=1` and `--output cooperative-read.json`.

**Two hosts.** These commands run in separate terminals on the selected peer
machines.

```bash
# Large peer. On a shared remote machine, use a distinct GPU for the small peer.
UCCL_P2P_COOPERATIVE_PROXY=0 python3 p2p/benchmarks/benchmark_proxy_fairness.py \
  --role peer --gpu 0 --port 29500 --build-label BUILD_ID

# Small peer; this may be the same remote machine at a different control port.
UCCL_P2P_COOPERATIVE_PROXY=0 python3 p2p/benchmarks/benchmark_proxy_fairness.py \
  --role peer --gpu 1 --port 29501 --build-label BUILD_ID
```

Then start the initiator. The first `--peer` serves large vectors and the second
serves small vectors. Use the same address twice with different ports if both
peer processes are on one machine.

```bash
UCCL_P2P_COOPERATIVE_PROXY=0 python3 p2p/benchmarks/benchmark_proxy_fairness.py \
  --role initiator --gpu 0 --mode read \
  --peer LARGE_PEER_IP:29500 --peer SMALL_PEER_IP:29501 \
  --large-bytes 67108864 --large-iov-bytes 524288 --large-depth 2 \
  --small-bytes 65536 --small-iov-bytes 4096 --small-depth 1 \
  --chunk-bytes 524288 --warmup-seconds 2 --duration 10 \
  --build-label BUILD_ID --output default-read.json
```

Both peers terminate after the initiator drains all operations, verifies payloads
and requests shutdown. For the cooperative arm, start **three new processes**
with `UCCL_P2P_COOPERATIVE_PROXY=1` and choose a different output filename.
Do not toggle the variable inside an existing process: engine configuration may
already be cached. Peers receive the mode from the control handshake. Alternate
the arm order across multiple pairs.

For WRITE, repeat with `--mode write`. Before submitting any requests, the
initiator decodes the native `FifoItem` descriptors returned by `advertisev`
(`addr` at byte 0, `size` at byte 8) and checks all cross-flow write intervals.
It rejects the run if any numeric ranges overlap, rather than silently measuring
serialized requests. The handshake checks matching host byte order.

If rejected, restart a peer with an allocation prefix, for example add
`--buffer-offset 268435456` to the **small peer's** command. This allocates an
extra 256 MiB per slot, registers the full allocation, and advertises only the
payload after the prefix. It changes the address layout but does not guarantee
disjoint addresses; the initiator checks the actual descriptors on every run.
Use the same offset and allocation settings for both A/B arms, and increase
`--max-buffer-mib` if the prefix plus payload across all slots exceeds its cap.
JSON records the peer offsets and the checked write intervals.

`--timeout` is a whole-process watchdog, including time a peer spends waiting for
the initiator, initialization, warmup, validation and teardown. Increase it for
slow registration or large buffer counts. Control sockets also have a timeout.
On successful shutdown the script deregisters and frees each allocation. On an
error or watchdog expiry it exits the process instead of freeing memory that
could still be referenced by an in-flight native request.

## Measurement boundaries

- Both flows are closed-loop: each has its own queue depth and separate local
  and remote allocations per outstanding slot. A slot is reused only after
  `poll_async` reports completion. Large requests are initially enqueued first.
- Warmup runs and drains before measurement. Each phase uses a different payload
  pattern; all buffers are fully checked after the phase drains. Initialization,
  network rendezvous, payload checking and teardown are outside the measurement.
  This verifies each slot's final payload, not every transfer's individual epoch.
- Small-request latency starts immediately before the Python vector submission
  call and ends when Python observes completion. It includes Python submission
  and polling overhead; it is neither a NIC timestamp nor serving latency. Both
  arms poll the small flow first. Percentiles use the nearest-rank definition,
  and the JSON includes samples for inspection.
- Submission continues for `--duration`, then all outstanding work drains.
  Large and total throughput divide all completed payload bytes by this complete
  interval, including the final drain. Units are decimal GB/s. The output records
  both requested submission duration and actual elapsed duration.
- CPU time is `time.process_time()` for the initiator process, including native
  proxy/background threads and Python, divided by the same elapsed interval for
  average CPU cores. It excludes peers and is not an isolated proxy CPU metric.
- JSON records workload, queue depths, environment, declared chunk limit, endpoint
  addresses, backend metadata, build labels and the loaded extension's SHA-256.
  Archive hardware/driver versions, CPU affinity and logs alongside it. Confirm
  the selected binary includes the opt-in implementation; the environment value
  alone cannot prove that an older binary used the cooperative scheduler.

No CPU simulation is used. A successful `--help` or syntax check supplies no
GPU/RDMA correctness or performance result. Hardware A/B results remain required
before making a speedup claim.
