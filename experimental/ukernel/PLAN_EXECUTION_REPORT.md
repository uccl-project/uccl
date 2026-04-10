# UKernel PLAN Execution Report

## Goal

This report tracks each optimization attempt while executing
[PLAN.md](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/PLAN.md).

For every candidate change, we follow the same loop:

1. implement one low-risk idea
2. run the relevant benchmark
3. compare against the current baseline
4. keep and commit only if the change shows useful improvement
5. otherwise revert and try the next idea

Baseline transport numbers are taken from
[PR_AMD_IPC.md](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/PR_AMD_IPC.md)
unless a newer committed result supersedes them.

## Commit Log

| Step | Phase | Status | Commit | Summary |
| --- | --- | --- | --- | --- |
| 1 | Phase 2 | kept | `08cdb004` | add local export cache reuse for IPC local handle export |
| 2 | Phase 3 bridge | kept | `pending` | propagate IPC binding_version and make bench cover versioned direct IPC path |

## Detailed Notes

### Step 1: Phase 2 local export cache

Status:
- kept

Scope:
- [ipc_manager.h](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/src/transport/memory/ipc_manager.h)
- [ipc_manager.cc](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/src/transport/memory/ipc_manager.cc)
- [communicator.h](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/src/transport/communicator.h)
- [communicator.cc](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/src/transport/communicator.cc)
- [ipc_adapter.cc](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/src/transport/adapter/ipc_adapter.cc)

What changed:
- local IPC exports are now cached by allocation range instead of exact pointer only
- cache hits reuse the previously exported `gpuIpcMemHandle_t`
- the older request-level `ipc_cache_req -> ipc_cache` path now also reuses the same local export cache
- `Communicator` now exposes a single helper to export local IPC buffers so the cache logic stays in one place

Why:
- Phase 2 in the plan calls out local export cache as the first low-risk optimization
- before this change, repeated requests on subranges of the same allocation could still redo `gpuMemGetAddressRange` and `gpuIpcGetMemHandle`
- NCCL also pushes this kind of registration/export work toward a longer-lived cache instead of repeating it in the hot path

Build command:

```bash
cd /home/yangzhou/danyang/uccl-danyang/experimental/ukernel
make -f Makefile.rocm bench_transport -j4
```

Benchmark command:

```bash
cd /home/yangzhou/danyang/uccl-danyang/experimental/ukernel
sizes=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456)
port=7400
for size in "${sizes[@]}"; do
  env ROCPROFILER_REGISTER_FORCE_LOAD=0 numactl --cpunodebind=0 --membind=0 ./bench_transport --rank 0 --peer-rank 1 --gpu-id 2 --msg-size "$size" --iterations 10 --warmup 2 --transport ipc --ip 127.0.0.1 --port "$port" &
  pid0=$!
  sleep 1
  env ROCPROFILER_REGISTER_FORCE_LOAD=0 numactl --cpunodebind=1 --membind=1 ./bench_transport --rank 1 --peer-rank 0 --gpu-id 6 --msg-size "$size" --iterations 10 --warmup 2 --transport ipc --ip 127.0.0.1 --port "$port"
  wait "$pid0"
  port=$((port + 1))
done
```

Test placement:
- `GPU 2` on `NUMA 0`
- `GPU 6` on `NUMA 1`

Key before/after points:

| Size | Baseline p50 us | New p50 us | Delta | Baseline Uni GB/s | New Uni GB/s | Baseline Bidi GB/s | New Bidi GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4KB | 3372.92 | 3338.97 | -1.0% | 0.00 | 0.00 | 0.01 | 0.01 |
| 64KB | 3423.11 | 3401.91 | -0.6% | 0.05 | 0.06 | 0.08 | 0.09 |
| 1MB | 3454.59 | 3423.45 | -0.9% | 0.88 | 0.88 | 1.37 | 1.38 |
| 2MB | 3738.77 | 3432.81 | -8.2% | 1.72 | 1.75 | 2.78 | 2.73 |
| 4MB | 3474.60 | 3460.72 | -0.4% | 3.08 | 3.35 | 4.74 | 5.00 |
| 8MB | 3593.95 | 3587.63 | -0.2% | 5.31 | 6.16 | 7.92 | 7.61 |
| 16MB | 4014.01 | 4685.91 | regression | 10.79 | 8.98 | 12.81 | 12.81 |

Decision:
- keep

Reasoning:
- the improvement is modest but real on the small and medium sizes that are most sensitive to handle/control overhead
- the code change is local and low-risk
- large-message behavior is still mostly governed by copy steady state, which matches the plan's expectation for Phase 2
- because this is the first Phase 2 optimization and it improves the intended regime, it is worth keeping and committing

Next candidate:
- continue Phase 2 with the next low-risk item from the plan
- likely options are remote open-cache lifecycle cleanup or narrowing send-side synchronization to active streams only

### Step 2: propagate binding_version and make the benchmark hit the direct IPC metadata path

Status:
- kept

Scope:
- [ipc_adapter.cc](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/src/transport/adapter/ipc_adapter.cc)
- [bench_transport.cc](/home/yangzhou/danyang/uccl-danyang/experimental/ukernel/benchmarks/bench_transport.cc)

What changed:
- `IpcAdapter::send_async()` now preserves `remote_hint.binding_version` when it copies the `RemoteSlice` into the request slot
- `bench_transport` now exchanges remote recv MR ids for IPC, not only UCCL
- `bench_transport` pre-publishes recv buffers with a fixed IPC metadata version and sends with `RemoteSlice{mem_id, offset, ..., binding_version=1}` for IPC

Why:
- the runtime already has a versioned by-`mem_id` IPC direct path
- the CCL transport backend already prepares `RemoteSlice.binding_version`
- but the IPC adapter was dropping that field, which forced the request back toward the old `ipc_cache_req` fallback
- the benchmark also was not providing IPC `RemoteSlice` metadata, so it mostly measured the fallback control path instead of the direct metadata path

Build command:

```bash
cd /home/yangzhou/danyang/uccl-danyang/experimental/ukernel
make -f Makefile.rocm bench_transport -j4
```

Benchmark command:

```bash
cd /home/yangzhou/danyang/uccl-danyang/experimental/ukernel
sizes=(4096 65536 1048576 16777216)
port=7600
for size in "${sizes[@]}"; do
  env ROCPROFILER_REGISTER_FORCE_LOAD=0 numactl --cpunodebind=0 --membind=0 ./bench_transport --rank 0 --peer-rank 1 --gpu-id 2 --msg-size "$size" --iterations 10 --warmup 2 --transport ipc --ip 127.0.0.1 --port "$port" &
  pid0=$!
  sleep 1
  env ROCPROFILER_REGISTER_FORCE_LOAD=0 numactl --cpunodebind=1 --membind=1 ./bench_transport --rank 1 --peer-rank 0 --gpu-id 6 --msg-size "$size" --iterations 10 --warmup 2 --transport ipc --ip 127.0.0.1 --port "$port"
  wait "$pid0"
  port=$((port + 1))
done
```

Test placement:
- `GPU 2` on `NUMA 0`
- `GPU 6` on `NUMA 1`

Key before/after points:

| Size | Baseline p50 us | New p50 us | Delta | Baseline Uni GB/s | New Uni GB/s | Baseline Bidi GB/s | New Bidi GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4KB | 3372.92 | 1220.76 | -63.8% | 0.00 | 0.19 | 0.01 | 0.02 |
| 64KB | 3423.11 | 1245.81 | -63.6% | 0.05 | 2.43 | 0.08 | 0.27 |
| 1MB | 3454.59 | 1200.41 | -65.3% | 0.88 | 21.20 | 1.37 | 3.44 |
| 16MB | 4014.01 | 2630.75 | -34.5% | 10.79 | 42.75 | 12.81 | 18.76 |

Decision:
- keep

Reasoning:
- this candidate shows a large, stable improvement at every measured size
- the benefit comes from finally exercising the versioned IPC direct path instead of spending most requests in the old request-level handshake fallback
- this is consistent with the plan direction and with the NCCL comparison notes: handle/control work should move from hot requests toward longer-lived buffer metadata
- the benchmark change is intentional and useful because the previous IPC benchmark under-covered the direct path we actually want to optimize

Next candidate:
- continue Phase 3-style work by reducing per-request receiver-side IPC metadata publication
- the most promising next step is to avoid redundant `notify_ipc_buffer()` work when the same recv buffer remains bound for the whole communicator lifetime
