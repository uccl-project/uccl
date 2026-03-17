## Build

on AMD
```
The current benchmark set is CUDA/gdrcopy-oriented and does not have a ROCm build path yet.
```

on Nvidia
```
cd experimental/ukernel
make bench
```

## Transport benchmark

`bench_transport` uses the transport runtime exactly the way the module is
expected to be used by upper layers:
- peer-link selection is controlled by `--transport auto|ipc|uccl`
- `auto` resolves per peer: same host goes IPC, cross host goes UCCL
- `ipc` is only valid for same-host peers and fails fast otherwise
- socket/redis exchanger is only the bootstrap metadata path
- same-host IPC data movement uses shared-memory ring control plus CUDA IPC
- standalone benchmark runs use `rank` as the local IPC id; library users can
  set `CommunicatorConfig.local_id` explicitly or rely on launcher-local-rank
  environment variables

The benchmark runs three phases:
- latency ping-pong
- one-way throughput
- bidirectional throughput

Each phase also validates payload contents after completion, so it checks data
correctness in addition to completion/progress.

The benchmark uses a transport-specific in-flight window. IPC runs with a
larger window, while UCCL uses a smaller one to avoid hitting the backend
queue limit during bidirectional stress.

If you launch multiple benchmark pairs on one host, give each pair a unique
`--port` so the bootstrap exchanger does not collide.
