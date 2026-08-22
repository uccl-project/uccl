# RDMA Fused Transport Design

## Goal

Support cross-node fused reduce+copy / fused AllGather over RDMA, without
requiring the GPU to wait on RDMA completions.

## Design principle

The transport layer stays generic. Fused behavior lives entirely in CCL:

- CCL owns a host-visible D2H ring of command indices.
- CCL owns a stable `Cmd` pool.
- Each pool slot also stores the executor completion context:
  `run` pointer + `op_idx` + `put_path`.
- A CCL consumer (executor drain thread) calls `RdmaFusedProxy::progress()`:
  - pop cmd_index from ring,
  - look up `FusedCmdSlot` in the pool,
  - call `TransportBackend::do_enqueue_reserved()` with a fresh be_idx,
  - publish `(be_idx, run, op_idx)` into `tpt_slots_`,
  - so the existing `drain_tpt_loop` can process the completion normally.

This means the RDMA put and completion still go through the same
`Communicator` / `RdmaTransportAdapter` as normal puts. The proxy is only
an alternate producer; it is not a new transport.

The GPU kernel writes cmd_index into the ring after finishing reduce/copy.
No new command struct and no transport changes.

## Software path (current testbed)

```text
GPU kernel
  reduce/copy local data
  write cmd_index to host-visible D2H ring

CCL consumer (executor drain thread)
  pop cmd_index
  look up Cmd[cmd_index]
  call Communicator::send_put_async_with_rid()

Transport
  unchanged
```

- Works on ConnectX-6 + L40S + MOFED 26.04.
- No UAR export required.
- Reuses buffer registration, MR/rkey lookup, QP selection, and tag semantics.
- No new command struct.

## Hardware path (GDAKI, future testbed)

```text
GPU kernel
  reduce/copy local data
  write mlx5 WQE directly to tx_wq
  ring doorbell via GPU-mapped UAR
Host CQ / counter
  completion observed without GPU wait
```

- Requires DOCA GPUNetIO / NVSHMEM IBGDA and a NIC that supports UAR export.
- Current ConnectX-6 + L40S fails at `Can't export UAR to GPU`.

## Selection

`UK_CCL_RDMA_FUSED_MODE=auto|proxy|gda`

- `auto`: probe GDAKI support, fall back to software D2H ring.
- `proxy`: force software D2H ring.
- `gda`: force hardware path (fails loudly if unsupported).

## Integration points

- `SprayExecutor` owns `RdmaFusedProxy` and drains the D2H ring from
  `drain_tpt_loop`.
- `RdmaFusedProxy::progress()` -> `SprayExecutor::submit_fused_cmd()`
  -> `TransportBackend::do_enqueue_reserved()` + `tpt_slots_.write()`.
- Device kernel: after a fused reduce+copy, calls
  `rdma_fused_ring_push()` (device/rdma_fused_ring_device.cuh).
- `DeviceBackend` will write `dst2 = ring handle`,
  `signal_tag = cmd_index`, `taskFlags |= kFlagRdmaFusedProxy`.

## Current status

- [x] `RdmaFusedProxy` ring + Cmd pool + progress (host side)
- [x] Real GDR-mapped D2H ring
- [x] Executor integration: `submit_fused_cmd()` publishes BeSlot
- [x] Device helper `rdma_fused_ring_push()`
- [x] Device kernel skips remote IPC copy in proxy mode and pushes cmd index
- [x] Host unit test `test_rdma_fused_proxy`
- [ ] DeviceBackend/executor allocation of CmdPool slots
- [ ] Cross-node fused validation

