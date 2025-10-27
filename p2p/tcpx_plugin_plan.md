# TCPX Plugin Implementation Plan

## 1. Objectives & Current State
- Deliver an NCCL net plugin that enables GPU-to-GPU transfers over TCPX on GCP A3-high (2 nodes x 8xH100, 4x gVNIC).
- Reuse the RDMA-based `uccl_engine` plugin (see `p2p/uccl_engine.cc`) as the architectural template.
- Bridge the existing standalone TCPX library in `p2p/tcpx` with Google's `nccl-plugin-gpudirecttcpx` APIs.

## 2. Preparatory Work
- [ ] Sync with the RDMA plugin implementation from `https://github.com/ai-dynamo/nixl/pull/895` and extract reusable patterns (metadata format, state machines, async flows).
- [ ] Inventory TCPX-side capabilities (`p2p/tcpx_engine.{h,cc}` and the `p2p/tcpx` submodules) vs. what `uccl_engine` expects (connection lifecycle, FIFO semantics, async polling).
- [ ] Lock in the control port and reuse the existing static GPU↔gVNIC mapping already defined in the TCPX helpers.

## 3. Phase A – TCPX Endpoint Parity
- [ ] Lock interface targets: reuse `uccl_engine`’s endpoint contract and only touch `tcpx_engine.{h,cc}` plus small helpers in `p2p/tcpx`.
- [ ] Control/metadata path:
  - replace the unfinished control-socket handshake with a minimal JSON blob (`{ip, port, gpu}`) exchanged via `send_ctrl_struct`.
  - wire `get_metadata` / `parse_metadata` to wrap and unwrap that blob without introducing new dependencies.
- [ ] Connection bring-up:
  - call the existing `tcpx_listen` / `tcpx_connect_v5` helpers and ensure the returned comm handle is stored in `Conn`.
  - keep gVNIC selection static by reusing the current PCI mapping helpers; no dynamic probing added.
- [ ] Memory registration & advertisement:
  - surface a thin shim around the TCPX library’s `register_memory` so `reg`/`dereg` simply forward the pointer and store the returned id.
  - implement `advertise` by serialising the FIFO slot returned from the TCPX library (or a locally fabricated struct until FIFO support lands).
- [ ] Async data flow:
  - implement `send_async`, `recv_async`, and `read_async` as thin wrappers over the TCPX transfer queue, returning monotonically increasing transfer ids.
  - add a simple `poll_async` map inside `Endpoint` that marks transfers complete when TCPX signals completion; avoid extra threads beyond the unpacker.
- [ ] Vector fallbacks: translate `*_v` calls to loops over the scalar variants so `uccl_engine` works without structural changes.

## 4. Phase B – Plugin Wiring in NIXL
- [ ] Clone the RDMA plugin scaffold (provider registration, environment hooks, stats) and rename it for TCPX (e.g., `net_tcpx`).
- [ ] Replace RDMA-specific calls with the TCPX equivalents by reusing `uccl_engine` but compiling it with `USE_TCPX` so it pulls in `tcpx::Endpoint`.
- [ ] Ensure the plugin exposes the same NCCL v7 entry points (`listen`, `connect`, `regMr`, `deregMr`, `isend`, `irecv`, `iflush`, `isendv`, `irecvv`, `ptest`).
- [ ] Plumb through topology hints needed for NUMA-aware NIC selection (leverage `find_best_dev` in `tcpx_engine.cc`).
- [ ] Update build scripts / Bazel / CMake (wherever NIXL houses plugins) to link the TCPX static library and add the right compile definitions.

## 5. Phase C – Multi-gVNIC & Performance Tuning
- [ ] Honor the existing static GPU↔gVNIC mapping in the TCPX library; skip additional binding logic unless gaps appear in testing.
- [ ] Implement per-channel flow control only if TCPX exposes it; otherwise rely on the base queue limits.
- [ ] Tune batching constants (`kDescRingSize`, inflight limits) after basic functionality is verified.

## 6. Phase D – Validation & Tooling
- [ ] Smoke test control handshake and MR registration locally using loopback.
- [ ] Reuse the existing RDMA tests in `p2p/tests` with a TCPX backend switch for basic coverage; deeper soak tests can follow post-merge.
- [ ] Optional: lightweight script for 2-node, 8x GPU runs to spot queue depth issues once the core path is stable.

## 7. Deliverables & Follow-ups
- [ ] Working TCPX NCCL net plugin library packaged with NIXL.
- [ ] Documentation updates covering deployment on A3-high (NIC layout, env vars, required kernel modules).
- [ ] Performance report comparing RDMA plugin vs. TCPX on the target hardware.
- [ ] Backlog items: native vector ops, dynamic gVNIC selection, telemetry hooks.

## 8. Open Decisions / Questions
- None for now; revisit if TCPX library changes or if additional NCCL entry points are required.
