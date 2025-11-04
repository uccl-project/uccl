TCPX Engine Refactor Plan
=========================

Goal
----
- Refactor `p2p/tcpx_engine.cc` so it keeps the lightweight RDMA-endpoint style API while *directly* implementing the functionality proven in the standalone `TcpxSession`/`TcpxTransfer` path (chunked pipeline, staged completion, sliding window) but in a single-channel, minimal-change form.
- Preserve compatibility with the existing UCCL shim (`p2p/uccl_engine.cc`) and the NIXL TCPX plugin (`thirdparty/nixl/src/plugins/tcpx`) so benchmarks and smoke tests continue to run.

Current State Summary
---------------------
- `tcpx::Endpoint` currently performs a minimal control handshake and issues `tcpx_isend`/`tcpx_irecv` calls directly. Completion is tracked through a shallow `PendingTransfer` map with limited state, lacking chunked staging and flow control—large transfers stall (see memory notes).
- The standalone `TcpxSession`/`TcpxTransfer` code under `p2p/tcpx` demonstrates the required behavior (chunk vector, Stage 1/Stage 2 pipeline, sliding windows), but it is architecturally heavier than desired and not meant to be pulled in wholesale.
- `p2p/uccl_engine.cc` expects the same surface as the RDMA endpoint (`connect/accept`, `reg/dereg/find_mr_by_addr`, `send_async/recv_async/read_async`, `queue_read_response`, `poll_async`). Any refactor must keep these signatures stable.

High-Level Strategy
-------------------
1. Keep `tcpx::Endpoint` as the only public abstraction but rebuild its internals to mirror the proved workflow from `TcpxSession`/`TcpxTransfer`. Re-implement the necessary pieces locally: single-channel bootstrap, per-connection channel state, memory registration cache, and the two-stage chunk pipeline.
2. Extend the existing control-plane exchange to negotiate all channel handles and parameters in one go (server advertises serialized handles; client consumes them). Reuse lightweight helpers already included (`bootstrap.h`, `tcpx_interface.h`) and continue to avoid heavy dependencies on the other `p2p/tcpx` components.
3. Introduce compact structs within `tcpx_engine.cc` to track per-channel state (recv inflight queue, pending kernels, send queue) and per-transfer progress (chunk vectors, CUDA events), closely following the logic from `TcpxTransfer` but tuned for the endpoint’s lifecycle.
4. Update async APIs (`send_async`, `recv_async`, `read_async`) to build chunk vectors, submit tcpx operations, launch unpack kernels, and enforce sliding-window limits, all within `tcpx::Endpoint`.
5. Enhance `poll_async` to drive both Stage 1 (tcpx_test + kernel launch) and Stage 2 (event drain + tcpx_irecv_consumed), ensuring transfers retire only after kernels finish and buffers are released.

Execution Plan
--------------
Phase 0 – Configuration & Baseline
- Reconfirm environment knobs from `memory.md` (`UCCL_TCPX_CHUNK_BYTES`, `UCCL_TCPX_MAX_{SEND,RECV}_INFLIGHT`, channel count envs). Define reasonable defaults inside `tcpx_engine.cc`, keeping them overridable via env variables.
- Document the control-plane metadata requirements for NIXL so any handshake extensions remain backward-compatible (IP/port/GPU stay first-class).
- Fix chunk sizing to the provided values (524288 bytes default, 1048576 for NVLink) unless overridden; expose a single knob (e.g., `UCCL_TCPX_CHUNK_BYTES`) that defaults to 524288 to match NCCL settings.
- Standardize on a single shared CUDA stream for unpack launches (reuse the existing engine-level stream instead of per-connection streams).

Phase 1 – Control Plane & Channel Bring-up
- Enhance the existing TCP control handshake to exchange all channel handles:
  - Server path (`accept`): after receiving the peer request, create a single listen comm pair (`tcpx_listen`) and send its serialized handle over the control socket. Wait for client ACK.
  - Client path (`connect`): parse the handle, call `tcpx_connect`, acknowledge, and cache the resulting send/recv comm pointers.
- Keep teardown symmetric and reuse existing `Conn` structure where possible to minimize structural churn.

Phase 2 – Memory Registration Cache
- Extend `MrEntry` to track single-channel memory handles (`tcpx_reg_mr`) for both send and recv directions, along with pointer metadata for `find_mr_by_addr`.
- On `reg`, register once per direction and store the opaque handles; on `dereg`, call `tcpx_dereg_mr`.
- Maintain fast lookup (e.g., unordered_map keyed by pointer range) so `advertise` and FIFO handlers remain efficient.

Phase 3 – Transfer Structures & Chunk Pipeline
- Extend `PendingTransfer` to capture:
  - Transfer kind (send/recv/read), total size, chunk size, tag, and connection pointer.
  - A vector of `ChunkState` entries, each holding channel index, offsets, request handle, and CUDA event for Stage 2.
  - State machines per chunk (`kIdle`, `kPosted`, `kKernelPending`, `kKernelDone`, `kReleased`).
- Recycle per-connection transfer objects similarly to `TcpxTransfer`: instantiate one state bundle per outstanding async call, but reuse scratch buffers/descriptor storage between calls to keep churn low.
- Build helper functions:
  - `select_channel_round_robin` to choose channels.
  - `post_chunk_send/recv` for Stage 1 submission (`tcpx_isend`/`tcpx_irecv` + descriptor build + unpack launch).
  - `advance_recv_stage1` to run `tcpx_test` and launch unpack kernels.
  - `advance_recv_stage2` to query events and call `tcpx_irecv_consumed`.
- For passive reads, reuse the same chunk machinery but set `needs_unpack` appropriately and trigger sends after FIFO notify.

Phase 4 – Sliding Window & Flow Control
- Introduce per-channel counters (send/recv inflight) with configurable limits. When a window hits the limit, block Stage 1 progress by draining Stage 2 (similar to `wait_for_channel_capacity` in the reference flow).
- Default limits: send inflight = 12, recv inflight = 16. Allow overrides via `UCCL_TCPX_MAX_SEND_INFLIGHT` / `UCCL_TCPX_MAX_RECV_INFLIGHT`.
- Use condition variables or spin-with-sleep loops to prevent busy-waiting while respecting RDMA-style nonblocking semantics.
- Emit debug logs summarizing inflight counts and chunk indices to ease troubleshooting.
- Use small helper lambdas to keep the refactor localized and avoid large structural rewrites; prefer augmenting existing methods over introducing new classes/files.

Phase 5 – API Implementation Updates
- `send_async/recv_async/read_async`: build the chunk vector up front, register transfer in `transfer_map_`, and immediately kick Stage 1 for the first window of chunks.
- `queue_read_response`: for each chunk derived from the requested size, push FIFO metadata with unique chunk IDs/tags so the reader can match responses; repurpose the 64-bit `token` field to pack `(chunk_idx << 32) | chunk_count` while keeping the rest of the `FifoItem` layout stable.
- `advertise`: include enough metadata (MR ID, base offset, tag) for the remote to reconstruct chunk boundaries.
- `poll_async`: iterate over active transfers, progress Stage 1 and Stage 2 for all channels, and mark transfer done only when every chunk reaches `kReleased`. Return `is_done=true` only at that point.
- Emit DEBUG-level counters after each polling sweep (`chunks_posted`, `chunks_inflight`, `chunks_completed`) to aid benchmarking.

Phase 6 – Validation & Tooling
- Update `p2p/tests/tcpx_smoke.cc` to cover large transfers and vector reads to confirm pipeline steps.
- Re-run `p2p/benchmarks/benchmark_nixl.py --backend tcpx` and capture logs (`p2p/server.log`, `p2p/client.log`) confirming no hangs at 64 MiB+.
- Adjust logging verbosity gates (`UCCL_TCPX_DEBUG`) to avoid excessive output in production runs.

Validation Checklist
- Run `p2p/tests/tcpx_smoke --mode {server,client} --size 67108864`.
- Run two-node benchmark: `PYTHONUNBUFFERED=1 python p2p/benchmarks/benchmark_nixl.py --backend tcpx ...`.
- Inspect `p2p/{server,client}.log` for sliding-window drain messages and ensure no hangs at 64 MiB+.
