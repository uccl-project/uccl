Project Memory (handover notes)

Scope
- Repository root: uccl-dev
- This document captures high‑signal context for the next engineer (goals, key code paths, decisions, current state). Environment minutiae are intentionally kept minimal.

Goal
- Implement and run a NIXL transport backend over Google GPUDirectTCPX (TCPX) on GCP A3‑high (2 nodes, 8×H100, multi gVNIC).
- Reuse the existing UCCL P2P engine abstraction and port the RDMA plugin shape to a TCPX plugin so NIXL collectives/benchmarks can run with TCPX as the backend.

Key Components
- TCPX engine (Phase A)
  - Files: p2p/tcpx_engine.h, p2p/tcpx_engine.cc
  - Provides Endpoint abstraction compatible with the UCCL C API shim in p2p/uccl_engine.{h,cc}
  - Highlights:
    - Control handshake over a TCP control socket with handle exchange (four stages, interleaved).
    - Dual TCPX comms (send_comm/recv_comm) per connection; tags used to match isend/irecv.
    - Bounce‑buffer receive path with CUDA unpack kernels. FifoItem is 64 bytes (mr_id, size, tag, offset, token, padding).
    - Robust control socket binding with retries (UCCL_TCPX_PORT_RETRIES), SO_REUSE{ADDR,PORT}, and detailed logs.

- UCCL engine shim
  - Files: p2p/uccl_engine.h, p2p/uccl_engine.cc
  - C API that NIXL loads. It forwards to Endpoint when compiled with USE_TCPX=1 and implements metadata exchange + listener thread that services FIFO/notify messages.

- NIXL TCPX plugin
  - Dir: thirdparty/nixl/src/plugins/tcpx
  - Files: tcpx_backend.{h,cpp}, tcpx_plugin.cpp, meson.build
  - Mirrors UCCL RDMA plugin shape. Creates a UCCL engine instance (backed by TCPX), registers memory, posts/read/write calls via the shim.
  - getConnInfo returns a connection string (ip:port?gpu) using uccl_engine_get_metadata; has fallbacks and extra logging.

- Benchmarks
  - Primary: p2p/benchmarks/benchmark_nixl.py
  - Extended to support backend=tcpx and to print explicit server‑side ZMQ checkpoints.
  - Progress thread explicitly disabled in config for TCPX (nixl_agent_config(False, False, 0)).

What was moved out of NCCL into this tree
- TCPX bounce‑buffer unpack logic (kernel and launcher) is integrated under p2p/tcpx/device and used by Endpoint on the recv path.

Critical Protocol/Struct Details
- Control plane
  - EndpointInfo (IPv4 string, port, gpu): exchanged over TCP control socket; used to seed handle exchange.
  - CtrlMsgHeader (type/flags/length) for framing struct payloads (CTRL_STRUCT, CTRL_ACK).
- FIFO item (FifoItem, 64 bytes, alignment preserved):
  - mr_id (u64), size (u32), tag (u32), offset (u64), token (u64, reserved), padding[32].
  - Used to advertise remote memory slice and tag, then active peer drives queue_read_response.

Current State
- TCPX plugin loads and engine binds control socket with logs like:
  - [TCPX‑plugin] init …
  - [tcpx] control socket bound on port …
  - [NIXL‑core] createBackend('TCPX'): getConnInfo() -> 0, str='ip:port?gpu'
- Server benchmark reaches “[server] waiting for remote metadata on ZMQ…]”. Start the client on the second node with the same environment to proceed.

Key Decisions / Fixes
- Avoid NIXL progress thread for TCPX: nixl_agent_config(False, False, 0) to prevent NOT_SUPPORTED paths (etcd/socket metadata).
- getConnInfo hardened: uses uccl_engine_get_metadata when available; falls back to env/device IP; never fails hard for lack of IP in dev setups.
- Control‑socket bind: added SO_REUSEPORT, port range retries via UCCL_TCPX_PORT_RETRIES, and explicit stderr logs on each attempt.
- Only load TCPX plugin during dev to sidestep UCX symbol dependencies: set NIXL_PLUGIN_DIR to the TCPX plugin directory.
- Removed stray pdb set_trace from the server path; retained concise prints to clarify ZMQ handshake points.

Runbook (two nodes)
- Build TCPX engine (libuccl_engine.so):
  - cd p2p; make USE_TCPX=1 -j; sudo install -m 0755 libuccl_engine.so /usr/local/lib/; sudo ldconfig
  - Or: sudo make USE_TCPX=1 install
- Build TCPX plugin and install Python package for NIXL (must use pip to ensure ABI match):
  - cd thirdparty/nixl; meson setup build --reconfigure; ninja -C build src/plugins/tcpx/libplugin_TCPX.so
  - pip uninstall -y nixl; pip install . --no-cache-dir --force-reinstall
  - export NIXL_PLUGIN_DIR=$(pwd)/build/src/plugins/tcpx
- Runtime env (both nodes; adjust NIC names):
  - export UCCL_TCPX_OOB_PORT=28900; export UCCL_TCPX_CHUNK_BYTES=4194304
  - export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0; export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="eth1,eth2,eth3,eth4"; export NCCL_SOCKET_IFNAME=eth0
  - export NCCL_NSOCKS_PERTHREAD=2; export NCCL_SOCKET_NTHREADS=1
  - export NCCL_MIN_ZCOPY_SIZE=4096; export NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE=4096; export NCCL_GPUDIRECTTCPX_RECV_SYNC=1
  - export NCCL_BUFFSIZE=8388608; export NCCL_ALGO=Ring; export NCCL_PROTO=Simple
  - export UCCL_TCPX_PORT_RETRIES=32; export UCCL_TCPX_LOCAL_DEVICE=<gpu_index>
  - export NIXL_LOG_LEVEL=DEBUG
- Launch:
  - Server (Node A): PYTHONUNBUFFERED=1 python benchmarks/benchmark_nixl.py --backend tcpx --role server --sizes 1048576 --iters 1
  - Client (Node B): PYTHONUNBUFFERED=1 python benchmarks/benchmark_nixl.py --backend tcpx --role client --remote-ip=<SERVER_IP> --sizes 1048576 --iters 1

Diagnostics / Self‑checks
- Confirm Python binding and linked libs:
  - python - <<'PY'\nimport nixl._bindings as b, os\nprint('bindings path:', b.__file__)\nos.system("ldd '{}'".format(b.__file__))\nPY
- Confirm TCPX plugin loaded: strings $NIXL_PLUGIN_DIR/libplugin_TCPX.so | head
- Confirm control port listening on server: ss -ltnp | grep $UCCL_TCPX_OOB_PORT

Notable Pitfalls
- Mixing a pip‑installed older nixl core with a newly built plugin/engine leads to NIXL_ERR_NOT_SUPPORTED or undefined symbols. Always pip uninstall/install after changing core, and prefer loading only the TCPX plugin during dev.
- UCX plugins in the same plugin dir can emit undefined reference errors if their deps aren’t present—point NIXL_PLUGIN_DIR to the TCPX dir only.

Key File Pointers
- TCPX engine: p2p/tcpx_engine.cc, p2p/tcpx_engine.h
- UCCL shim: p2p/uccl_engine.{h,cc}
- TCPX plugin: thirdparty/nixl/src/plugins/tcpx/*
- Benchmark: p2p/benchmarks/benchmark_nixl.py

Roadmap Toward TcpxSession/TcpxTransfer-Level Performance
- Current behaviour snapshot (single-request engine)
  - Registration: `uccl_engine_register_memory` maps one mr_id to the entire tensor; there is no pre-slicing.  
  - Submission: every `uccl_engine_write/read` issues exactly one `Endpoint::send_async/recv_async` covering the full `(addr,len)` from the descriptor; the plugin simply forwards the whole region.  
  - Request handles: `Endpoint::send_async` and `populate_conn_handles_` cache only one TCPX request pointer per transfer. The returned `transfer_id` is a scalar (monotonic counter).  
  - Receive path mirrors this: `recv_async` captures one `tcpx_irecv` request and a single `needs_unpack` flag.  
  - Bookkeeping: `PendingTransfer` holds `{request, dst_ptr, size, tag, cuda_event}` for that single segment; there is no vector of sub-requests.  
  - Progress engine: `poll_request_` invokes `tcpx_test` once per `transfer_id`. If the 64 MB request is still in flight, it immediately returns “not done” and higher layers spin.  
  - Completion: `complete_pending_transfer_` assumes one CUDA event and one `tcpx_irecv_consumed` call.  
  - FIFO/READ: listener thread sends one FIFO item per original descriptor, so the server posts exactly one `queue_read_response`. No chunk-level IDs exist.  
  - Result: with big payloads the engine serialises the entire transfer through a single TCPX request. Once the request stalls (e.g., device expects ≤4 MiB DMA-BUF), the client loops in `tcpx_test` and the server never observes completion.
- Known failure context (Nov 4 2025 repro)
  - `python benchmarks/benchmark_nixl.py --backend tcpx --role server --sizes 67108864 --iters 1`  
    - Server waits after **“[server] received signal: START”**; client emits repeated `[tcpx] tcpx_test transfer_id=1 completed=0 size=0`.  
    - Eventually client hits `tcpx_accept_v5 failed (client reverse)` followed by fatal `Check failed: addr.sa.sa_family == AF_INET6` inside the TCPX flow-steering helper (due to teardown chaos).  
  - 256 B WRITE (before chunk fix) failed with `ioctl get dma_buf frags: Inappropriate ioctl for device` and `tcpx_reg_mr failed rc=3`; this was addressed by aligning metadata port and verifying registration, but highlights the sensitivity to sub-4 KiB slices.
- Accepted downtime: OK to break large transfers while refactoring (`plugin chunking` can be skipped). Focus is on landing full engine upgrade.

Engine-level rework plan (single-channel pipeline first)
1. **Define chunk metadata**
   - Introduce a `ChunkTransfer` struct in `tcpx_engine.h` (`request`, `offset`, `length`, `needs_unpack`, `cuda_event`, `state` enum).  
   - Extend `PendingTransfer` to own `std::vector<ChunkTransfer>` plus counters for `{posted, completed}` chunks.
   - Generate deterministic chunk tags (`base_tag + chunk_idx`) to keep FIFO/READ consistent.
2. **Chunk-aware submission**
   - Refactor `send_async_with_tag` and `recv_async_with_tag` to:  
     - slice `[addr, size]` into `chunk_bytes = min(UCCL_TCPX_CHUNK_BYTES, default 4 MiB)` pieces;  
     - for each chunk, call `tcpx_isend/tcpx_irecv` with its own tag, store the resulting request handle in the vector;  
     - register chunk-specific metadata (offset, bytes) so unpack stage knows where to copy.  
   - Guard with an environment knob (e.g., `UCCL_TCPX_PIPELINE=1`) to ease bring-up.
3. **Stage 1 progress loop (`poll_request_`)**
   - Iterate over all chunk requests belonging to a transfer:  
     - Invoke `tcpx_test` for each pending chunk.  
     - For completed recv chunks: read TCPX metadata (unpack descriptors), prepare CUDA launch data, enqueue kernel via existing `enqueue_unpack_`. Attach/record CUDA event to the chunk slot.  
     - For completed send chunks: mark state `kSendDone`.  
   - Return “still in progress” if any chunk remains pending or waiting on GPU.
4. **Stage 2 completion & resource return**
   - Introduce helper `drain_chunk_completion(ChunkTransfer&)` that:  
     - Waits (`cudaEventQuery`/`cudaEventSynchronize`) for recv chunk events.  
     - Calls `tcpx_irecv_consumed` once per recv chunk to release bounce buffers.  
     - Marks chunk as `kDone`.  
   - Modify `poll_async` to loop until all chunk states reach `kDone`; only then erase the transfer from `transfer_map_`.
5. **READ FIFO integration**
   - Modify `listener_thread_func`’s `UCCL_VECTOR_READ` case: for each chunk produced by the server (based on requested size and chunk bytes), send a FIFO item with `fifo_data.id = chunk_index`.  
   - Ensure client `uccl_engine_get_fifo_item` + `queue_read_response` consumes the right chunk metadata by matching `(conn + fifo_data.id)`.  
   - Update LOG messages to include chunk index for easier debugging.
6. **Sliding window controls**
   - Add configurable limits (env `UCCL_TCPX_MAX_RECV_INFLIGHT`, `UCCL_TCPX_MAX_SEND_INFLIGHT`).  
   - `send_async`/`recv_async` should check inflight counters and, if the window is full, block (or return retry) until `poll_async` drains some chunks. A simple approach is a condition variable guarded by `transfer_mu_`.  
   - Provide metrics counters (chunks posted/completed) gated by DEBUG env.
7. **Multi-channel extension (final step)**
   - After the single-channel chunked pipeline is stable, port `ChannelManager` concepts:  
     - During connect/accept, create `N` TCPX comm pairs (controlled by `UCCL_TCPX_CHANNELS`); store them in the connection object.  
     - Modify chunk submission to pick a channel round-robin (or based on per-channel inflight). Each chunk tracks the channel it was posted on.  
     - Ensure deregistration/teardown iterates over all channel-specific comms and mhandles.

Handoff guidance for the next engineer
- Repro summary:  
  - 1 MiB transfer succeeds with current code; 64 MiB WRITE/READ hangs (client spins in `tcpx_test`, server idle after `START`).  
  - Environment: GCP A3-high (2 nodes, 8×H100), `UCCL_TCPX_OOB_PORT=28900`, `UCCL_TCPX_CHUNK_BYTES=4194304`, TCPX plugin directory set via `NIXL_PLUGIN_DIR`.  
  - Failure logs: see `p2p/server.log` / `p2p/client.log` entries dated `2025-11-04 00:36`.  
- Immediate priority: implement steps 1–6 above directly in `p2p/tcpx_engine.cc` / `p2p/uccl_engine.cc`. Plugin-level chunking is intentionally skipped since downtime is acceptable.  
- Testing checklist after each major change:  
  - `PYTHONUNBUFFERED=1 python benchmarks/benchmark_nixl.py --backend tcpx --role {server,client} --sizes 67108864 --iters 1`.  
  - `p2p/tests/tcpx_smoke --mode {server,client}` with large `--size`.  
  - Existing 1 MiB regression to ensure no small-transfer breakage.  
- Documentation/logging expectations: keep `tcpx_backend.cpp` INFO/DEBUG logs intact; add temporary verbose counters (guarded by env) to trace chunk scheduling during bring-up.  
- Long-term success criteria: end-to-end throughput similar to `p2p/tcpx/tests/test_tcpx_perf_multi_new` when using equivalent channel count and chunk size; absence of hangs for 64 MB–1 GB payloads; CPU/GPU utilization comparable to TcpxSession/TcpxTransfer pipeline.
