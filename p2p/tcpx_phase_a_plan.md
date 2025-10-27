# Phase A – TCPX Endpoint Parity Implementation Plan

## Goal
Bring the `tcpx::Endpoint` implementation to feature parity with the RDMA `Endpoint` used by `uccl_engine`, reusing the existing TCPX C wrappers and keeping code churn minimal.

## File-Level Changes

1. `p2p/tcpx_engine.h` (≈100 LoC)
   - Tighten the `Conn` struct (store both send and recv comm handles if needed) while keeping the existing ring/unpacker fields required for bounce-buffer draining.
   - Introduce lightweight metadata/MR/transfer bookkeeping structs plus atomics for `next_mr_id_` and `next_transfer_id_`.
   - Add containers for registered memory (`std::unordered_map<uint64_t, MrEntry>`) and inflight transfers (`std::unordered_map<uint64_t, PendingTransfer>`), protected by a mutex.
   - Declare helper methods for connection lookup, lazy MR registration, and a `queue_read_response` helper that `uccl_engine` can call after advertising a read.

2. `p2p/tcpx_engine.cc` (≈280 LoC)
   - Fix current build blockers (missing headers, semicolons, incorrect `static` on member definitions, dangling `stop_` usage).
   - Define a plain `struct EndpointInfo { char ip[INET_ADDRSTRLEN]; uint16_t port; int gpu; };` for metadata exchange; implement `get_metadata`/`parse_metadata` using this POD plus `send_ctrl_struct`/`recv_ctrl_struct`.
   - Complete `Endpoint::connect` and `Endpoint::accept` handshakes:
     - Exchange `EndpointInfo`, validate GPU ids, and store both TCP control socket and TCPX comm handles in the `Conn`.
   - Implement memory registration:
     - Assign monotonically increasing MR ids, call `tcpx_reg_mr` per connection on first use, and store `{size, ptr_type, mhandle}` in the MR map.
     - Provide `dereg` that walks all connection-specific handles and invokes `tcpx_dereg_mr`.
   - Implement `advertise` by populating `FifoItem` with `{addr=mr_id, size=len}` or an equivalent token understood by the TCPX read helper.
   - Implement async paths:
     - `send_async` → call `tcpx_isend`, stash returned request pointer in `PendingTransfer` with type `Send`.
     - `recv_async`/`read_async` → post `tcpx_irecv` (for read, treat `slot_item.addr` as MR id token + call `tcpx_isend` on the owner after we enqueue the receive).
     - `poll_async` → call `tcpx_test` on the stored request pointer and mark completion.
   - Keep the existing descriptor ring and unpacker thread: extend `unpacker_thread_func_` to populate `tcpx::rx::UnpackDescriptorBlock` via `buildDescriptorBlock`, launch the CUDA unpack kernel shipped under `p2p/tcpx/device`, and mark transfers complete once the kernel drains the bounce buffer into the destination GPU buffer.

3. `p2p/uccl_engine.cc` (≈40 LoC guarded by `#ifdef USE_TCPX`)
   - Adjust `listener_thread_func` and the `UCCL_READ`/`UCCL_FIFO` handling so that the TCPX path expects `FifoItem.addr` to carry an MR id token rather than an RDMA descriptor.
   - Ensure we trigger a data push from the provider after responding to a read request (e.g., call a new `endpoint->queue_read_response` helper that uses `send_async`).

4. `p2p/Makefile` (≈10 LoC)
   - Link against the TCPX plugin library (e.g., `-ltcpx_plugin` or provided `.so`), add include path for `p2p/tcpx/include`, and drop unused RDMA-only flags when `USE_TCPX=1`.

## Validation Steps
- Build with `USE_TCPX=1` (`make USE_TCPX=1 libuccl_engine.so`) to confirm the new endpoint compiles and links.
- Run an existing single-node unit test that exercises `uccl_engine` over loopback; add a simple smoke test if needed (Phase A scope).
- Add a focused GPU unpack test: post a receive, ensure descriptors trigger the CUDA unpack kernel, and verify data lands in the registered device buffer.
- Manual sanity: create two processes invoking `uccl_engine_connect`/`uccl_engine_accept` and verify send/recv and read flows complete without deadlocks.

## Notes
- We stick to the C-style wrappers in `p2p/tcpx/include/tcpx_interface.h`; the richer session/transfer managers remain unused for Phase A.
- The static gVNIC binding stays as currently coded; no dynamic NIC probing or extra logging is introduced.
