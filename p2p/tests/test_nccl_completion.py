"""Exercise the real NCCL adapter with delayed, independently owned completions."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

STUBS = r"""
#pragma once
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
enum ReqType { ReqRead, ReqWrite };
REQUEST
enum class NcclReqType { ReqRead, ReqWrite };
struct NcclRequest {
  NcclReqType type;
  uint32_t peer_id;
  void* context;
  uint32_t engine_idx;
};
struct ConnID { void* context; int sock_fd, dev; uint64_t peer_id; };
using NcclConnID = ConnID;
struct Conn { ConnID uccl_conn_id_{}; };
struct P2PMhandle {};
struct CompressCtx {};
struct SendConnection {};
struct NcclFlow {};
struct NotifyMsg {};
struct FifoItem { uint64_t addr; uint32_t size; char padding[28]; };
using NcclFifoItem = FifoItem;
inline int get_numa_node_from_iface() { return 0; }
template <typename T> struct EndpointOps;

struct Event { bool ready = false; };
using gpuEvent_t = Event*;
using gpuError_t = int;
constexpr int gpuSuccess = 0, gpuErrorNotReady = 1;
inline int released = 0;
inline int gpuEventQuery(Event* event) {
  return event->ready ? gpuSuccess : gpuErrorNotReady;
}
inline void gpuEventDestroy(Event* event) { delete event; ++released; }
inline char const* gpuGetErrorString(int) { return "unexpected GPU error"; }
struct NCCLEndpoint {
  struct Conn { std::atomic<uint64_t> writes_completed{0}; };
  Conn connection;
  uint64_t next_write = 0;
  ASYNC_HANDLE
  int gpu_index();
  ConnID uccl_connect(int, int, int, int, std::string const&, uint16_t);
  ConnID uccl_accept(int, int, int, std::string&, int*, int*);
  uint16_t get_p2p_listen_port(int);
  int get_p2p_listen_fd(int);
  bool initialize_engine_by_dev(int, bool);
  int send_notification(uint64_t, NotifyMsg const&);
  int prepare_fifo_metadata(void*, void*, void const*, size_t, char*);
  int uccl_read_async(NcclFlow*, void*, void*, size_t,
                      NcclFifoItem const&, NcclRequest* request) {
    assert(!request->context);
    auto* state = new AsyncHandle;
    state->event = new Event;
    request->context = state;
    return 0;
  }
  int uccl_write_async(NcclFlow* flow, void* mr, void* data, size_t size,
                       NcclFifoItem const& item, NcclRequest* request) {
    uccl_read_async(flow, mr, data, size, item, request);
    auto* state = static_cast<AsyncHandle*>(request->context);
    state->conn = &connection;
    state->write_seq = ++next_write;
    return 0;
  }
  bool uccl_poll_ureq_once(NcclRequest* request);
};
POLL
"""

CHECKS = r"""
#include "nccl_ops.h"
int main() {
  auto endpoint = std::make_shared<NCCLEndpoint>();
  using Ops = EndpointOps<NCCLEndpoint>;
  Conn conn{};
  FifoItem item{};
  UcclRequest read{}, write{};
  assert(Ops::uccl_read_async(endpoint, &conn, nullptr, nullptr, 64, item, &read) == 0);
  assert(Ops::uccl_write_async(endpoint, &conn, nullptr, nullptr, 64, item, &write) == 0);
  assert(read.context && write.context && read.context != write.context);
  for (int i = 0; i < 3; ++i) {
    assert(!Ops::uccl_poll_ureq_once(endpoint, &read));
    assert(!Ops::uccl_check_ureq_once(endpoint, &write));
    assert(released == 0);
  }
  auto* write_state = static_cast<NCCLEndpoint::AsyncHandle*>(write.context);
  write_state->event->ready = true;
  // A local send event alone must not complete a one-sided write.
  assert(!Ops::uccl_check_ureq_once(endpoint, &write));
  assert(released == 0);
  endpoint->connection.writes_completed.store(1);
  assert(Ops::uccl_check_ureq_once(endpoint, &write));
  assert(write.context == nullptr && released == 1);
  assert(!Ops::uccl_poll_ureq_once(endpoint, &read));
  static_cast<NCCLEndpoint::AsyncHandle*>(read.context)->event->ready = true;
  assert(Ops::uccl_poll_ureq_once(endpoint, &read));
  assert(read.context == nullptr && released == 2);
  // Cleared handles must not be destroyed again, and a slot can be reused.
  assert(Ops::uccl_poll_ureq_once(endpoint, &read));
  assert(released == 2);
  assert(Ops::uccl_read_async(endpoint, &conn, nullptr, nullptr, 64, item, &read) == 0);
  assert(!Ops::uccl_poll_ureq_once(endpoint, &read));
  static_cast<NCCLEndpoint::AsyncHandle*>(read.context)->event->ready = true;
  assert(Ops::uccl_poll_ureq_once(endpoint, &read));
  assert(read.context == nullptr && released == 3);
  // A prior write's ACK cannot complete the next write; an ACK also cannot
  // replace the local event, which protects the source buffer's lifetime.
  assert(Ops::uccl_write_async(endpoint, &conn, nullptr, nullptr, 64, item, &write) == 0);
  write_state = static_cast<NCCLEndpoint::AsyncHandle*>(write.context);
  write_state->event->ready = true;
  assert(!Ops::uccl_check_ureq_once(endpoint, &write));
  write_state->event->ready = false;
  endpoint->connection.writes_completed.store(2);
  assert(!Ops::uccl_check_ureq_once(endpoint, &write));
  write_state->event->ready = true;
  assert(Ops::uccl_check_ureq_once(endpoint, &write));
  assert(write.context == nullptr && released == 4);
}
"""


class NcclCompletionTests(unittest.TestCase):
    def test_pending_handles_survive_polling_and_are_released_once(self):
        engine = (ROOT / "p2p/engine.h").read_text()
        start = engine.index("struct UcclRequest {")
        request = engine[start : engine.index("};", start) + 2]
        native = (ROOT / "p2p/nccl/nccl_endpoint.cc").read_text()
        start = native.index("struct NCCLEndpoint::AsyncHandle {")
        handle = native[start : native.index("};", start) + 2]
        handle = handle.replace("NCCLEndpoint::", "")
        start = native.index("bool NCCLEndpoint::uccl_poll_ureq_once(")
        poll = native[start : native.index("\n}\n", start) + 3]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "transport").mkdir()
            (directory / "transport/endpoint_ops.h").write_text(
                STUBS.replace("REQUEST", request)
                .replace("ASYNC_HANDLE", handle)
                .replace("POLL", poll)
            )
            shutil.copy2(ROOT / "p2p/transport/nccl_ops.h", directory)
            source = directory / "check.cc"
            source.write_text(CHECKS)
            executable = directory / "check"
            result = subprocess.run(
                [
                    os.environ.get("CXX", "c++"),
                    "-std=c++17",
                    str(source),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [str(executable)], capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
