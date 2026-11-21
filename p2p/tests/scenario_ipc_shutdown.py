"""Exercise accepted IPC operations through poller shutdown."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

STUBS = r"""
#include "util/jring.h"
#include <atomic>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool value, char const* reason) {
  if (!value) throw std::runtime_error(reason);
}
namespace uccl { void pin_thread_to_numa(int) {} }
struct Event { int remaining = 0; bool destroyed = false; };
struct Mapping { int device = 0; bool closed = false; };
using gpuEvent_t = Event*;
constexpr int gpuErrorNotReady = 1;
int current_device = -1;
std::function<void()> on_query;
int gpuEventQuery(Event* event) {
  if (on_query) {
    auto action = std::move(on_query);
    on_query = {};
    action();
  }
  require(!event->destroyed, "queried a destroyed event");
  if (event->remaining > 0) { --event->remaining; return gpuErrorNotReady; }
  return 0;
}
int gpuEventDestroy(Event* event) {
  require(!event->destroyed && event->remaining == 0,
          "destroyed an unfinished or already destroyed event");
  event->destroyed = true;
  return 0;
}
int gpuSetDevice(int device) { current_device = device; return 0; }
int gpuIpcCloseMemHandle(void* pointer) {
  auto* mapping = static_cast<Mapping*>(pointer);
  require(!mapping->closed && mapping->device == current_device,
          "closed an IPC mapping twice or on the wrong device");
  mapping->closed = true;
  return 0;
}
#define GPU_RT_CHECK(result) require((result) == 0, "device call failed")
struct TransferStatus { std::atomic<bool> done{false}; };
struct Sleeper { void update_timer() {} void maybe_sleep() {} };
"""

ENDPOINT = r"""
class Endpoint {
 public:
  int numa_node_ = 0;
  std::atomic<bool> stop_{false};
  Sleeper ipc_proxy_adaptive_sleeper_;
  jring_t* ipc_inflight_ring_;
  Endpoint() {
    auto bytes = jring_get_buf_ring_size(sizeof(IpcInflightOp*), 16);
    ipc_inflight_ring_ = static_cast<jring_t*>(aligned_alloc(64, bytes));
    require(ipc_inflight_ring_, "could not allocate IPC inbox");
    require(jring_init(ipc_inflight_ring_, 16, sizeof(IpcInflightOp*), 1, 1) == 0,
            "could not initialize IPC inbox");
  }
  ~Endpoint() { free(ipc_inflight_ring_); }
  void enqueue(IpcInflightOp* op) {
    alignas(16) IpcInflightOp* slot[2] = {op, nullptr};
    require(jring_mp_enqueue_bulk(ipc_inflight_ring_, slot, 1, nullptr) == 1,
            "could not enqueue accepted IPC operation");
  }
  void ipc_poller_thread_func();
};
"""

SCENARIOS = r"""
int main() {
  int cases = 0;
  for (int round = 0; round < 32; ++round) {
    for (int mode = 0; mode < 6; ++mode) {
      Endpoint endpoint;
      TransferStatus first, second;
      Event a{mode == 3 ? 3 : 0}, b, c, d;
      Mapping one{1}, two{2};
      current_device = -1;
      if (mode != 0) {
        auto* op = new IpcInflightOp{};
        op->status = &first;
        if (mode != 4) op->events = {&a};
        if (mode == 2) {
          op->events = {&a, &b, &c, &d};
          op->raw_ptrs_v = {&one, nullptr, &two};
          op->gpu_idxs_v = {1, 0, 2};
        } else if (mode != 4) {
          op->raw_ptr = &one;
          op->gpu_idx = 1;
        }
        endpoint.enqueue(op);
      }
      if (mode == 5) {
        // A second submission finishes before shutdown while an older op is polled.
        on_query = [&] {
          auto* op = new IpcInflightOp{};
          op->status = &second;
          op->events = {&b};
          op->raw_ptr = &two;
          op->gpu_idx = 2;
          endpoint.enqueue(op);
          endpoint.stop_.store(true, std::memory_order_release);
        };
      } else {
        endpoint.stop_.store(true, std::memory_order_release);
      }
      endpoint.ipc_poller_thread_func();
      require(jring_count(endpoint.ipc_inflight_ring_) == 0,
              "shutdown left an accepted operation in the IPC inbox");
      if (mode != 0) require(first.done.load(), "first completion was not published");
      if (mode != 0 && mode != 4)
        require(a.destroyed && one.closed, "first operation resources leaked");
      if (mode == 2)
        require(b.destroyed && c.destroyed && d.destroyed && two.closed,
                "vector operation resources leaked");
      if (mode == 5)
        require(second.done.load() && b.destroyed && two.closed,
                "submission accepted before shutdown was dropped");
      ++cases;
    }
  }
  std::cout << cases << " IPC shutdown lifecycles passed\n";
}
"""


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "p2p" / "engine.cc")
    args = parser.parse_args()
    source = args.source.read_text()
    begin = source.index("void Endpoint::ipc_poller_thread_func()")
    end = source.index("\nstd::shared_ptr<UnifiedTask>", begin)
    header = (root / "p2p" / "engine.h").read_text()
    struct_begin = header.index("struct IpcInflightOp {")
    struct_end = header.index("\n};", struct_begin) + 3
    program = (
        STUBS
        + header[struct_begin:struct_end]
        + ENDPOINT
        + source[begin:end]
        + SCENARIOS
    )
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "ipc-shutdown"
        subprocess.run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++17",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                "-I",
                str(root / "include"),
                "-x",
                "c++",
                "-",
                "-o",
                str(executable),
            ],
            input=program,
            text=True,
            check=True,
            timeout=60,
        )
        subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    main()
