"""Execute the actual staging component with CPU endpoint/runtime doubles."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
STUB = r"""
#pragma once
#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_set>
#include <vector>
#include "device/cnrt/staging.h"
struct FifoItem { size_t size = 0; uint64_t addr = 0; };
struct Conn { struct { uint64_t peer_id = 0; } uccl_conn_id_; };
struct P2PMhandle {};
struct UcclRequest {};
struct SendConnection {};
inline bool inside_python = false;
inline int allocations = 0;
using CNaddr = uintptr_t;
constexpr int cnrtSuccess = 0, cnrtMemcpyDevToDev = 0,
    cnrtMemcpyDevToHost = 1, cnrtMemcpyHostToDev = 2;
inline int cnrtSetDevice(int) { return 0; }
inline int cnrtMalloc(void** p, size_t n) {
  *p = new char[n]; ++allocations; return 0;
}
inline void cnFree(CNaddr p) { delete[] reinterpret_cast<char*>(p); --allocations; }
inline int cnrtMemcpy(void* d, void const* s, size_t n, int) {
  std::memcpy(d, s, n); return 0;
}
inline int cnrtMemset(void* p, int value, size_t n) { std::memset(p, value, n); return 0; }
class Endpoint {
 public:
  int local_gpu_idx_ = 0, ep_ = 0, fail_registration = 0, registration_calls = 0;
  uint64_t next_mr = 0;
  std::unordered_set<uint64_t> registered;
  Conn connection;
  P2PMhandle memory;
  bool reg(void*, size_t, uint64_t& id) {
    if (++registration_calls == fail_registration) return false;
    id = ++next_mr; registered.insert(id); return true;
  }
  void dereg(uint64_t id) { assert(registered.erase(id) == 1); }
  Conn* get_conn(uint64_t) { return &connection; }
  P2PMhandle* get_mhandle(uint64_t id) { return registered.count(id) ? &memory : nullptr; }
};
inline SendConnection* uccl_resolve_send_group(int, uint64_t) { return nullptr; }
inline int uccl_write_async_on_group(SendConnection*, Conn*, P2PMhandle*, void*,
                                    size_t, FifoItem const&, UcclRequest*) { return 0; }
inline int uccl_write_async(int, Conn*, P2PMhandle*, void*, size_t,
                           FifoItem const&, UcclRequest*) { return 0; }
inline int uccl_read_async(int, Conn*, P2PMhandle*, void*, size_t,
                          FifoItem const&, UcclRequest*) { return 0; }
inline bool uccl_poll_ureq_once(int, UcclRequest*) { return true; }
inline void uccl_drive_recv(int) {}
inline int prepare_fifo_metadata(int, P2PMhandle*, void*, size_t, char*) { return 0; }
#define UCCL_LOG(...) std::cerr
using PyGILState_STATE = int;
inline int Py_IsInitialized() { return 1; }
inline int PyGILState_Ensure() { return 0; }
inline int PyErr_CheckSignals() { return 0; }
inline void PyGILState_Release(int) {}
"""
CHECK = r"""
#include "engine.h"
int main() {
  for (int fail : {0, 1, 2}) {
    Endpoint ep;
    ep.fail_registration = fail;
    {
      uccl::device::NetworkStaging stage(ep);
      // A zero-length send exercises sender initialization without a peer.
      bool ok = stage.write(0, nullptr, 0, FifoItem{});
      assert(ok == (fail == 0));
      if (fail) {
        assert(allocations == 0 && ep.registered.empty());
        assert(stage.write(0, nullptr, 0, FifoItem{})); // retry after failure
      }
      stage.shutdown();
      stage.shutdown(); // explicit shutdown and destructor must be idempotent
      assert(allocations == 0 && ep.registered.empty());
    }
  }
  for (int fail : {0, 1}) {
    Endpoint ep;
    ep.fail_registration = fail;
    {
      uccl::device::NetworkStaging stage(ep);
      char output[64]{};
      char buffer[8]{};
      assert(stage.advertise(0, buffer, sizeof(buffer), output) == (fail == 0));
      stage.shutdown(); // joins the receiver poller before releasing memory
      assert(allocations == 0 && ep.registered.empty());
    }
  }
}
"""


class CnrtStagingTests(unittest.TestCase):
    def test_shutdown_and_failed_registration_release_owned_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for header in (
                "engine.h",
                "endpoint_wrapper.h",
                "util/debug.h",
                "util/gpu_rt.h",
                "util/util.h",
                "Python.h",
            ):
                path = directory / header
                path.parent.mkdir(exist_ok=True)
                path.write_text(
                    STUB if header == "engine.h" else '#include "engine.h"\n'
                )
            source = directory / "test.cc"
            source.write_text(CHECK)
            exe = directory / "test"
            command = [
                os.environ.get("CXX", "c++"),
                "-std=c++17",
                "-pthread",
                "-I",
                str(directory),
                "-I",
                str(ROOT / "p2p"),
                str(ROOT / "p2p/device/cnrt/staging.cc"),
                str(source),
                "-o",
                str(exe),
            ]
            result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [str(exe)], text=True, capture_output=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
