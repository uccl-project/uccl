"""Compile the factory against endpoint doubles; no GPU or NIC is required."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ENDPOINTS = r"""
#pragma once
#include <memory>
#include <variant>
struct RDMAEndpoint {
  int gpu, device;
  RDMAEndpoint(int g, int d) : gpu(g), device(d) {}
};
#if defined(UCCL_USE_MUSA)
using GenericEndpoint = std::variant<std::shared_ptr<RDMAEndpoint>>;
#else
struct NCCLEndpoint : RDMAEndpoint { using RDMAEndpoint::RDMAEndpoint; };
struct CxiEndpoint : RDMAEndpoint { using RDMAEndpoint::RDMAEndpoint; };
using GenericEndpoint = std::variant<std::shared_ptr<RDMAEndpoint>,
    std::shared_ptr<NCCLEndpoint>, std::shared_ptr<CxiEndpoint>>;
#endif
"""
NUMA = r"""
#include <iostream>
#include <vector>
struct UnifiedTask {};
constexpr int kTaskRingSize = 1024;
namespace uccl { inline void* create_ring(size_t, int) { return nullptr; } }
int get_numa_node_from_iface() { return 9; }
struct RdmaDeviceManager {
  static RdmaDeviceManager& instance() {
    static RdmaDeviceManager manager;
    return manager;
  }
  std::vector<int> get_best_dev_idx(int gpu) { return {gpu + 10}; }
  int get_numa_node(int device) { return device + 20; }
};
int uccl_get_numa_node(GenericEndpoint const& ep, int gpu_idx) {
  return std::visit([&](auto const& endpoint) -> int {
    using T = std::decay_t<decltype(*endpoint)>;
#if defined(UCCL_USE_MUSA)
    (void)sizeof(T);
    return RdmaDeviceManager::instance().get_numa_node(
        RdmaDeviceManager::instance().get_best_dev_idx(gpu_idx)[0]);
#else
    if constexpr (std::is_same_v<T, NCCLEndpoint> ||
                  std::is_same_v<T, CxiEndpoint>)
      return get_numa_node_from_iface();
    else
      return RdmaDeviceManager::instance().get_numa_node(
          RdmaDeviceManager::instance().get_best_dev_idx(gpu_idx)[0]);
#endif
  }, ep);
}
"""


class TransportFactoryTests(unittest.TestCase):
    def check_program(self, source, musa=False):
        compiler = os.environ.get("CXX", "c++")
        self.assertIsNotNone(shutil.which(compiler))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copy2(ROOT / "p2p/transport_factory.h", directory)
            (directory / "engine.h").write_text(ENDPOINTS)
            exe = directory / "test"
            command = [
                compiler,
                "-std=c++17",
                "-I",
                str(directory),
                "-I",
                str(ROOT / "p2p/util"),
                "-I",
                str(ROOT / "p2p"),
                "-I",
                str(ROOT / "include"),
                "-x",
                "c++",
                "-",
                "-o",
                str(exe),
            ]
            if musa:
                command.append("-DUCCL_USE_MUSA")
            result = subprocess.run(
                command, input=source, text=True, capture_output=True, timeout=30
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            transports = (
                ("rdma", "ib")
                if musa
                else ("rdma", "ib", "efa", "nccl", "tcp", "tcpx", "cxi", "unknown")
            )
            for transport in (None, *transports):
                with self.subTest(musa=musa, transport=transport):
                    env = dict(os.environ)
                    env.pop("UCCL_P2P_TRANSPORT", None)
                    if transport is not None:
                        env["UCCL_P2P_TRANSPORT"] = transport
                    result = subprocess.run(
                        [str(exe)], env=env, text=True, capture_output=True, timeout=10
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_factory_selects_type_and_preserves_eager_and_lazy_arguments(self):
        source = r"""
#include <cassert>
#include "transport_factory.h"
int main() {
  for (int gpu : {-1, 3}) {
    auto ep = uccl_create_endpoint(gpu);
    std::visit([&](auto const& endpoint) {
      assert(endpoint && endpoint->gpu == gpu && endpoint->device == 0);
    }, ep);
#if defined(UCCL_USE_MUSA)
    assert(std::holds_alternative<std::shared_ptr<RDMAEndpoint>>(ep));
#else
    if (is_nccl_transport())
      assert(std::holds_alternative<std::shared_ptr<NCCLEndpoint>>(ep));
    else if (is_cxi_transport())
      assert(std::holds_alternative<std::shared_ptr<CxiEndpoint>>(ep));
    else
      assert(std::holds_alternative<std::shared_ptr<RDMAEndpoint>>(ep));
#endif
  }
}
"""
        for musa in (False, True):
            self.check_program(source, musa)

    def test_explicit_constructor_initializes_numa_before_starting_threads(self):
        # Execute the actual constructor block with OS/GPU operations replaced.
        engine = (ROOT / "p2p/engine.cc").read_text()
        start = engine.index("  ep_ = uccl_create_endpoint(local_gpu_idx_);")
        stop = engine.index("  send_proxy_thread_ =", start)
        block = engine[start:stop]
        source = (
            '#include <cassert>\n#include <type_traits>\n#include "transport_factory.h"\n'
            + NUMA
            + """
int main() {
  int local_gpu_idx_ = 3;
  int numa_node_ = -1234;
  bool engine_initialized_ = false;
  GenericEndpoint ep_;
  void* send_unified_task_ring_ = nullptr;
  void* recv_unified_task_ring_ = nullptr;
"""
            + block
            + """
  assert(engine_initialized_);
  assert(numa_node_ == ((is_nccl_transport() || is_cxi_transport()) ? 9 : 33));
}
"""
        )
        for musa in (False, True):
            self.check_program(source, musa)

    def test_connect_uses_peer_id_instead_of_transport_specific_socket(self):
        engine = (ROOT / "p2p/engine.cc").read_text()
        start = engine.index("bool Endpoint::connect(")
        connect = engine[start : engine.index("\n}\n", start) + 3]
        common = (ROOT / "p2p/util/common.h").read_text()
        start = common.index("struct ConnID {")
        conn_id = common[start : common.index("};", start) + 2]
        source = (
            r"""
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
using PeerID = uint64_t;
"""
            + conn_id
            + r"""
constexpr int INVALID_GPU = -1;
bool inside_python = false;
void check_python_signals() { assert(false); }
struct Conn {
  uint64_t id;
  ConnID transport;
  std::string ip;
  uint16_t port;
  std::string extra;
};
ConnID next_result{};
ConnID uccl_connect(int, int, std::string const&, int) { return next_result; }
struct Endpoint {
  int local_gpu_idx_ = 0, ep_ = 0;
  std::atomic<uint64_t> next_conn_id_{0};
  std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
  bool connect(std::string, int, int, uint64_t&);
  ~Endpoint() {
    for (auto const& item : conn_id_to_conn_) delete item.second;
  }
};
"""
            + connect
            + r"""
int main() {
  Endpoint endpoint;
  // RDMA identifies its first peer as zero and encodes it as a null context.
  // Its socket field is not a connection-validity indicator. A negative value
  // deterministically exposes the regression without reading uninitialized data.
  for (int fd : {-12345, -1, 0, 42}) {
    next_result = {nullptr, fd, 0, 0};
    uint64_t id = UINT64_MAX;
    assert(endpoint.connect("127.0.0.1", 0, 1234, id));
    auto* saved = endpoint.conn_id_to_conn_.at(id);
    assert(saved->transport.peer_id == 0);
    assert(saved->port == 1234);
  }
  // NCCL/CXI rejection uses the invalid peer ID. Never publish a failed
  // connection, even if its socket field happens to be nonnegative.
  auto count = endpoint.conn_id_to_conn_.size();
  for (int fd : {-1, 0, 42}) {
    next_result = {nullptr, fd, -1, UINT64_MAX};
    uint64_t id = UINT64_MAX;
    assert(!endpoint.connect("127.0.0.1", 0, 1234, id));
    assert(endpoint.conn_id_to_conn_.size() == count);
    assert(endpoint.conn_id_to_conn_.count(id) == 0);
  }
}
"""
        )
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "test"
            result = subprocess.run(
                [
                    os.environ.get("CXX", "c++"),
                    "-std=c++17",
                    "-pthread",
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(exe),
                ],
                input=source,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [str(exe)], capture_output=True, text=True, timeout=20
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
