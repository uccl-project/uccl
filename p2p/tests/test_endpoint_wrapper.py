"""Execute the wrapper helpers with transport doubles, without Linux/GPU SDKs."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
STUBS = r"""
#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>
struct NotifyMsg { int value; };
struct RDMAEndpoint {};
struct NCCLEndpoint {
  int calls = 0;
  int send_notification(uint64_t peer, NotifyMsg const& message) {
    assert(peer == 17 && message.value == 23);
    ++calls;
    return 42;
  }
};
struct CxiEndpoint {
  int calls = 0;
  int send_notification(uint64_t peer, NotifyMsg const& message) {
    assert(peer == 17 && message.value == 23);
    ++calls;
    return -7;
  }
};
#if defined(UCCL_USE_MUSA)
using GenericEndpoint = std::variant<std::shared_ptr<RDMAEndpoint>>;
#else
using GenericEndpoint = std::variant<std::shared_ptr<RDMAEndpoint>,
    std::shared_ptr<NCCLEndpoint>, std::shared_ptr<CxiEndpoint>>;
#endif
int get_numa_node_from_iface() { return 9; }
struct RdmaDeviceManager {
  static RdmaDeviceManager& instance() {
    static RdmaDeviceManager manager;
    return manager;
  }
  std::vector<int> get_best_dev_idx(int gpu) { return {gpu + 10}; }
  int get_numa_node(int device) { return device + 20; }
};
"""
CHECKS = r"""
int main() {
  GenericEndpoint rdma = std::make_shared<RDMAEndpoint>();
  assert(uccl_get_numa_node(rdma, 3) == 33);
  assert(uccl_send_notification(rdma, 17, NotifyMsg{23}) == -1);
  GenericEndpoint empty_rdma = std::shared_ptr<RDMAEndpoint>{};
  assert(uccl_send_notification(empty_rdma, 17, NotifyMsg{23}) == -1);
#if !defined(UCCL_USE_MUSA)
  auto nccl = std::make_shared<NCCLEndpoint>();
  auto cxi = std::make_shared<CxiEndpoint>();
  assert(uccl_get_numa_node(GenericEndpoint{nccl}, 3) == 9);
  assert(uccl_get_numa_node(GenericEndpoint{cxi}, 3) == 9);
  assert(uccl_send_notification(GenericEndpoint{nccl}, 17, NotifyMsg{23}) == 42);
  assert(uccl_send_notification(GenericEndpoint{cxi}, 17, NotifyMsg{23}) == -7);
  assert(nccl->calls == 1 && cxi->calls == 1);
  GenericEndpoint empty_nccl = std::shared_ptr<NCCLEndpoint>{};
  GenericEndpoint empty_cxi = std::shared_ptr<CxiEndpoint>{};
  assert(uccl_send_notification(empty_nccl, 17, NotifyMsg{23}) == -1);
  assert(uccl_send_notification(empty_cxi, 17, NotifyMsg{23}) == -1);
#endif
}
"""


class EndpointWrapperTests(unittest.TestCase):
    def test_numa_and_notification_dispatch(self):
        # Compile the production helper bodies, not a reimplementation.
        wrapper = (ROOT / "p2p/endpoint_wrapper.h").read_text()
        start = wrapper.index("inline int uccl_get_numa_node(")
        stop = wrapper.index("inline bool initialize_rdma_ctx_for_gpu(", start)
        adapters = "template <typename T> struct EndpointOps;\n"
        for name, typ in (
            ("rdma", "RDMAEndpoint"),
            ("nccl", "NCCLEndpoint"),
            ("cxi", "CxiEndpoint"),
        ):
            text = (ROOT / "p2p/transport" / f"{name}_ops.h").read_text()
            methods = ""
            for method in ("uccl_get_numa_node", "uccl_send_notification"):
                first = text.index(f"static int {method}(")
                opening = text.index("{", first)
                depth = 1
                end = opening + 1
                while depth:
                    depth += (text[end] == "{") - (text[end] == "}")
                    end += 1
                methods += text[first:end] + "\n"
            adapters += f"template <> struct EndpointOps<{typ}> {{\n{methods}}};\n"
        source = STUBS + adapters + wrapper[start:stop] + CHECKS
        compiler = os.environ.get("CXX", "c++")
        self.assertIsNotNone(shutil.which(compiler))
        for musa in (False, True):
            with self.subTest(musa=musa), tempfile.TemporaryDirectory() as directory:
                exe = str(Path(directory) / "test")
                command = [compiler, "-std=c++17", "-x", "c++", "-", "-o", exe]
                if musa:
                    command.append("-DUCCL_USE_MUSA")
                result = subprocess.run(
                    command, input=source, text=True, capture_output=True, timeout=30
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run(
                    [exe], text=True, capture_output=True, timeout=10
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
