"""Backend selection/transport validation must not need any vendor SDK."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r"""
#include "util/transport_type.h"
#include <cassert>
#include <string>
int main() {
  using B = uccl::device::Backend;
  assert(std::string(B::name) == EXPECT_NAME);
  static_assert(UCCL_DEVICE_HAS_DMABUF == EXPECT_DMABUF);
  assert(B::staged_network == EXPECT_STAGING);
  assert(transport_available(TransportType::RDMA));
  assert(transport_available(TransportType::NCCL) == bool(UCCL_P2P_ENABLE_NCCL));
  try {
    (void)get_transport_type();
    return EXPECT_REJECT ? 1 : 0;
  } catch (std::invalid_argument const&) {
    return EXPECT_REJECT ? 0 : 1;
  }
}
"""


class BackendConfigTests(unittest.TestCase):
    def test_sdk_free_config_and_unsupported_transport_validation(self):
        for backend, extra, dmabuf, staging, reject in (
            ("cuda", [], 1, 0, 0),
            ("hip", [], 1, 0, 0),
            ("hip", ["-D__UCCL_DTK__"], 0, 0, 0),
            ("cnrt", [], 1, 1, 0),
            ("musa", [], 0, 0, 1),
            ("cuda", ["-DUCCL_P2P_ENABLE_CXI=0"], 1, 0, 1),
        ):
            with (
                self.subTest(backend=backend, extra=extra),
                tempfile.TemporaryDirectory() as directory,
            ):
                exe = str(Path(directory) / "test")
                command = [
                    os.environ.get("CXX", "c++"),
                    "-std=c++17",
                    "-I",
                    str(ROOT / "include"),
                    "-I",
                    str(ROOT / "p2p"),
                    f'-DUCCL_DEVICE_BACKEND_HEADER="util/device/{backend}/config.h"',
                    f'-DEXPECT_NAME="{backend}"',
                    f"-DEXPECT_DMABUF={dmabuf}",
                    f"-DEXPECT_STAGING={staging}",
                    f"-DEXPECT_REJECT={reject}",
                    *extra,
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    exe,
                ]
                result = subprocess.run(
                    command, input=SOURCE, capture_output=True, text=True, timeout=30
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run(
                    [exe],
                    env=dict(os.environ, UCCL_P2P_TRANSPORT="cxi"),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
