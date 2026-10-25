"""CPU-only tests of real headers and native build rules, not GPU certification."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
STUBS = Path(__file__).parent / "musa_stubs"


class MusaBackendTests(unittest.TestCase):
    def compile(self, text, *defines, run=True, env=None):
        compiler = os.environ.get("CXX", "c++")
        self.assertIsNotNone(shutil.which(compiler), "C++17 compiler required")
        with tempfile.TemporaryDirectory() as directory:
            exe = str(Path(directory) / "test")
            command = [
                compiler,
                "-std=c++17",
                "-x",
                "c++",
                "-",
                "-I",
                str(STUBS),
                "-I",
                str(ROOT / "include"),
                "-I",
                str(ROOT / "p2p"),
                *defines,
                "-o",
                exe,
            ]
            result = subprocess.run(command, input=text, text=True, capture_output=True)
            if run:
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run(
                    [exe], text=True, capture_output=True, env=env, timeout=10
                )
            return result

    def test_runtime_mapping_and_allocation_range(self):
        result = self.compile(
            r"""
#include <cassert>
#include <type_traits>
#include "util/gpu_rt.h"
int main() {
  static_assert(std::is_same<gpuError_t, musaError_t>::value);
  static_assert(std::is_same<gpuStream_t, musaStream_t>::value);
  static_assert(std::is_same<gpuIpcMemHandle_t, musaIpcMemHandle_t>::value);
  void* ptr = nullptr;
  assert(gpuMalloc(&ptr, 4096) == gpuSuccess && ptr == allocated_ptr);
  gpuPointerAttribute_t attr{};
  assert(gpuPointerGetAttributes(&attr, ptr) == gpuSuccess);
  assert(gpuMemTypeOf(attr) == gpuMemoryTypeDevice && attr.device == 3);
  void* base = nullptr;
  size_t size = 0;
  assert(gpuMemGetAddressRange(&base, &size, (void*)0x12340010) == gpuSuccess);
  assert(base == (void*)0x12340000 && size == 4096);
  range_result = MUSA_ERROR_INVALID_VALUE;
  base = (void*)0x42;
  size = 17;
  assert(gpuMemGetAddressRange(&base, &size, ptr) != gpuSuccess);
  assert(base == (void*)0x42 && size == 17);
}
""",
            "-DUCCL_USE_MUSA",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_musa_rejects_conflicting_backends(self):
        for macro in ("__HIP_PLATFORM_AMD__", "__CAMBRICON_PLATFORM_MLU__", "USE_CUDA"):
            with self.subTest(macro=macro):
                result = self.compile(
                    '#include "util/gpu_rt.h"\nint main() {}',
                    "-DUCCL_USE_MUSA",
                    f"-D{macro}",
                    run=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("UCCL_USE_MUSA", result.stderr)
                self.assertIn("conflict", result.stderr.lower())

    def test_ipc_export_uses_real_allocation_base_and_checks_bounds(self):
        result = self.compile(
            r"""
#include <cassert>
#include <cstdint>
#include "util/gpu_rt.h"
int main() {
  GpuIpcExportContext context;
  GpuIpcExport exported{};
  void* ptr = reinterpret_cast<void*>(0x12341234);
  assert(gpuExportIpcRange(&context, ptr, 64, &exported) == gpuSuccess);
  assert(last_ipc_export_ptr == reinterpret_cast<void*>(0x12341000));
  assert(exported.offset == 0x234 && exported.handle.reserved[0] == 42);
  assert(exported.owner_device == 3);
  assert(ipc_export_calls == 1);
  exported.offset = 99;
  exported.handle.reserved[0] = 7;
  assert(gpuExportIpcRange(&context, ptr, 4096, &exported) != gpuSuccess);
  assert(gpuExportIpcRange(&context, ptr, SIZE_MAX, &exported) != gpuSuccess);
  assert(ipc_export_calls == 1 && exported.offset == 99 &&
         exported.handle.reserved[0] == 7);
  range_result = MUSA_ERROR_INVALID_VALUE;
  assert(gpuExportIpcRange(&context, ptr, 64, &exported) != gpuSuccess);
  assert(ipc_export_calls == 1 && exported.offset == 99);
  range_result = MUSA_SUCCESS;
  ipc_export_result = musaErrorUnknown;
  assert(gpuExportIpcRange(&context, ptr, 64, &exported) == musaErrorUnknown);
  assert(exported.offset == 99 && exported.handle.reserved[0] == 7);
  ipc_export_result = musaSuccess;
  pointer_result = musaErrorUnknown;
  assert(gpuExportIpcRange(&context, ptr, 64, &exported) == musaErrorUnknown);
  assert(exported.offset == 99 && exported.handle.reserved[0] == 7);
}
""",
            "-DUCCL_USE_MUSA",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_common_ipc_export_contract(self):
        result = self.compile(
            r"""
#include <cassert>
#include "util/gpu_rt.h"
#include "device/ipc.h"
int main() {
  using namespace uccl::device;
  assert(ipc_open_device(0, 3) == 0);
  assert(ipc_async_device(0, 3, false) == 3);
  assert(ipc_async_device(0, 3, true) == 3);
  assert(ipc_async_device(2, -1, false) == 2);
  assert(ipc_async_device(2, -1, true) == 2);
  GpuIpcExportContext context;
  GpuIpcExport exported{};
  void* ptr = reinterpret_cast<void*>(0x12341234);
  assert(gpuExportIpcRange(&context, ptr, 64, &exported) == gpuSuccess);
  assert(exported.handle.reserved[0] == 42);
  assert(exported.offset == 0x234);
  assert(exported.owner_device == 3);
}
""",
            "-DUCCL_USE_MUSA",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_ipc_registration_paths_use_common_export(self):
        for name in ("engine.cc", "engine_api.cc", "uccl_engine.cc"):
            source = (ROOT / "p2p" / name).read_text()
            self.assertNotIn("gpuIpcGetMemHandle", source)
        self.assertIn("export_ipc_range", (ROOT / "p2p/engine_api.cc").read_text())

    def test_endpoint_wrapper_owns_numa_and_notification_dispatch(self):
        wrapper = (ROOT / "p2p/endpoint_wrapper.h").read_text()
        self.assertIn("uccl_get_numa_node", wrapper)
        self.assertIn("uccl_send_notification", wrapper)

        engine = (ROOT / "p2p/engine.cc").read_text()
        notification = engine[engine.index("int Endpoint::send_notification") :]
        notification = notification[: notification.index("MR* Endpoint::get_mr")]
        self.assertNotIn("is_nccl_transport", notification)
        initialization = engine[engine.index("void Endpoint::initialize_engine") :]
        self.assertNotIn("get_numa_node_from_iface", initialization)

    def test_shared_endpoint_factory_owns_constructor_selection(self):
        factory = ROOT / "p2p/transport_factory.h"
        self.assertTrue(factory.is_file(), "shared endpoint factory is missing")
        source = factory.read_text()
        self.assertIn("uccl_create_endpoint", source)
        self.assertIn("GenericEndpoint", source)

        engine = (ROOT / "p2p/engine.cc").read_text()
        first = engine[engine.index("Endpoint::Endpoint(uint32_t") :]
        first = first[: first.index("Endpoint::Endpoint()")]
        second = engine[engine.index("Endpoint::Endpoint()") :]
        second = second[: second.index("Endpoint::~Endpoint")]
        self.assertNotIn("make_shared<NCCLEndpoint>", first)
        self.assertNotIn("make_shared<CxiEndpoint>", first)
        self.assertNotIn("make_shared<RDMAEndpoint>", first)
        self.assertNotIn("make_shared<NCCLEndpoint>", second)
        self.assertNotIn("make_shared<CxiEndpoint>", second)
        self.assertNotIn("make_shared<RDMAEndpoint>", second)

    def test_transport_selection_is_fail_closed(self):
        source = r"""
#include <iostream>
#include <stdexcept>
#include "util/transport_type.h"
int main() {
  try { std::cout << static_cast<int>(get_transport_type()); }
  catch (std::invalid_argument const& e) { std::cerr << e.what(); return 2; }
}
"""
        for name in (
            None,
            "rdma",
            "ib",
            "nccl",
            "tcp",
            "tcpx",
            "efa",
            "cxi",
            "typo",
            "",
        ):
            with self.subTest(transport=name):
                env = os.environ.copy()
                env.pop("UCCL_P2P_TRANSPORT", None)
                if name is not None:
                    env["UCCL_P2P_TRANSPORT"] = name
                result = self.compile(source, "-DUCCL_USE_MUSA", env=env)
                if name is None or name in ("rdma", "ib"):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, "0")
                else:
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("musa", result.stderr)

    def test_existing_transport_selection_unchanged(self):
        source = (
            '#include <iostream>\n#include "util/transport_type.h"\n'
            "int main() { std::cout << static_cast<int>(get_transport_type()); }"
        )
        for name, value in (("rdma", "0"), ("nccl", "1"), ("efa", "2"), ("cxi", "3")):
            result = self.compile(
                source, env={**os.environ, "UCCL_P2P_TRANSPORT": name}
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, value)

    def test_musa_makefile_isolated_rdma_build(self):
        makefile = ROOT / "p2p/Makefile.musa"
        self.assertTrue(makefile.is_file(), "MUSA native Makefile is missing")
        with tempfile.TemporaryDirectory() as directory:
            build_dir = Path(directory) / "isolated-build"
            result = subprocess.run(
                [
                    "make",
                    "-n",
                    "-f",
                    str(makefile),
                    "all",
                    f"BUILD_DIR={build_dir}",
                    "PYTHON=true",
                    "PYTHON_CONFIG=true",
                    "NB_DIR=/fake/nanobind",
                    "MUSA_HOME=/fake/musa",
                    "NB_OBJECTS=",
                ],
                cwd=ROOT / "p2p",
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-DUCCL_USE_MUSA", result.stdout)
        self.assertIn(f"{build_dir}/", result.stdout)
        self.assertIn("-lmusart", result.stdout)
        self.assertIn("-lmusa", result.stdout)
        self.assertIn("rdma/ibverbs_dl.cc", result.stdout)
        self.assertIn("libuccl_p2p.so", result.stdout)
        for unwanted in (
            "-lcuda",
            "-lcudart",
            "-lamdhip64",
            "nccl_endpoint.cc",
            "cxi_endpoint.cc",
            "mlu_staging.cc",
        ):
            self.assertNotIn(unwanted, result.stdout)

    def test_newer_foreign_backend_artifact_cannot_skip_musa_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "p2p/Makefile.musa", root / "Makefile.musa")
            shutil.copytree(ROOT / "p2p/make", root / "make")
            for name in ("cached.o", ".build/musa/uccl_engine.o", "uccl_engine.cc"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                os.utime(path, (100, 100))
            library = root / "libuccl_p2p.so"
            library.write_text("previous CUDA backend")
            os.utime(library, (200, 200))
            result = subprocess.run(
                [
                    "make",
                    "-n",
                    "-f",
                    "Makefile.musa",
                    "libuccl_p2p.so",
                    "CORE_OBJECTS=cached.o",
                    "BUILD_DIR=.build/musa",
                    "PYTHON=true",
                    "PYTHON_CONFIG=true",
                    "NB_DIR=/fake/nanobind",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("-lmusart", result.stdout)

    def test_install_removes_stale_python_extension_variants(self):
        result = subprocess.run(
            [
                "make",
                "-n",
                "-f",
                "Makefile.musa",
                "install",
                "PYTHON=true",
                "PYTHON_CONFIG=true",
                "NB_DIR=/fake/nanobind",
                "NB_OBJECTS=",
                "INSTALL_DIR=/fake/uccl",
            ],
            cwd=ROOT / "p2p",
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rm -f /fake/uccl/p2p.so /fake/uccl/p2p.abi3.so", result.stdout)
        self.assertIn("/fake/uccl/p2p.cpython-*.so", result.stdout)


if __name__ == "__main__":
    unittest.main()
