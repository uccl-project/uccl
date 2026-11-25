"""Verify allocation export offsets and fd lifetime without a GPU or NIC."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_gpu_ipc_export import RUNTIME, runtime_header

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r"""
#include <cassert>
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include "util/gpu_rt.h"
inline bool missing = false, fail = false;
inline int exported_fd = -1;
inline Error export_fd(void* out, uintptr_t base, size_t size, int kind,
                       unsigned long long flags) {
  assert(base == reinterpret_cast<uintptr_t>(attributes.deviceBasePointer));
  assert(size == 2 * 1024 * 1024 && flags == 0);
  if (fail) return unknown;
  exported_fd = open("/dev/null", O_RDONLY);
  assert(exported_fd >= 0);
  *static_cast<int*>(out) = exported_fd;
  return success;
}
inline void* driver_open(char const*, int) { return reinterpret_cast<void*>(1); }
inline void* driver_symbol(void*, char const*) {
  return missing ? nullptr : reinterpret_cast<void*>(&export_fd);
}
#define dlopen driver_open
#define dlsym driver_symbol
#include "device/dmabuf.h"
int main(int argc, char** argv) {
  missing = argc > 1 && argv[1][0] == 'm';
  fail = argc > 1 && argv[1][0] == 'f';
  {
    uccl::device::DmaBuf exported;
    bool ok = exported.open(reinterpret_cast<void*>(0x12341234), 64);
    assert(ok == (!missing && !fail));
    if (ok) {
      assert(exported.offset == 0x234 && exported.fd == exported_fd);
      assert(fcntl(exported_fd, F_GETFD) >= 0); // consumer can register this fd
    } else {
      assert(exported.fd == -1);
    }
  }
  if (exported_fd >= 0) assert(fcntl(exported_fd, F_GETFD) == -1 && errno == EBADF);
}
"""


class DmaBufExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        (self.directory / "runtime.h").write_text(RUNTIME)
        (self.directory / "util").mkdir()
        (self.directory / "util/debug.h").write_text(
            "#pragma once\n#include <iostream>\n#define UCCL_LOG(...) std::cerr\n"
        )

    def compile_and_run(self, source, *defines, modes=((),)):
        exe = str(self.directory / "test")
        result = subprocess.run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++17",
                "-pthread",
                *defines,
                "-I",
                str(self.directory),
                "-I",
                str(ROOT / "include"),
                "-I",
                str(ROOT / "p2p"),
                "-x",
                "c++",
                "-",
                "-o",
                exe,
            ],
            input=source,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for mode in modes:
            with self.subTest(args=mode):
                result = subprocess.run(
                    [exe, *mode], text=True, capture_output=True, timeout=10
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_dtk_keeps_shared_driver_abi_without_enabling_p2p_export(self):
        source = r"""
#include <cassert>
#include <type_traits>
#include "device/dmabuf.h"
using DriverExport = gpuDriverResult_t (*)(
    void*, gpuDevicePtr_t, size_t, gpuMemRangeHandleType, unsigned long long);
static_assert(std::is_same_v<gpuMemGetHandleForAddressRange_fn, DriverExport>);
static_assert(UCCL_DEVICE_HAS_DMABUF == 0);
int main() {
  // EP/GIN can still declare the function pointer and fall back when the
  // runtime symbol is unavailable. P2P must not try to call an export API.
  gpuMemGetHandleForAddressRange_fn export_fn = nullptr;
  assert(export_fn == nullptr);
  assert(GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME[0] == '\0');
  uccl::device::DmaBuf exported;
  assert(!exported.open(reinterpret_cast<void*>(0x12341234), 64));
  assert(exported.fd == -1);
}
"""
        (self.directory / "hip").mkdir()
        for name in ("hip_runtime.h", "hip_runtime_api.h"):
            (self.directory / "hip" / name).write_text(runtime_header("hip"))
        self.compile_and_run(source, "-D__HIP_PLATFORM_AMD__", "-D__UCCL_DTK__")

    def test_export_subrange_errors_and_fd_ownership(self):
        header = runtime_header("cuda") + """
#define CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD 1
inline char const* cudaGetErrorString(Error) { return "runtime error"; }
"""
        (self.directory / "cuda.h").write_text("#pragma once\n" + header)
        (self.directory / "cuda_runtime.h").write_text('#include "cuda.h"\n')
        self.compile_and_run(SOURCE, modes=(("success",), ("missing",), ("failure",)))


if __name__ == "__main__":
    unittest.main()
