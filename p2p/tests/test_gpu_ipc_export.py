"""Compile real runtime adapters against narrow, non-SDK contract doubles."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = r"""
#pragma once
#include <cstddef>
#include <cstdint>
enum Error { success = 0, invalid = 1, unknown = 2 };
struct Handle { char reserved[64]{}; };
struct Attributes {
  int type = 2;
  int device = 3;
  void* deviceBasePointer = reinterpret_cast<void*>(0x12341000);
  size_t size = 4096;
};
inline Attributes attributes;
inline Error pointer_error = success, export_error = success;
inline int export_calls = 0;
inline void* export_base = nullptr;
inline Error get_attributes(Attributes* out, void const*) {
  if (pointer_error == success) *out = attributes;
  return pointer_error;
}
inline Error export_handle(Handle* out, void* base) {
  ++export_calls;
  export_base = base;
  if (export_error == success) out->reserved[0] = export_calls;
  return export_error;
}
inline Error driver_range(uintptr_t* base, size_t* size, uintptr_t) {
  *base = reinterpret_cast<uintptr_t>(attributes.deviceBasePointer);
  *size = attributes.size;
  return success;
}
inline Error runtime_range(void** base, size_t* size, void*) {
  *base = attributes.deviceBasePointer;
  *size = attributes.size;
  return success;
}
inline Error unused_api(...) { return success; }
"""

CHECKS = r"""
#include <cassert>
#include "util/gpu_rt.h"
int main() {
  GpuIpcExportContext context;
  GpuIpcExport out;
  void* ptr = reinterpret_cast<void*>(0x12341234);
  assert(gpuExportIpcRange(&context, ptr, 64, &out) == gpuSuccess);
  assert(out.offset == 0x234);
  assert(export_base == attributes.deviceBasePointer);
#if defined(__CAMBRICON_PLATFORM_MLU__)
  assert(out.owner_device == 3);
  assert(gpuExportIpcRange(&context, ptr, 64, &out) == gpuSuccess);
  assert(export_calls == 1);
  GpuIpcExportContext other;
  assert(gpuExportIpcRange(&other, ptr, 64, &out) == gpuSuccess);
  assert(export_calls == 2);
  attributes.device = 4;
  assert(gpuExportIpcRange(&context, ptr, 64, &out) == gpuSuccess);
  assert(export_calls == 3 && out.owner_device == 4);
#else
  assert(out.owner_device == -1);
  assert(gpuExportIpcRange(&context, ptr, 64, &out) == gpuSuccess);
  assert(export_calls == 2);
#endif
  int calls = export_calls;
  out.offset = 99;
  out.owner_device = 42;
  out.handle.reserved[0] = 7;
  assert(gpuExportIpcRange(&context, ptr, 4096, &out) != gpuSuccess);
  assert(gpuExportIpcRange(&context, ptr, SIZE_MAX, &out) != gpuSuccess);
  assert(gpuExportIpcRange(nullptr, ptr, 64, &out) != gpuSuccess);
  assert(gpuExportIpcRange(&context, nullptr, 64, &out) != gpuSuccess);
  assert(gpuExportIpcRange(&context, ptr, 64, nullptr) != gpuSuccess);
  assert(export_calls == calls);
  assert(out.offset == 99 && out.owner_device == 42 && out.handle.reserved[0] == 7);
  GpuIpcExportContext fresh;
  export_error = unknown;
  assert(gpuExportIpcRange(&fresh, ptr, 64, &out) == unknown);
  assert(out.offset == 99 && out.owner_device == 42 && out.handle.reserved[0] == 7);
}
"""


def runtime_header(backend):
    if backend == "cnrt":
        aliases = {
            "cnrtRet_t": "Error",
            "cnrtSuccess": "success",
            "cnrtErrorArgsInvalid": "invalid",
            "cnrtIpcMemHandle": "Handle",
            "cnrtPointerAttributes_t": "Attributes",
            "cnrtPointerGetAttributes": "get_attributes",
            "cnrtAcquireMemHandle": "export_handle",
            "cnrtMemTypeDevice": "2",
            "cnrtQueue_t": "void*",
            "cnrtNotifier_t": "void*",
            "cnrtMemTransDir_t": "int",
            "CNresult": "Error",
            "CNaddr": "uintptr_t",
        }
        for function in (
            "cnrtQueueCreate",
            "cnrtNotifierCreate",
            "cnrtQueueWaitNotifier",
            "cnrtMemcpyAsync",
            "cnrtGetPeerAccessibility",
            "cnrtGetDeviceCount",
        ):
            aliases[function] = "unused_api"
    else:
        aliases = {
            f"{backend}Error_t": "Error",
            f"{backend}Success": "success",
            f"{backend}ErrorInvalidValue": "invalid",
            f"{backend}IpcMemHandle_t": "Handle",
            f"{backend}PointerAttributes": "Attributes",
            f"{backend}PointerAttribute_t": "Attributes",
            f"{backend}PointerGetAttributes": "get_attributes",
            f"{backend}IpcGetMemHandle": "export_handle",
            f"{backend}MemGetAddressRange": "runtime_range",
            f"{backend}MemoryTypeDevice": "2",
            f"{backend}MemoryTypeManaged": "3",
            f"{backend}Deviceptr_t": "uintptr_t",
            f"{backend}MemRangeHandleType": "int",
            "CUresult": "Error",
            "CUdeviceptr": "uintptr_t",
            "CUmemRangeHandleType": "int",
            "CUDA_SUCCESS": "success",
            "cuMemGetAddressRange": "driver_range",
        }
    return '#include "runtime.h"\n' + "\n".join(
        f"#define {name} {value}" for name, value in aliases.items()
    )


class GpuIpcExportTests(unittest.TestCase):
    def test_cuda_hip_cnrt_export_contract(self):
        compiler = os.environ.get("CXX", "c++")
        self.assertIsNotNone(shutil.which(compiler))
        for backend, macro, headers in (
            ("cuda", "USE_CUDA", ("cuda.h", "cuda_runtime.h")),
            (
                "hip",
                "__HIP_PLATFORM_AMD__",
                ("hip/hip_runtime.h", "hip/hip_runtime_api.h"),
            ),
            ("cnrt", "__CAMBRICON_PLATFORM_MLU__", ("cn_api.h", "cnrt.h")),
        ):
            with self.subTest(
                backend=backend
            ), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                (directory / "runtime.h").write_text(RUNTIME)
                for header in headers:
                    path = directory / header
                    path.parent.mkdir(exist_ok=True)
                    path.write_text(runtime_header(backend))
                exe = str(directory / "test")
                result = subprocess.run(
                    [
                        compiler,
                        "-std=c++17",
                        "-pthread",
                        f"-D{macro}",
                        "-I",
                        str(directory),
                        "-I",
                        str(ROOT / "include"),
                        "-x",
                        "c++",
                        "-",
                        "-o",
                        exe,
                    ],
                    input=CHECKS,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run(
                    [exe], capture_output=True, text=True, timeout=10
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
