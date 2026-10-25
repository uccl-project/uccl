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
inline int copy_calls = 0, peer_calls = 0, last_dst_device = -1, last_src_device = -1;
inline void* last_dst = nullptr;
inline void const* last_src = nullptr;
inline size_t last_size = 0;
inline void* last_stream = nullptr;
inline int last_kind = -1;
inline Error copy_error = success;
inline Error copy(void* dst, void const* src, size_t size, int kind, void* stream) {
  ++copy_calls; last_dst = dst; last_src = src; last_size = size;
  last_kind = kind; last_stream = stream; return copy_error;
}
inline Error cnrt_copy(void* dst, void* src, size_t size, void* stream, int kind) {
  return copy(dst, src, size, kind, stream);
}
inline Error peer_copy(void* dst, int dst_device, void const* src,
                       int src_device, size_t size, void* stream) {
  ++peer_calls; last_dst = dst; last_src = src; last_size = size;
  last_dst_device = dst_device; last_src_device = src_device;
  last_stream = stream; return copy_error;
}
"""

CHECKS = r"""
#include <cassert>
#include "util/gpu_rt.h"
#include "device/ipc.h"
#include "util/nccl_types.h"
int main() {
  using NcclSend = ncclResult_t (*)(void const*, size_t, ncclDataType_t, int, ncclComm_t, gpuStream_t);
  static_assert(std::is_same_v<decltype(&ncclSend), NcclSend>);
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
  using namespace uccl::device;
  assert(ipc_open_device(0, -1) == 0);
  assert(ipc_open_device(0, 3) == (Backend::ipc_owner_context ? 3 : 0));
  // Cross-process exporter ordinals are not local CUDA/HIP ordinals.
#if defined(__CAMBRICON_PLATFORM_MLU__)
  assert(ipc_async_device(0, 3, false) == 3);
#else
  assert(ipc_async_device(0, 3, false) == 0);
#endif
  // Same-process direct addresses still use the allocation owner's context.
  assert(ipc_async_device(0, 3, true) == 3);
  assert(ipc_async_device(2, -1, false) == 2);
  assert(ipc_async_device(2, -1, true) == 2);
  auto* dst = reinterpret_cast<void*>(0x5000);
  auto* src = reinterpret_cast<void*>(0x7000);
  auto* stream = reinterpret_cast<void*>(0x9000);
  assert(ipc_copy_async(dst, 3, src, 0, 128, false, 2, stream) == success);
  assert(peer_calls == int(Backend::ipc_owner_context));
  assert(copy_calls == int(!Backend::ipc_owner_context));
  assert(last_dst == dst && last_src == src && last_size == 128 && last_stream == stream);
  if (Backend::ipc_owner_context) assert(last_dst_device == 3 && last_src_device == 0);
  assert(ipc_copy_async(dst, 0, src, 3, 128, false, 2, stream) == success);
  if (Backend::ipc_owner_context) assert(last_dst_device == 0 && last_src_device == 3);
  int peers = peer_calls;
  // Host and same-device transfers must always use the regular copy API.
  assert(ipc_copy_async(dst, 3, src, 0, 128, true, 1, stream) == success);
  assert(last_kind == 1 && peer_calls == peers);
  assert(ipc_copy_async(dst, 0, src, 0, 128, false, 2, stream) == success);
  assert(last_kind == 2 && peer_calls == peers);
  copy_error = unknown;
  assert(ipc_copy_async(dst, 3, src, 0, 128, false, 2, stream) == unknown);

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
    aliases.update(
        {
            f"{backend}Stream_t": "void*",
            f"{backend}MemcpyAsync": "cnrt_copy" if backend == "cnrt" else "copy",
            f"{backend}MemcpyPeerAsync": "peer_copy",
        }
    )
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
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as directory,
            ):
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
                        "-I",
                        str(ROOT / "p2p"),
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
