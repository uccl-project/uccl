// Compile against real SDK headers. CPU-only test doubles are not accepted.
#include "util/gpu_rt.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef UCCL_USE_MUSA
#error "Compile this probe with UCCL_USE_MUSA"
#endif

int main(int argc, char** argv) {
  int device = argc > 1 ? std::atoi(argv[1]) : 0;
  int count = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&count));
  if (device < 0 || device >= count) return 2;
  GPU_RT_CHECK(gpuSetDevice(device));
  char bdf[64]{};
  GPU_RT_CHECK(gpuDeviceGetPCIBusId(bdf, sizeof(bdf), device));
  int runtime_version = 0, driver_version = 0;
  GPU_RT_CHECK(musaRuntimeGetVersion(&runtime_version));
  GPU_RT_CHECK(musaDriverGetVersion(&driver_version));
  std::printf("device=%d bdf=%s runtime=%d driver=%d ipc_handle_bytes=%zu\n",
              device, bdf, runtime_version, driver_version,
              sizeof(gpuIpcMemHandle_t));

  constexpr size_t bytes = 4096;
  unsigned char input[bytes], output[bytes]{};
  for (size_t i = 0; i < bytes; ++i) input[i] = i % 251;
  void* ptr = nullptr;
  GPU_RT_CHECK(gpuMalloc(&ptr, bytes));
  gpuPointerAttribute_t attrs{};
  GPU_RT_CHECK(gpuPointerGetAttributes(&attrs, ptr));
  assert(gpuMemTypeOf(attrs) == gpuMemoryTypeDevice && attrs.device == device);
  void* base = nullptr;
  size_t allocation_size = 0;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &allocation_size,
                                     static_cast<unsigned char*>(ptr) + 17));
  assert(base == ptr && allocation_size >= bytes);

  gpuStream_t stream;
  gpuEvent_t event;
  GPU_RT_CHECK(gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking));
  GPU_RT_CHECK(gpuEventCreateWithFlags(&event, gpuEventDisableTiming));
  GPU_RT_CHECK(
      gpuMemcpyAsync(ptr, input, bytes, gpuMemcpyHostToDevice, stream));
  GPU_RT_CHECK(gpuEventRecord(event, stream));
  GPU_RT_CHECK(gpuEventSynchronize(event));
  GPU_RT_CHECK(gpuMemcpy(output, ptr, bytes, gpuMemcpyDeviceToHost));
  assert(std::memcmp(input, output, bytes) == 0);
  GpuIpcExportContext context;
  GpuIpcExport exported;
  GPU_RT_CHECK(gpuExportIpcRange(
      &context, static_cast<unsigned char*>(ptr) + 17, bytes - 17, &exported));
  assert(exported.offset == 17 && exported.owner_device == device);
  GPU_RT_CHECK(gpuEventDestroy(event));
  GPU_RT_CHECK(gpuStreamDestroy(stream));
  GPU_RT_CHECK(gpuFree(ptr));
  std::puts("PASS runtime probe (IPC import and NIC DMA not tested here)");
}
