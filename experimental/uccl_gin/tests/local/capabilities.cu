#include "queue_fixture.hpp"

using namespace local_test;

__global__ void mapped_write(int* status) {
#if defined(__CUDA_ARCH__)
  *status = __CUDA_ARCH__;
#endif
}

int main(int argc, char** argv) {
  int const device = integer(argc, argv, "--device", 0);
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  require(props.canMapHostMemory, "device cannot map host memory");
  int runtime, driver, devices;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  auto status =
      mscclpp::detail::gpuCallocHostUnique<int>(1, cudaHostAllocMapped);
  int* mapped;
  CUDA_CHECK(cudaHostGetDevicePointer(&mapped, status.get(), 0));
  QueueFixture fixture(device, 1, 32);
  mapped_write<<<1, 1, 0, fixture.stream>>>(mapped);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(fixture.stream));
  require(*status != 0, "mapped-host device write did not reach host");
  std::printf(
      "{\"test\":\"capabilities\",\"device\":%d,\"name\":\"%s\","
      "\"cc\":\"%d.%d\",\"compiled_arch\":%d,\"runtime\":%d,"
      "\"driver\":%d,\"async_engines\":%d,\"mapped_host\":true,"
      "\"production_fifo_alloc\":true,\"pass\":true}\n",
      device, props.name, props.major, props.minor, *status, runtime, driver,
      props.asyncEngineCount);
  for (int peer = 0; peer < devices; ++peer) {
    if (peer == device) continue;
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, device, peer));
    std::printf(
        "{\"test\":\"peer_access\",\"src\":%d,\"dst\":%d,"
        "\"access\":%d}\n",
        device, peer, access);
  }
}
