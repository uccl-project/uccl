#include "common.hpp"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

std::once_flag peer_ok_flag[MAX_NUM_GPUS][MAX_NUM_GPUS];

bool pin_thread_to_cpu(int cpu) {
  int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpu < 0 || cpu >= num_cpus) return false;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);

  pthread_t current_thread = pthread_self();
  return !pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
  asm volatile("yield" ::: "memory");
#else
  // Fallback
  asm volatile("" ::: "memory");
#endif
}

int get_num_max_nvl_peers() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    std::abort();
  }
  return deviceCount;
}

void maybe_enable_peer_access(int src_dev, int dst_dev) {
  if (src_dev == dst_dev) return;
  std::call_once(peer_ok_flag[src_dev][dst_dev], [&]() {
    GPU_RT_CHECK(gpuSetDevice(dst_dev));
    gpuError_t err = gpuDeviceEnablePeerAccess(src_dev, 0);
    if (err != gpuSuccess && err != gpuErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "Peer access from dst_dev=%d to src_dev=%d failed: %s\n",
              dst_dev, src_dev, gpuGetErrorString(err));
    }

    GPU_RT_CHECK(gpuSetDevice(src_dev));
    err = gpuDeviceEnablePeerAccess(dst_dev, 0);
    if (err != gpuSuccess && err != gpuErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "Peer access from src_dev=%d to dst_dev=%d failed: %s\n",
              src_dev, dst_dev, gpuGetErrorString(err));
    }
  });
}

uint64_t make_wr_id(uint32_t tag, uint32_t slot) {
  return (uint64_t(tag) << 32) | uint64_t(slot);
}
uint32_t wr_tag(uint64_t wrid) { return uint32_t(wrid >> 32); }
uint32_t wr_slot(uint64_t wrid) { return uint32_t(wrid); }
