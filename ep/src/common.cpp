#include "common.hpp"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <fstream>
#include <sstream>
#include <vector>

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

static std::vector<int> cpus_in_numa_node(int numa_node) {
  std::ifstream f("/sys/devices/system/node/node" + std::to_string(numa_node) +
                  "/cpulist");
  std::string line;
  std::getline(f, line);
  std::vector<int> cpus;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    size_t dash = tok.find('-');
    if (dash == std::string::npos) {
      cpus.push_back(std::stoi(tok));
    } else {
      int a = std::stoi(tok.substr(0, dash));
      int b = std::stoi(tok.substr(dash + 1));
      for (int c = a; c <= b; ++c) cpus.push_back(c);
    }
  }
  return cpus;
}

// Pin a single thread to a unique CPU inside its NUMA node.
// threads_per_rank: number of CPU threads you spawn per local rank (e.g., 2 or
// 4)
bool pin_thread_unique(int numa_node, int local_rank, int thread_idx,
                       int threads_per_rank) {
  auto cpus = cpus_in_numa_node(numa_node);
  if (cpus.empty()) {
    fprintf(stderr, "No CPUs for NUMA node %d\n", numa_node);
    return false;
  }

  // Make a unique ordinal per (rank, thread)
  int ordinal = local_rank * threads_per_rank + thread_idx;

  // (Optional) prefer physical cores over SMT by using the first half of the
  // list. If your cpulist is [phys0,ht0,phys1,ht1,...], you can compact to even
  // indices instead.

  int chosen_cpu = cpus[ordinal % cpus.size()];

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(chosen_cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    perror("sched_setaffinity");
    return false;
  }

  // Give scheduler a chance to migrate if needed
  sched_yield();
  int now = sched_getcpu();
  printf(
      "Pinned to NUMA node %d, thread_idx: %d, local_rank: %d, running on CPU "
      "%d.\n",
      numa_node, thread_idx, local_rank, now);
  return true;
}

bool pin_thread_to_numa(int numa_node) {
  char cpumap_path[1024];
  snprintf(cpumap_path, sizeof(cpumap_path),
           "/sys/devices/system/node/node%d/cpulist", numa_node);
  std::string cpumap_path_str(cpumap_path);
  std::ifstream cpumap_file(cpumap_path_str);
  if (!cpumap_file.is_open()) {
    fprintf(stderr, "Failed to open %s\n", cpumap_path);
    return false;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  std::string line;
  std::getline(cpumap_file, line);

  // Parse CPU ranges like "0-3,7-11"
  std::stringstream ss(line);
  std::string range;
  while (std::getline(ss, range, ',')) {
    size_t dash = range.find('-');
    if (dash != std::string::npos) {
      // Handle range like "0-3"
      int start = std::stoi(range.substr(0, dash));
      int end = std::stoi(range.substr(dash + 1));
      for (int cpu = start; cpu <= end; cpu++) {
        CPU_SET(cpu, &cpuset);
      }
    } else {
      // Handle single CPU like "7"
      int cpu = std::stoi(range);
      CPU_SET(cpu, &cpuset);
    }
  }

  return !sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
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
