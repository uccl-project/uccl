#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

// Shared host-pinned buffer for intra-node communication.
//
// All local ranks on the same node share a single POSIX shared-memory region.
// Each rank owns a contiguous slice; any local rank can read/write any slice.
// The buffer is registered with CUDA (cudaHostRegister) so the GPU accesses
// it via PCIe-mapped device pointers.
//
// Usage for intra-node data transfer:
//   Sender proxy: memcpy(dst_rank_slice + rptr, my_slice + lptr, bytes)
//   No RDMA NIC involvement for same-node peers.
struct SharedBuffer {
  void* mmap_ptr = nullptr;        // mmap'd host pointer (full region)
  void* device_ptr = nullptr;      // cudaHostGetDevicePointer result (full)
  size_t total_size = 0;           // per_rank_size * local_world_size
  size_t per_rank_size = 0;
  int local_rank = -1;
  int local_world_size = 0;
  std::string shm_name;
  int fd = -1;
  bool is_creator = false;         // rank 0 creates, others open

  // Host pointer to a specific local rank's slice.
  void* host_slice(int lr) const {
    return static_cast<uint8_t*>(mmap_ptr) + lr * per_rank_size;
  }

  // Device pointer to a specific local rank's slice.
  void* device_slice(int lr) const {
    return static_cast<uint8_t*>(device_ptr) + lr * per_rank_size;
  }

  // My own host slice.
  void* my_host_ptr() const { return host_slice(local_rank); }
  // My own device slice.
  void* my_device_ptr() const { return device_slice(local_rank); }
};

// Allocate (or open) a POSIX shared-memory buffer.
// `tag` distinguishes different buffers (e.g. "rdma", "atomic").
// local_rank 0 creates; others spin-wait until it exists.
// Returns a fully initialized SharedBuffer with CUDA registration.
inline SharedBuffer allocate_shared_buffer(
    const char* tag, size_t per_rank_bytes, int local_rank,
    int local_world_size, int device_index) {
  SharedBuffer sb;
  sb.per_rank_size = per_rank_bytes;
  sb.local_rank = local_rank;
  sb.local_world_size = local_world_size;
  sb.total_size = per_rank_bytes * local_world_size;

  // Unique name per node.
  char hostname[256] = {};
  gethostname(hostname, sizeof(hostname));
  sb.shm_name = std::string("/uccl_") + tag + "_" + hostname;

  if (local_rank == 0) {
    // Unlink stale segment if any.
    shm_unlink(sb.shm_name.c_str());
    sb.fd = shm_open(sb.shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (sb.fd < 0) {
      perror("shm_open(create)");
      std::abort();
    }
    if (ftruncate(sb.fd, sb.total_size) != 0) {
      perror("ftruncate");
      std::abort();
    }
    sb.is_creator = true;
  } else {
    // Spin until rank 0 creates the segment.
    for (int tries = 0;; ++tries) {
      sb.fd = shm_open(sb.shm_name.c_str(), O_RDWR, 0666);
      if (sb.fd >= 0) break;
      if (tries > 5000) {
        fprintf(stderr,
                "[SharedBuffer] rank %d: timed out waiting for shm '%s'\n",
                local_rank, sb.shm_name.c_str());
        std::abort();
      }
      usleep(1000);
    }
    sb.is_creator = false;
  }

  sb.mmap_ptr =
      mmap(nullptr, sb.total_size, PROT_READ | PROT_WRITE, MAP_SHARED, sb.fd, 0);
  if (sb.mmap_ptr == MAP_FAILED) {
    perror("mmap");
    std::abort();
  }

  // Zero my slice.
  std::memset(sb.my_host_ptr(), 0, per_rank_bytes);

  // Register the entire region with CUDA for GPU access.
  cudaSetDevice(device_index);
  auto err = cudaHostRegister(sb.mmap_ptr, sb.total_size,
                               cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaHostRegister failed: %s\n", cudaGetErrorString(err));
    std::abort();
  }
  err = cudaHostGetDevicePointer(&sb.device_ptr, sb.mmap_ptr, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaHostGetDevicePointer failed: %s\n",
            cudaGetErrorString(err));
    std::abort();
  }

  printf("[SharedBuffer] rank %d: tag=%s total=%.1f MiB per_rank=%.1f MiB "
         "host=%p dev=%p\n",
         local_rank, tag, sb.total_size / (1024.0 * 1024.0),
         per_rank_bytes / (1024.0 * 1024.0), sb.mmap_ptr, sb.device_ptr);

  return sb;
}

inline void free_shared_buffer(SharedBuffer& sb) {
  if (sb.mmap_ptr && sb.mmap_ptr != MAP_FAILED) {
    cudaHostUnregister(sb.mmap_ptr);
    munmap(sb.mmap_ptr, sb.total_size);
    sb.mmap_ptr = nullptr;
  }
  if (sb.fd >= 0) {
    close(sb.fd);
    sb.fd = -1;
  }
  if (sb.is_creator) {
    shm_unlink(sb.shm_name.c_str());
  }
}
