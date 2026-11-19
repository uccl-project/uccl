#pragma once
#include <linux/mempolicy.h>
#include <mscclpp/core.hpp>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite::multinode {
// Storage for the README host-packet schedule under one GPU per process.
// Local source rows stay in their owner's NUMA region. Remote rows are packed
// in the consuming NUMA region. No worker or CUDA operation is started here.
class SharedHostPackets {
 public:
#ifndef UCCL_LITE_HOST_SLOTS
#define UCCL_LITE_HOST_SLOTS 3
#endif
  static constexpr int slots = UCCL_LITE_HOST_SLOTS;
  static_assert(slots >= 2 && slots <= 8, "bounded host packet pipeline");
  // Optional progress-worker summaries; individual ownership counters remain
  // authoritative and are never replaced when granting reuse or receive credit.
  static constexpr int reuseReady = 14, localReady = 15, remoteReady = 16,
                       allReady = 17;
  static constexpr int flagCount = 18;

 private:
  struct Region {
    char name[96]{};
    int fd = -1, device = -1;
    char *host = nullptr, *mapped = nullptr;
    size_t bytes = 0;
    bool owner = false, registered = false;
    ~Region() {
      int previous = 0;
      if (registered) {
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaHostUnregister(host);
        cudaSetDevice(previous);
      }
      if (host) munmap(host, bytes);
      if (fd >= 0) close(fd);
      if (owner && name[0]) shm_unlink(name);
    }
  };
  std::array<Region, 2> regions_;
  int rank_, group_;
  size_t capacity_, controlBytes_, bytes_;
  static void check(cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }
  static void systemCheck(bool ok, char const* operation) {
    if (!ok)
      throw std::runtime_error(std::string(operation) + ": " +
                               std::strerror(errno));
  }
  void map(Region& r) {
    void* p =
        mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, r.fd, 0);
    systemCheck(p != MAP_FAILED, "mmap host packets");
    r.host = static_cast<char*>(p);
    r.bytes = bytes_;
  }

 public:
  SharedHostPackets(mscclpp::Communicator& comm, int rank, size_t capacity)
      : rank_(rank), group_((rank % 4) / 2), capacity_(capacity) {
    if (rank < 0 || rank >= 8 || !capacity || capacity > (1UL << 20) ||
        capacity % 16)
      throw std::invalid_argument("invalid shared host packet configuration");
    size_t page = sysconf(_SC_PAGESIZE);
    auto align = [page](size_t n) { return (n + page - 1) / page * page; };
    controlBytes_ = align(slots * flagCount * 64);
    bytes_ = align(controlBytes_ + slots * 6 * capacity_);
    int device;
    check(cudaGetDevice(&device));
    if (device != rank % 4)
      throw std::invalid_argument(
          "shared packets require contiguous four-GPU hosts");
    struct Identity {
      char name[96];
      size_t bytes;
    };
    std::array<Identity, 8> ids{};
    if ((rank % 2) == 0) {
      auto& own = regions_[group_];
      static std::atomic<unsigned long long> serial{0};
      do {
        std::snprintf(own.name, sizeof(own.name), "/uccl_lite_packet_%d_%llu",
                      getpid(), ++serial);
        own.fd = shm_open(own.name, O_RDWR | O_CREAT | O_EXCL, 0600);
      } while (own.fd < 0 && errno == EEXIST);
      systemCheck(own.fd >= 0, "create host packets");
      own.owner = true;
      systemCheck(ftruncate(own.fd, bytes_) == 0, "size host packets");
      map(own);
      unsigned long mask = 1UL << group_;
      systemCheck(syscall(SYS_mbind, own.host, bytes_, MPOL_BIND, &mask,
                          8 * sizeof(mask), 0) == 0,
                  "bind host packets");
      std::memset(own.host, 0, bytes_);
      std::memcpy(ids[rank].name, own.name, sizeof(own.name));
      ids[rank].bytes = bytes_;
    }
    comm.bootstrap()->allGather(ids.data(), sizeof(Identity));
    for (int group = 0; group < 2; ++group) {
      auto& r = regions_[group];
      auto const& id = ids[rank / 4 * 4 + group * 2];
      if (id.bytes != bytes_ || !id.name[0] || id.name[95])
        throw std::runtime_error("shared packet identity/capacity mismatch");
      if (!r.owner) {
        r.fd = shm_open(id.name, O_RDWR, 0600);
        systemCheck(r.fd >= 0, "open host packets");
        map(r);
      }
      r.device = device;
      check(cudaHostRegister(
          r.host, bytes_, cudaHostRegisterPortable | cudaHostRegisterMapped));
      r.registered = true;
      check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&r.mapped),
                                     r.host, 0));
    }
    // All processes now hold mappings; names need not survive a later crash.
    comm.bootstrap()->barrier();
    if (regions_[group_].owner) {
      systemCheck(shm_unlink(regions_[group_].name) == 0,
                  "unlink host packets");
      regions_[group_].name[0] = 0;
    }
  }
  SharedHostPackets(SharedHostPackets const&) = delete;
  SharedHostPackets& operator=(SharedHostPackets const&) = delete;
  size_t bytes() const { return bytes_; }
  size_t capacity() const { return capacity_; }
  char* host(int group) const { return regions_.at(group).host; }
  char* mapped(int group) const { return regions_.at(group).mapped; }
  size_t payloadOffset(int slot) const {
    if (slot < 0 || slot >= slots) throw std::out_of_range("packet slot");
    return controlBytes_ + size_t(slot) * 6 * capacity_;
  }
  size_t remoteOffset(int slot, int sourceGroup, size_t length) const {
    if (sourceGroup < 0 || sourceGroup >= 2 || !length || length > capacity_ ||
        length % 16)
      throw std::invalid_argument("remote packet extent");
    return payloadOffset(slot) + 2 * capacity_ +
           size_t(sourceGroup) * 2 * length;
  }
  char* row(int sourceRank, int slot, size_t length,
            bool device = false) const {
    if (sourceRank < 0 || sourceRank >= 8 || !length || length > capacity_ ||
        length % 16)
      throw std::invalid_argument("packet row extent");
    bool local = sourceRank / 4 == rank_ / 4;
    int group = local ? (sourceRank % 4) / 2 : group_;
    size_t offset = payloadOffset(slot) + (local ? 0 : 2 * capacity_) +
                    size_t(local ? sourceRank % 2 : sourceRank % 4) * length;
    return (device ? mapped(group) : host(group)) + offset;
  }
  uint64_t* flag(int group, int slot, int index) const {
    if (slot < 0 || slot >= slots || index < 0 || index >= flagCount)
      throw std::out_of_range("packet flag");
    return reinterpret_cast<uint64_t*>(host(group) +
                                       (slot * flagCount + index) * 64);
  }
};
}  // namespace uccl_lite::multinode
