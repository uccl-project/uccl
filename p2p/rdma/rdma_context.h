// Simple RDMA context wrapper - no memory management
#pragma once
#include "common.h"
#include "rdma_device.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <dlfcn.h>
#include <endian.h>

class RdmaContext;

struct MrCacheKey {
  uintptr_t addr;
  size_t size;
  bool is_gpu;
  bool use_dmabuf;

  bool operator==(MrCacheKey const& other) const {
    return addr == other.addr && size == other.size && is_gpu == other.is_gpu &&
           use_dmabuf == other.use_dmabuf;
  }
};

struct MrCacheKeyHash {
  size_t operator()(MrCacheKey const& key) const {
    size_t hash = std::hash<uintptr_t>{}(key.addr);
    hash ^=
        std::hash<size_t>{}(key.size) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^=
        std::hash<bool>{}(key.is_gpu) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<bool>{}(key.use_dmabuf) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
    return hash;
  }
};

struct MrCacheEntry {
  MrCacheKey key;
  struct ibv_mr* mr = nullptr;
  uint64_t refs = 0;
};

struct MrCacheHandleRef {
  std::shared_ptr<RdmaContext> context;
  MrCacheEntry* entry = nullptr;
};

class RdmaContext {
 public:
  explicit RdmaContext(std::shared_ptr<RdmaDevice> dev,
                       uint64_t context_id = 0);

  // Getters
  struct ibv_context* getCtx() const;
  struct ibv_context* ctx() const;
  struct ibv_pd* getPD() const;
  struct ibv_pd* pd() const;

  uint32_t getVendorID() const;
  uint8_t getMaxQpRdAtom() const;
  uint8_t getMaxQpInitRdAtom() const;

  // Query GID by index
  void getGID(int gid_index, union ibv_gid* gid, int port = 1) const;

  union ibv_gid queryGid(int gid_index, int port = 1) const;

  union ibv_gid detectGid(int gid_index, int port = 1) const;

  int getGidIndex(int gid_index, int port = 1) const;

  uint16_t queryLid(int port = 1) const;

  // Create address handle from remote GID
  struct ibv_ah* createAH(union ibv_gid remote_gid, int port = 1) const;

  // Check if a pointer refers to GPU device memory.
  static bool isGpuPointer(void* ptr);

  // Register GPU memory via DMA-BUF for GPUDirect RDMA.
  // Uses kernel DMA-BUF subsystem instead of nvidia_peermem.
  // Returns nullptr on failure so the caller can report the error.
  struct ibv_mr* regMemGpuDmabuf(void* addr, size_t size) const;

  struct ibv_mr* regMem(void* addr, size_t size) const;

  MrCacheEntry* acquireCachedMr(void* addr, size_t size);

  void releaseCachedMr(MrCacheEntry* entry);

  static void deregMem(struct ibv_mr* mr);
  const uint64_t getContextID() const;

 private:
  struct RegistrationMode {
    bool is_gpu;
    bool use_dmabuf;
  };

  static bool containsRange(uintptr_t outer_addr, size_t outer_size,
                            uintptr_t inner_addr, size_t inner_size);

  RegistrationMode getRegistrationMode(void* addr) const;

  struct ibv_mr* regMemImpl(void* addr, size_t size, bool use_dmabuf) const;

  std::shared_ptr<struct ibv_context> ctx_;
  std::shared_ptr<struct ibv_pd> pd_;
  uint8_t max_qp_rd_atom_ = 1;
  uint8_t max_qp_init_rd_atom_ = 1;
  std::mutex mr_cache_mu_;
  std::unordered_map<MrCacheKey, std::unique_ptr<MrCacheEntry>, MrCacheKeyHash>
      mr_cache_;
  uint64_t context_id_;
  uint32_t vendor_id_;
  mutable int gid_index_;
};
