#pragma once

#include "fifo_util.hpp"
#include "gdrapi.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// TODO: support AMD

namespace mscclpp {
namespace detail {

static inline size_t roundUp(size_t x, size_t a) { return (x + a - 1) / a * a; }
static constexpr size_t kGdrPageBytes = 64 * 1024;

static inline CUdeviceptr alignUpDevicePtr(CUdeviceptr x, size_t a) {
  return (x + (a - 1)) & ~(CUdeviceptr)(a - 1);
}

inline gdr_t globalGdr() {
  static gdr_t g = []() -> gdr_t {
    MSCCLPP_CUTHROW(cuInit(0));
#if defined(gdr_open_safe)
    gdr_t h = gdr_open_safe();
#else
    gdr_t h = gdr_open();
#endif
    if (!h) {
      throw std::runtime_error(
          "gdr_open failed. Is gdrdrv loaded (lsmod | grep gdrdrv) "
          "and permission for /dev/gdrdrv ok?");
    }
    return h;
  }();
  return g;
}

inline void ensureDriverContextForCurrentDevice() {
  int dev = 0;
  MSCCLPP_CUDATHROW(cudaGetDevice(&dev));
  (void)cudaFree(0);

  MSCCLPP_CUTHROW(cuInit(0));
  CUdevice cu_dev;
  MSCCLPP_CUTHROW(cuDeviceGet(&cu_dev, dev));

  CUcontext ctx = nullptr;
  MSCCLPP_CUTHROW(cuDevicePrimaryCtxRetain(&ctx, cu_dev));
  MSCCLPP_CUTHROW(cuCtxSetCurrent(ctx));
}

struct GdrMapping {
  int device = -1;
  CUdeviceptr base_dptr = 0;
  CUdeviceptr dptr = 0;
  size_t bytes = 0;
  size_t alloc_bytes = 0;
  size_t map_bytes = 0;
  gdr_mh_t mh{};
  void* map_base = nullptr;
  void* host_ptr = nullptr;
  bool pinned = false;
  bool mapped = false;
};

inline GdrMapping gdrAllocAndMapBytes(size_t req_bytes) {
  GdrMapping m{};
  MSCCLPP_CUDATHROW(cudaGetDevice(&m.device));
  ensureDriverContextForCurrentDevice();

  m.bytes = roundUp(req_bytes, kGdrPageBytes);
  m.alloc_bytes = m.bytes + kGdrPageBytes;

  MSCCLPP_CUTHROW(cuMemAlloc(&m.base_dptr, m.alloc_bytes));
  m.dptr = alignUpDevicePtr(m.base_dptr, kGdrPageBytes);

  if ((size_t)(m.base_dptr + m.alloc_bytes - m.dptr) < m.bytes) {
    (void)cuMemFree(m.base_dptr);
    throw std::runtime_error("gdrAllocAndMapBytes: aligned window too small");
  }

  MSCCLPP_CUTHROW(cuMemsetD8(m.dptr, 0, m.bytes));

  gdr_t g = globalGdr();

  int rc = gdr_pin_buffer(g, m.dptr, m.bytes, 0, 0, &m.mh);
  if (rc != 0) {
    std::fprintf(stderr,
                 "gdr_pin_buffer failed rc=%d bytes=%zu base_dptr=0x%llx "
                 "aligned_dptr=0x%llx\n",
                 rc, m.bytes, (unsigned long long)m.base_dptr,
                 (unsigned long long)m.dptr);
    (void)cuMemFree(m.base_dptr);
    throw std::runtime_error("gdr_pin_buffer failed.");
  }
  m.pinned = true;

  gdr_info_t info{};
  rc = gdr_get_info(g, m.mh, &info);
  if (rc != 0) {
    std::fprintf(stderr, "gdr_get_info failed rc=%d\n", rc);
    (void)gdr_unpin_buffer(g, m.mh);
    (void)cuMemFree(m.base_dptr);
    throw std::runtime_error("gdr_get_info failed.");
  }

  m.map_bytes = (size_t)info.mapped_size;

  rc = gdr_map(g, m.mh, &m.map_base, m.map_bytes);
  if (rc != 0 || !m.map_base) {
    std::fprintf(
        stderr,
        "gdr_map failed rc=%d map_base=%p map_bytes=%zu pin_bytes=%zu "
        "base_dptr=0x%llx aligned_dptr=0x%llx info.va=0x%llx "
        "info.mapped_size=%zu info.page_size=%zu wc=%d mapping_type=%d\n",
        rc, m.map_base, m.map_bytes, m.bytes, (unsigned long long)m.base_dptr,
        (unsigned long long)m.dptr, (unsigned long long)info.va,
        (size_t)info.mapped_size, (size_t)info.page_size, info.wc_mapping,
        info.mapping_type);

    (void)gdr_unpin_buffer(g, m.mh);
    (void)cuMemFree(m.base_dptr);
    throw std::runtime_error("gdr_map failed.");
  }
  m.mapped = true;

  long long const off =
      (long long)((unsigned long long)info.va - (unsigned long long)m.dptr);
  m.host_ptr = (void*)((char*)m.map_base + off);

  return m;
}

inline void gdrUnmapUnpinFree(GdrMapping& m) noexcept {
  if (m.device >= 0) {
    int cur = -1;
    if (cudaGetDevice(&cur) == cudaSuccess && cur != m.device) {
      (void)cudaSetDevice(m.device);
    }
    (void)cudaFree(0);
    try {
      ensureDriverContextForCurrentDevice();
    } catch (...) {
    }
  }

  gdr_t g = nullptr;
  try {
    g = globalGdr();
  } catch (...) {
    g = nullptr;
  }

  if (g && m.pinned) {
    if (m.mapped && m.map_base)
      (void)gdr_unmap(g, m.mh, m.map_base, m.map_bytes);
    (void)gdr_unpin_buffer(g, m.mh);
  }

  if (m.base_dptr) (void)cuMemFree(m.base_dptr);

  m = {};
}

template <class T = void>
struct GdrDeleter {
  GdrMapping m{};
  void operator()(T*) noexcept { gdrUnmapUnpinFree(m); }
};

template <class T>
using UniqueGdrGpuPtr = std::unique_ptr<T, GdrDeleter<T>>;

template <class T>
inline UniqueGdrGpuPtr<T> gpuCallocGdrUnique(size_t nelems = 1) {
  GdrDeleter<T> del{};
  del.m = gdrAllocAndMapBytes(nelems * sizeof(T));
  T* dev_ptr = reinterpret_cast<T*>(static_cast<uintptr_t>(del.m.dptr));
  return UniqueGdrGpuPtr<T>(dev_ptr, del);
}

template <class T>
inline T* getGdrHostPtr(UniqueGdrGpuPtr<T> const& p) {
  return reinterpret_cast<T*>(p.get_deleter().m.host_ptr);
}

template <class T>
inline CUdeviceptr getGdrDevicePtr(UniqueGdrGpuPtr<T> const& p) {
  return p.get_deleter().m.dptr;
}

static constexpr size_t kU64PerPage = kGdrPageBytes / sizeof(uint64_t);

struct GdrU64Page {
  GdrMapping m{};
};

inline GdrU64Page createU64Page() {
  GdrU64Page p{};
  p.m = gdrAllocAndMapBytes(kGdrPageBytes);
  return p;
}

struct U64Slot {
  uint32_t page;
  uint16_t idx;
};

class GdrU64Pool {
 public:
  static GdrU64Pool& instanceForDevice(int dev) {
    static std::mutex g_mu;
    static std::vector<std::unique_ptr<GdrU64Pool>> pools;

    std::lock_guard<std::mutex> lk(g_mu);
    if ((int)pools.size() <= dev) pools.resize(dev + 1);
    if (!pools[dev]) pools[dev] = std::make_unique<GdrU64Pool>(dev);
    return *pools[dev];
  }

  explicit GdrU64Pool(int dev) : dev_(dev) {}

  void reservePages(size_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    while (pages_.size() < n) addPageLocked();
  }

  void alloc(CUdeviceptr& out_dptr, uint64_t*& out_hptr, U64Slot& out_slot) {
    int cur = -1;
    MSCCLPP_CUDATHROW(cudaGetDevice(&cur));
    if (cur != dev_) MSCCLPP_CUDATHROW(cudaSetDevice(dev_));
    (void)cudaFree(0);
    ensureDriverContextForCurrentDevice();

    std::lock_guard<std::mutex> lk(mu_);
    if (free_.empty()) addPageLocked();

    out_slot = free_.back();
    free_.pop_back();

    auto& pg = pages_[out_slot.page].m;
    const size_t byte_off =
        static_cast<size_t>(out_slot.idx) * sizeof(uint64_t);

    out_dptr = pg.dptr + static_cast<CUdeviceptr>(byte_off);
    out_hptr = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(pg.host_ptr) +
                                           byte_off);
  }

  void free(U64Slot const& s) noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    free_.push_back(s);
  }

 private:
  void addPageLocked() {
    int cur = -1;
    MSCCLPP_CUDATHROW(cudaGetDevice(&cur));
    if (cur != dev_) MSCCLPP_CUDATHROW(cudaSetDevice(dev_));
    (void)cudaFree(0);
    ensureDriverContextForCurrentDevice();

    pages_.push_back(createU64Page());
    const uint32_t p = static_cast<uint32_t>(pages_.size() - 1);

    free_.reserve(free_.size() + kU64PerPage);
    for (uint16_t i = 0; i < static_cast<uint16_t>(kU64PerPage); ++i) {
      free_.push_back(U64Slot{p, i});
    }
  }

  int dev_;
  std::mutex mu_;
  std::vector<GdrU64Page> pages_;
  std::vector<U64Slot> free_;
};

struct GdrU64Deleter {
  int device = 0;
  U64Slot slot{};
  CUdeviceptr dptr = 0;
  uint64_t* host_ptr = nullptr;

  void operator()(uint64_t*) noexcept {
    GdrU64Pool::instanceForDevice(device).free(slot);
    slot = U64Slot{};
    dptr = 0;
    host_ptr = nullptr;
  }
};

using UniqueGdrU64Ptr = std::unique_ptr<uint64_t, GdrU64Deleter>;

inline UniqueGdrU64Ptr gpuCallocGdrU64Unique() {
  int dev = 0;
  MSCCLPP_CUDATHROW(cudaGetDevice(&dev));
  (void)cudaFree(0);
  ensureDriverContextForCurrentDevice();

  GdrU64Deleter del{};
  del.device = dev;

  CUdeviceptr d{};
  uint64_t* h{};
  U64Slot s{};
  GdrU64Pool::instanceForDevice(dev).alloc(d, h, s);

  del.slot = s;
  del.dptr = d;
  del.host_ptr = h;

  uint64_t* dev_ptr = reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(d));
  return UniqueGdrU64Ptr(dev_ptr, del);
}

inline uint64_t* getGdrHostPtr(UniqueGdrU64Ptr const& p) {
  return p.get_deleter().host_ptr;
}

inline CUdeviceptr getGdrDevicePtr(UniqueGdrU64Ptr const& p) {
  return p.get_deleter().dptr;
}

}  // namespace detail
}  // namespace mscclpp
