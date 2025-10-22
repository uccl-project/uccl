#pragma once
#include "common.hpp"
#include "ring_buffer.cuh"
#include <cstdint>
#ifdef USE_MSCCLPP_FIFO_BACKEND
#include "fifo_cmd_queue.hpp"
#endif

#define NOT_IMPLEMENTED()                                                    \
  do {                                                                       \
    std::fprintf(stderr, "ERROR: Not implemented (%s:%d in %s)\n", __FILE__, \
                 __LINE__, __func__);                                        \
    assert(false && "NOT IMPLEMENTED");                                      \
    std::abort();                                                            \
  } while (0)

namespace d2hq {

struct HostD2HHandle {
#ifdef USE_MSCCLPP_FIFO_BACKEND
  d2hq::FifoCmdQueue* fifo = nullptr;  // queue wrapper (pop-based)
#else
  DeviceToHostCmdBuffer* ring = nullptr;  // indexed ring buffer
#endif

  inline uint64_t volatile_head() const noexcept {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    // No random-access head in FIFO; provide a conservative value (no new work)
    // so callers using ring-style scans do nothing. Real consumption should
    // use try_pop() via host_try_pop_next().
    NOT_IMPLEMENTED();
    return 0;
#else
    return ring->volatile_head();
#endif
  }

  inline uint64_t volatile_tail() const noexcept {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    // Same note as above; return 0 to keep scans inert under FIFO.
    NOT_IMPLEMENTED();
    return 0;
#else
    return ring->volatile_tail();
#endif
  }

  inline CmdType volatile_load_cmd_type(size_t idx) const noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->volatile_load_cmd_type(static_cast<int>(idx));
#else
    // FIFO has no indexed view; return EMPTY so scan-based callers will skip.
    (void)idx;
    NOT_IMPLEMENTED();
    return CmdType::EMPTY;
#endif
  }

  inline TransferCmd& load_cmd_entry(size_t idx) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->load_cmd_entry(static_cast<int>(idx));
#else
    // FIFO is pop-driven. Provide a thread-local slot to return by reference.
    static thread_local TransferCmd tmp;
    if (!fifo || !fifo->try_pop(tmp)) {
      tmp.cmd_type = CmdType::EMPTY;
    }
    (void)idx;  // ignored in FIFO
    NOT_IMPLEMENTED();
    return tmp;
#endif
  }

  inline void volatile_clear_cmd_type(size_t idx) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring->volatile_clear_cmd_type(static_cast<int>(idx));
#else
    (void)idx;
    NOT_IMPLEMENTED();
#endif
  }

  inline void cpu_volatile_store_tail(uint64_t new_tail) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring->cpu_volatile_store_tail(new_tail);
#else
    (void)new_tail;
    NOT_IMPLEMENTED();
#endif
  }

  inline void mark_acked(size_t idx) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring->mark_acked(idx);
#else
    (void)idx;
    NOT_IMPLEMENTED();
#endif
  }

  inline void clear_acked(size_t idx) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring->clear_acked(idx);
#else
    (void)idx;
#endif
  }

  inline bool is_acked(size_t idx) const noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->is_acked(idx);
#else
    (void)idx;
    NOT_IMPLEMENTED();
    return false;
#endif
  }

  inline uint64_t advance_tail_from_mask() noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->advance_tail_from_mask();
#else
    NOT_IMPLEMENTED();
    return 0;
#endif
  }

  inline size_t capacity() const noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->capacity;
#else
    NOT_IMPLEMENTED();
    return fifo ? static_cast<size_t>(fifo->size()) : 0;
#endif
  }
};

#ifdef USE_MSCCLPP_FIFO_BACKEND
inline void init_handle(HostD2HHandle& h, d2hq::FifoCmdQueue* q) { h.fifo = q; }

inline HostD2HHandle make_handle(d2hq::FifoCmdQueue* q) {
  HostD2HHandle h{};
  init_handle(h, q);
  return h;
}
#else
inline void init_handle(HostD2HHandle& h, DeviceToHostCmdBuffer* rb) {
  h.ring = rb;
}

inline HostD2HHandle make_handle(DeviceToHostCmdBuffer* rb) {
  HostD2HHandle h{};
  init_handle(h, rb);
  return h;
}
#endif

#ifdef USE_MSCCLPP_FIFO_BACKEND

// Core overload — constructs HostD2HHandle from raw mscclpp::Fifo* array.
inline void init_d2h_from_fifo(
    mscclpp::Fifo* const* fifos, size_t count,
    std::vector<std::unique_ptr<d2hq::FifoCmdQueue>>& fifo_wrappers,
    std::vector<HostD2HHandle>& storage, std::vector<HostD2HHandle*>& out) {
  storage.resize(count);
  fifo_wrappers.clear();
  fifo_wrappers.reserve(count);
  out.clear();
  out.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    // Create wrapper and link it to the existing FIFO
    auto wrapper = std::make_unique<d2hq::FifoCmdQueue>(fifos[i]);
    storage[i].fifo = wrapper.get();
    fifo_wrappers.push_back(std::move(wrapper));
    out.push_back(&storage[i]);
  }
}

inline void init_d2h_from_fifo(
    std::vector<std::unique_ptr<mscclpp::Fifo>> const& fifo_uptrs,
    std::vector<std::unique_ptr<d2hq::FifoCmdQueue>>& fifo_wrappers,
    std::vector<HostD2HHandle>& storage, std::vector<HostD2HHandle*>& out) {
  std::vector<mscclpp::Fifo*> ptrs;
  ptrs.reserve(fifo_uptrs.size());
  for (auto const& u : fifo_uptrs) ptrs.push_back(u.get());
  init_d2h_from_fifo(ptrs.data(), ptrs.size(), fifo_wrappers, storage, out);
}

// Lightweight 3-arg overload — matches your call site in init_env()
// Uses a TLS wrapper shelf so BenchEnv doesn't need to store wrappers.
inline void init_d2h_from_fifo(
    std::vector<std::unique_ptr<mscclpp::Fifo>> const& fifo_uptrs,
    std::vector<HostD2HHandle>& storage) {
  static thread_local std::vector<std::unique_ptr<d2hq::FifoCmdQueue>>
      fifo_wrappers_tls;

  fifo_wrappers_tls.clear();
  fifo_wrappers_tls.reserve(fifo_uptrs.size());

  storage.resize(fifo_uptrs.size());
  for (size_t i = 0; i < fifo_uptrs.size(); ++i) {
    auto wrapper = std::make_unique<d2hq::FifoCmdQueue>(fifo_uptrs[i].get());
    storage[i].fifo = wrapper.get();
    fifo_wrappers_tls.push_back(std::move(wrapper));
  }
}
#else

inline void init_d2h_from_ring(DeviceToHostCmdBuffer* rbs, size_t count,
                               std::vector<HostD2HHandle>& storage) {
  storage.resize(count);
  for (size_t i = 0; i < count; ++i) {
    storage[i] = make_handle(&rbs[i]);
  }
}

#endif  // USE_MSCCLPP_FIFO_BACKEND

inline void init_from_addr(HostD2HHandle& h, uintptr_t addr) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
  h.fifo = reinterpret_cast<d2hq::FifoCmdQueue*>(addr);
#else
  h.ring = reinterpret_cast<DeviceToHostCmdBuffer*>(addr);
#endif
}

}  // namespace d2hq