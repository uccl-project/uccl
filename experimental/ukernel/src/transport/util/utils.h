#pragma once

#include "jring.h"
#include <cstdarg>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

std::string generate_host_id(bool include_ip = false);

}  // namespace Transport
}  // namespace UKernel

namespace UKernel {
namespace Transport {

namespace detail {
template <typename F>
struct FinalAction {
  explicit FinalAction(F f) : clean_(f) {}
  ~FinalAction() {
    if (enabled_) clean_();
  }
  void disable() { enabled_ = false; }

 private:
  F clean_;
  bool enabled_{true};
};
}  // namespace detail

template <typename F>
inline detail::FinalAction<F> finally(F f) {
  return detail::FinalAction<F>(f);
}

inline std::string FormatVarg(char const* fmt, va_list ap) {
  char* ptr = nullptr;
  int len = vasprintf(&ptr, fmt, ap);
  if (len < 0) return "<FormatVarg() error>";
  std::string out(ptr, static_cast<size_t>(len));
  free(ptr);
  return out;
}

inline std::string Format(char const* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::string s = FormatVarg(fmt, ap);
  va_end(ap);
  return s;
}

// Keep 64-byte alignment to match jring's CACHE_LINE_SIZE assumptions.
constexpr std::size_t kRingAlignment = 64;

inline jring_t* create_ring(size_t element_size, size_t element_count) {
  size_t ring_sz = jring_get_buf_ring_size(
      static_cast<uint32_t>(element_size), static_cast<uint32_t>(element_count));
  if (ring_sz == static_cast<size_t>(-1)) return nullptr;
  jring_t* ring = reinterpret_cast<jring_t*>(
      aligned_alloc(kRingAlignment, ring_sz));
  if (ring == nullptr) return nullptr;
  if (jring_init(ring, static_cast<uint32_t>(element_count),
                 static_cast<uint32_t>(element_size), 1, 1) < 0) {
    free(ring);
    return nullptr;
  }
  return ring;
}

inline jring_t* attach_shared_ring(char const* shm_name, int& shm_fd,
                                   size_t shm_size) {
  shm_fd = shm_open(shm_name, O_RDWR, 0666);
  if (shm_fd < 0) return nullptr;

  void* ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    close(shm_fd);
    shm_fd = -1;
    return nullptr;
  }
  return reinterpret_cast<jring_t*>(ptr);
}

inline jring_t* create_shared_ring(char const* shm_name, size_t element_size,
                                   size_t element_count, int& shm_fd,
                                   size_t& shm_size, bool* is_creator) {
  shm_size = jring_get_buf_ring_size(
      static_cast<uint32_t>(element_size), static_cast<uint32_t>(element_count));
  if (shm_size == static_cast<size_t>(-1)) return nullptr;

  shm_fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
  if (shm_fd >= 0) {
    if (is_creator) *is_creator = true;
  } else {
    if (errno != EEXIST) return nullptr;
    if (is_creator) *is_creator = false;
    return attach_shared_ring(shm_name, shm_fd, shm_size);
  }

  if (ftruncate(shm_fd, static_cast<off_t>(shm_size)) < 0) {
    close(shm_fd);
    shm_unlink(shm_name);
    return nullptr;
  }

  void* ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    close(shm_fd);
    shm_unlink(shm_name);
    return nullptr;
  }

  auto* ring = reinterpret_cast<jring_t*>(ptr);
  if (jring_init(ring, static_cast<uint32_t>(element_count),
                 static_cast<uint32_t>(element_size), 1, 1) < 0) {
    munmap(ptr, shm_size);
    close(shm_fd);
    shm_unlink(shm_name);
    return nullptr;
  }
  return ring;
}

inline void detach_shared_ring(jring_t* ring, int shm_fd, size_t shm_size) {
  if (ring != nullptr) {
    munmap(reinterpret_cast<void*>(ring), shm_size);
  }
  if (shm_fd >= 0) {
    close(shm_fd);
  }
}

inline void destroy_shared_ring(char const* shm_name, jring_t* ring, int shm_fd,
                                size_t shm_size) {
  detach_shared_ring(ring, shm_fd, shm_size);
  shm_unlink(shm_name);
}

}  // namespace Transport
}  // namespace UKernel
