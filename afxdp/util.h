#pragma once

#include <glog/logging.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <sys/socket.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "util_jring.h"

namespace uccl {

template <class T>
inline T Percentile(std::vector<T>& vectorIn, double percent) {
    if (vectorIn.size() == 0) return (T)0;
    auto nth = vectorIn.begin() + (percent * vectorIn.size()) / 100;
    std::nth_element(vectorIn.begin(), nth, vectorIn.end());
    return *nth;
}

inline uint16_t ipv4_checksum(const void* data, size_t header_length) {
    unsigned long sum = 0;

    const uint16_t* p = (const uint16_t*)data;

    while (header_length > 1) {
        sum += *p++;
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        header_length -= 2;
    }

    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return ~sum;
}

inline bool pin_thread_to_cpu(int cpu) {
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu < 0 || cpu >= num_cpus) return false;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    pthread_t current_thread = pthread_self();

    return !pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

inline void apply_setsockopt(int xsk_fd) {
    int ret;
    int sock_opt;

    sock_opt = 1;

    ret = setsockopt(xsk_fd, SOL_SOCKET, SO_PREFER_BUSY_POLL, (void*)&sock_opt,
                     sizeof(sock_opt));
    if (ret == -EPERM) {
        fprintf(stderr,
                "Ignore SO_PREFER_BUSY_POLL as it failed: this option needs "
                "privileged mode.\n");
    } else if (ret < 0) {
        fprintf(stderr, "Ignore SO_PREFER_BUSY_POLL as it failed\n");
    }

    sock_opt = 20;
    if (setsockopt(xsk_fd, SOL_SOCKET, SO_BUSY_POLL, (void*)&sock_opt,
                   sizeof(sock_opt)) < 0) {
        fprintf(stderr, "Ignore SO_BUSY_POLL as it failed\n");
    }

    sock_opt = 64;
    ret = setsockopt(xsk_fd, SOL_SOCKET, SO_BUSY_POLL_BUDGET, (void*)&sock_opt,
                     sizeof(sock_opt));
    if (ret == -EPERM) {
        fprintf(stderr,
                "Ignore SO_BUSY_POLL_BUDGET as it failed: this option needs "
                "privileged mode.\n");
    } else if (ret < 0) {
        fprintf(stderr, "Ignore SO_BUSY_POLL_BUDGET as it failed\n");
    }
}

namespace detail {
template <typename F>
struct FinalAction {
    FinalAction(F f) : clean_{f} {}
    ~FinalAction() {
        if (enabled_) clean_();
    }
    void disable() { enabled_ = false; };

   private:
    F clean_;
    bool enabled_{true};
};
}  // namespace detail

template <typename F>
inline detail::FinalAction<F> finally(F f) {
    return detail::FinalAction<F>(f);
}

class Spin {
   private:
    pthread_spinlock_t spin_;

   public:
    Spin() { pthread_spin_init(&spin_, PTHREAD_PROCESS_PRIVATE); }
    ~Spin() { pthread_spin_destroy(&spin_); }
    void Lock() { pthread_spin_lock(&spin_); }
    void Unlock() { pthread_spin_unlock(&spin_); }
    bool TryLock() { return pthread_spin_trylock(&spin_) == 0; }
};

#ifndef likely
#define likely(X) __builtin_expect(!!(X), 1)
#endif

#ifndef unlikely
#define unlikely(X) __builtin_expect(!!(X), 0)
#endif

#define load_acquire(X) __atomic_load_n(X, __ATOMIC_ACQUIRE)
#define store_release(X, Y) __atomic_store_n(X, Y, __ATOMIC_RELEASE)

static inline std::string FormatVarg(const char* fmt, va_list ap) {
    char* ptr = nullptr;
    int len = vasprintf(&ptr, fmt, ap);
    if (len < 0) return "<FormatVarg() error>";

    std::string ret(ptr, len);
    free(ptr);
    return ret;
}

[[maybe_unused]] static inline std::string Format(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const std::string s = FormatVarg(fmt, ap);
    va_end(ap);
    return s;
}

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
// 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │
// ...
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif
// TODO(ilias): Adding an assertion for now, to prevent incompatibilities
// with the C helper library.
static_assert(hardware_constructive_interference_size == 64);
static_assert(hardware_destructive_interference_size == 64);

inline jring_t* create_ring(size_t element_size, size_t element_count) {
    size_t ring_sz = jring_get_buf_ring_size(element_size, element_count);
    LOG(INFO) << "Ring size: " << ring_sz
              << " bytes, msg size: " << element_size
              << " bytes, element count: " << element_count;
    jring_t* ring = CHECK_NOTNULL(reinterpret_cast<jring_t*>(
        aligned_alloc(hardware_constructive_interference_size, ring_sz)));
    if (jring_init(ring, element_count, element_size, 1, 1) < 0) {
        LOG(ERROR) << "Failed to initialize ring buffer";
        free(ring);
        exit(EXIT_FAILURE);
    }
    return ring;
}

}  // namespace uccl
