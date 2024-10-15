#pragma once

#include <arpa/inet.h>
#include <glog/logging.h>
#include <linux/in.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <sys/socket.h>

#include <algorithm>
#include <chrono>
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

/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 1982, 1986, 1990, 1993
 *      The Regents of the University of California.
 * Copyright(c) 2010-2014 Intel Corporation.
 * Copyright(c) 2014 6WIND S.A.
 * All rights reserved.
 *
 * These checksum routines were originally from DPDK.
 */

/**
 * @internal Calculate a sum of all words in the buffer.
 * Helper routine for the rte_raw_cksum().
 *
 * @param buf
 *   Pointer to the buffer.
 * @param len
 *   Length of the buffer.
 * @param sum
 *   Initial value of the sum.
 * @return
 *   sum += Sum of all words in the buffer.
 */
static inline uint32_t __raw_cksum(const void* buf, size_t len, uint32_t sum) {
    /* workaround gcc strict-aliasing warning */
    uintptr_t ptr = (uintptr_t)buf;
    typedef uint16_t __attribute__((__may_alias__)) u16_p;
    const u16_p* u16 = (const u16_p*)ptr;

    while (len >= (sizeof(*u16) * 4)) {
        sum += u16[0];
        sum += u16[1];
        sum += u16[2];
        sum += u16[3];
        len -= sizeof(*u16) * 4;
        u16 += 4;
    }
    while (len >= sizeof(*u16)) {
        sum += *u16;
        len -= sizeof(*u16);
        u16 += 1;
    }

    /* if length is in odd bytes */
    if (len == 1) sum += *((const uint8_t*)u16);

    return sum;
}

/**
 * @internal Reduce a sum to the non-complemented checksum.
 * Helper routine for the rte_raw_cksum().
 *
 * @param sum
 *   Value of the sum.
 * @return
 *   The non-complemented checksum.
 */
static inline uint16_t __raw_cksum_reduce(uint32_t sum) {
    sum = ((sum & 0xffff0000) >> 16) + (sum & 0xffff);
    sum = ((sum & 0xffff0000) >> 16) + (sum & 0xffff);
    return (uint16_t)sum;
}

/**
 * Process the non-complemented checksum of a buffer.
 *
 * @param buf
 *   Pointer to the buffer.
 * @param len
 *   Length of the buffer.
 * @return
 *   The non-complemented checksum.
 */
static inline uint16_t raw_cksum(const void* buf, size_t len) {
    uint32_t sum;

    sum = __raw_cksum(buf, len, 0);
    return __raw_cksum_reduce(sum);
}

/**
 * Process the pseudo-header checksum of an IPv4 header.
 *
 * The checksum field must be set to 0 by the caller.
 *
 * @param ipv4_hdr
 *   The pointer to the contiguous IPv4 header.
 * @return
 *   The non-complemented checksum to set in the L4 header.
 */
static inline uint16_t ipv4_phdr_cksum(uint8_t proto, uint32_t saddr,
                                       uint32_t daddr, uint16_t l4len) {
    struct ipv4_psd_header {
        uint32_t saddr; /* IP address of source host. */
        uint32_t daddr; /* IP address of destination host. */
        uint8_t zero;   /* zero. */
        uint8_t proto;  /* L4 protocol type. */
        uint16_t len;   /* L4 length. */
    } psd_hdr;

    psd_hdr.saddr = htonl(saddr);
    psd_hdr.daddr = htonl(daddr);
    psd_hdr.zero = 0;
    psd_hdr.proto = proto;
    psd_hdr.len = htons(l4len);
    return raw_cksum(&psd_hdr, sizeof(psd_hdr));
}

static inline uint16_t ipv4_udptcp_cksum(uint8_t proto, uint32_t saddr,
                                         uint32_t daddr, uint16_t l4len,
                                         const void* l4hdr) {
    uint32_t cksum;

    cksum = raw_cksum(l4hdr, l4len);
    cksum += ipv4_phdr_cksum(proto, saddr, daddr, l4len);
    cksum = ((cksum & 0xffff0000) >> 16) + (cksum & 0xffff);
    cksum = (~cksum) & 0xffff;
    if (cksum == 0) cksum = 0xffff;

    return (uint16_t)cksum;
}

static inline uint16_t tcp_hdr_chksum(uint32_t local_ip, uint32_t remote_ip,
                                      uint16_t len) {
    return ipv4_phdr_cksum(IPPROTO_TCP, local_ip, remote_ip, len);
}

}  // namespace uccl
