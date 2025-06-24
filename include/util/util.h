#pragma once
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif
#include "util/jring.h"
#include <arpa/inet.h>
#include <glog/logging.h>
#include <linux/in.h>
#include <net/if.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <vector>
#include <fcntl.h>
#include <ifaddrs.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace uccl {

#define UCCL_LOG_PLUGIN VLOG(1) << "[Plugin] "
#define UCCL_LOG_IO VLOG(2) << "[IO] "
#define UCCL_LOG_EP VLOG(3) << "[Endpoint] "
#define UCCL_LOG_RE VLOG(3) << "[Resource] "
#define UCCL_LOG_ENGINE VLOG(4) << "[Engine] "

#define POISON_64 UINT64_MAX
#define POISON_32 UINT32_MAX

#define UCCL_INIT_CHECK(x, msg)      \
  do {                               \
    if (!(x)) {                      \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)

/// Convert a default bytes/second rate to Gbit/s
inline double rate_to_gbps(double r) { return (r / (1000 * 1000 * 1000)) * 8; }

/// Convert a Gbit/s rate to the default bytes/second
inline double gbps_to_rate(double r) { return (r / 8) * (1000 * 1000 * 1000); }

inline int receive_message(int sockfd, void* buffer, size_t n_bytes) {
  int bytes_read = 0;
  int r;
  while (bytes_read < n_bytes) {
    // Make sure we read exactly n_bytes
    r = read(sockfd, buffer + bytes_read, n_bytes - bytes_read);
    if (r < 0 && !(errno == EAGAIN || errno == EWOULDBLOCK)) {
      CHECK(false) << "ERROR reading from socket";
    }
    if (r > 0) {
      bytes_read += r;
    }
  }
  return bytes_read;
}

inline int send_message(int sockfd, void const* buffer, size_t n_bytes) {
  int bytes_sent = 0;
  int r;
  while (bytes_sent < n_bytes) {
    // Make sure we write exactly n_bytes
    r = write(sockfd, buffer + bytes_sent, n_bytes - bytes_sent);
    if (r < 0 && !(errno == EAGAIN || errno == EWOULDBLOCK)) {
      CHECK(false) << "ERROR writing to socket";
    }
    if (r > 0) {
      bytes_sent += r;
    }
  }
  return bytes_sent;
}

inline void send_ready(int bootstrap_fd) {
  bool ready = true;
  int ret = send_message(bootstrap_fd, &ready, sizeof(bool));
  DCHECK(ret == sizeof(bool)) << ret;
}

inline void send_abort(int bootstrap_fd) {
  bool ready = false;
  int ret = send_message(bootstrap_fd, &ready, sizeof(bool));
  DCHECK(ret == sizeof(bool)) << ret;
}

inline void wait_ready(int bootstrap_fd) {
  bool ready;
  int ret = receive_message(bootstrap_fd, &ready, sizeof(bool));
  DCHECK(ret == sizeof(bool) && ready == true) << ret << ", " << ready;
}

inline bool wait_sync(int bootstrap_fd) {
  bool ready;
  int ret = receive_message(bootstrap_fd, &ready, sizeof(bool));
  DCHECK(ret == sizeof(bool)) << ret;
  return ready;
}

inline void net_barrier(int bootstrap_fd) {
  bool sync = true;
  int ret = send_message(bootstrap_fd, &sync, sizeof(bool));
  ret = receive_message(bootstrap_fd, &sync, sizeof(bool));
  DCHECK(ret == sizeof(bool) && sync) << ret << ", " << sync;
}

inline void create_listen_socket(int* listen_fd, uint16_t listen_port) {
  *listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  DCHECK(*listen_fd >= 0) << "ERROR: opening socket";
  int flag = 1;
  DCHECK(setsockopt(*listen_fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int)) >=
         0)
      << "ERROR: setsockopt SO_REUSEADDR fails";
  struct sockaddr_in serv_addr;
  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(listen_port);
  DCHECK(bind(*listen_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) >= 0)
      << "ERROR: binding";

  DCHECK(!listen(*listen_fd, 128)) << "ERROR: listen";
  VLOG(5) << "[Endpoint] server ready, listening on port " << listen_port;
}

#define UINT_CSN_BIT 8
#define UINT_CSN_MASK ((1 << UINT_CSN_BIT) - 1)

constexpr bool seqno_lt(uint8_t a, uint8_t b) {
  return static_cast<int8_t>(a - b) < 0;
}
constexpr bool seqno_le(uint8_t a, uint8_t b) {
  return static_cast<int8_t>(a - b) <= 0;
}
constexpr bool seqno_eq(uint8_t a, uint8_t b) {
  return static_cast<int8_t>(a - b) == 0;
}
constexpr bool seqno_ge(uint8_t a, uint8_t b) {
  return static_cast<int8_t>(a - b) >= 0;
}
constexpr bool seqno_gt(uint8_t a, uint8_t b) {
  return static_cast<int8_t>(a - b) > 0;
}

/**
 * @brief An X-bit (8/16) unsigned integer used for Chunk Sequence Number (CSN).
 */
class UINT_CSN {
 public:
  UINT_CSN() : value_(0) {}
  UINT_CSN(uint32_t value) : value_(value & UINT_CSN_MASK) {}
  UINT_CSN(const UINT_CSN& other) : value_(other.value_) {}

  static inline bool uintcsn_seqno_le(UINT_CSN a, UINT_CSN b) {
    return seqno_le(a.value_, b.value_);
  }

  static inline bool uintcsn_seqno_lt(UINT_CSN a, UINT_CSN b) {
    return seqno_lt(a.value_, b.value_);
  }

  static inline bool uintcsn_seqno_eq(UINT_CSN a, UINT_CSN b) {
    return seqno_eq(a.value_, b.value_);
  }

  static inline bool uintcsn_seqno_ge(UINT_CSN a, UINT_CSN b) {
    return seqno_ge(a.value_, b.value_);
  }

  static inline bool uintcsn_seqno_gt(UINT_CSN a, UINT_CSN b) {
    return seqno_gt(a.value_, b.value_);
  }

  UINT_CSN& operator=(const UINT_CSN& other) {
    value_ = other.value_;
    return *this;
  }
  bool operator==(const UINT_CSN& other) const {
    return value_ == other.value_;
  }
  UINT_CSN operator+(const UINT_CSN& other) const {
    return UINT_CSN(value_ + other.value_);
  }
  UINT_CSN operator-(const UINT_CSN& other) const {
    return UINT_CSN(value_ - other.value_);
  }
  UINT_CSN& operator+=(const UINT_CSN& other) {
    value_ += other.value_;
    value_ &= UINT_CSN_MASK;
    return *this;
  }
  UINT_CSN& operator-=(const UINT_CSN& other) {
    value_ -= other.value_;
    value_ &= UINT_CSN_MASK;
    return *this;
  }
  bool operator<(const UINT_CSN& other) const {
    return seqno_lt(value_, other.value_);
  }
  bool operator<=(const UINT_CSN& other) const {
    return seqno_le(value_, other.value_);
  }
  bool operator>(const UINT_CSN& other) const {
    return seqno_gt(value_, other.value_);
  }
  bool operator>=(const UINT_CSN& other) const {
    return seqno_ge(value_, other.value_);
  }

  inline uint32_t to_uint32() const { return value_; }

 private:
  uint8_t value_;
};

struct alignas(64) PollCtx {
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> fence;               // Sync rx/tx memcpy visibility.
  std::atomic<bool> done;                // Sync cv wake-up.
  std::atomic<uint16_t> num_unfinished;  // Number of unfinished requests.
  uint64_t timestamp;                    // Timestamp for request issuing.
  uint32_t engine_idx;                   // Engine index for request issuing.
  PollCtx() : fence(false), done(false), num_unfinished(0), timestamp(0){};
  ~PollCtx() { clear(); }

  inline void clear() {
    mu.~mutex();
    cv.~condition_variable();
    fence = false;
    done = false;
    num_unfinished = 0;
    timestamp = 0;
  }

  inline void write_barrier() {
    std::atomic_store_explicit(&fence, true, std::memory_order_release);
  }

  inline void read_barrier() {
    std::ignore = std::atomic_load_explicit(&fence, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
  }
};

inline void uccl_wakeup(PollCtx* ctx) {
  std::lock_guard<std::mutex> lock(ctx->mu);
  ctx->done = true;
  ctx->cv.notify_one();
}

inline bool uccl_try_wakeup(PollCtx* ctx) {
  if (--(ctx->num_unfinished) == 0) {
    ctx->write_barrier();
    uccl_wakeup(ctx);
    return true;
  }
  return false;
}

template <class T>
static inline T Percentile(std::vector<T>& vectorIn, double percent) {
  if (vectorIn.size() == 0) return (T)0;
  auto nth = vectorIn.begin() + (percent * vectorIn.size()) / 100;
  std::nth_element(vectorIn.begin(), nth, vectorIn.end());
  return *nth;
}

#define DIVUP(x, y) (((x) + (y)-1) / (y))

template <class T>
static inline T Percentile(std::vector<T> const& vectorIn, double percent) {
  if (vectorIn.size() == 0) return (T)0;
  std::vector<T> vectorCopy = vectorIn;
  auto nth = vectorCopy.begin() + (percent * vectorCopy.size()) / 100;
  std::nth_element(vectorCopy.begin(), nth, vectorCopy.end());
  return *nth;
}

static inline bool pin_thread_to_cpu(int cpu) {
  int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpu < 0 || cpu >= num_cpus) return false;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);

  pthread_t current_thread = pthread_self();

  return !pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

static inline void apply_setsockopt(int xsk_fd) {
  int ret;
  int sock_opt;

#ifdef SO_PREFER_BUSY_POLL
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
#endif
  sock_opt = 20;
  if (setsockopt(xsk_fd, SOL_SOCKET, SO_BUSY_POLL, (void*)&sock_opt,
                 sizeof(sock_opt)) < 0) {
    fprintf(stderr, "Ignore SO_BUSY_POLL as it failed\n");
  }
#ifdef SO_BUSY_POLL_BUDGET
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
#endif
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
static inline detail::FinalAction<F> finally(F f) {
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

// #define barrier() asm volatile("" ::: "memory")
// #define load_acquire(p)                   \
//     ({                                    \
//         typeof(*p) __p = ACCESS_ONCE(*p); \
//         barrier();                        \
//         __p;                              \
//     })
// #define store_release(p, v)  \
//     do {                     \
//         barrier();           \
//         ACCESS_ONCE(*p) = v; \
//     } while (0)

#define load_acquire(X) __atomic_load_n(X, __ATOMIC_ACQUIRE)
#define store_release(X, Y) __atomic_store_n(X, Y, __ATOMIC_RELEASE)
// clang-format off
#define ACCESS_ONCE(x) (*(volatile decltype(x)*)&(x))
// clang-format on
#define is_power_of_two(x) ((x) != 0 && !((x) & ((x)-1)))

#define KB(x) (static_cast<size_t>(x) << 10)
#define MB(x) (static_cast<size_t>(x) << 20)
#define GB(x) (static_cast<size_t>(x) << 30)

static inline std::string FormatVarg(char const* fmt, va_list ap) {
  char* ptr = nullptr;
  int len = vasprintf(&ptr, fmt, ap);
  if (len < 0) return "<FormatVarg() error>";

  std::string ret(ptr, len);
  free(ptr);
  return ret;
}

[[maybe_unused]] static inline std::string Format(char const* fmt, ...) {
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

static inline jring_t* create_ring(size_t element_size, size_t element_count) {
  size_t ring_sz = jring_get_buf_ring_size(element_size, element_count);
  VLOG(5) << "Ring size: " << ring_sz << " bytes, msg size: " << element_size
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

static inline uint16_t ipv4_checksum(void const* data, size_t header_length) {
  unsigned long sum = 0;

  uint16_t const* p = (uint16_t const*)data;

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
static inline uint32_t __raw_cksum(void const* buf, size_t len, uint32_t sum) {
  /* workaround gcc strict-aliasing warning */
  uintptr_t ptr = (uintptr_t)buf;
  typedef uint16_t __attribute__((__may_alias__)) u16_p;
  u16_p const* u16 = (u16_p const*)ptr;

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
  if (len == 1) sum += *((uint8_t const*)u16);

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
static inline uint16_t raw_cksum(void const* buf, size_t len) {
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
                                         void const* l4hdr) {
  uint32_t cksum;

  cksum = raw_cksum(l4hdr, l4len);
  cksum += ipv4_phdr_cksum(proto, saddr, daddr, l4len);
  cksum = ((cksum & 0xffff0000) >> 16) + (cksum & 0xffff);
  cksum = (~cksum) & 0xffff;
  if (cksum == 0) cksum = 0xffff;

  return (uint16_t)cksum;
}

// 0x04030201 (network order) -> 1.2.3.4
static inline std::string ip_to_str(uint32_t ip) {
  struct sockaddr_in sa;
  char str[INET_ADDRSTRLEN];
  sa.sin_addr.s_addr = ip;
  inet_ntop(AF_INET, &(sa.sin_addr), str, INET_ADDRSTRLEN);
  return std::string(str);
}

// 1.2.3.4 -> 0x04030201 (network order)
static inline uint32_t str_to_ip(std::string const& ip) {
  struct sockaddr_in sa;
  DCHECK(inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr)) != 0);
  return sa.sin_addr.s_addr;
}

// Return -1 if not found
static inline int get_dev_index(char const* dev_name) {
  int ret = -1;
  struct ifaddrs* addrs;
  CHECK(getifaddrs(&addrs) == 0) << "error: getifaddrs failed";

  for (struct ifaddrs* iap = addrs; iap != NULL; iap = iap->ifa_next) {
    if (iap->ifa_addr && (iap->ifa_flags & IFF_UP) &&
        iap->ifa_addr->sa_family == AF_INET) {
      struct sockaddr_in* sa = (struct sockaddr_in*)iap->ifa_addr;
      if (strcmp(dev_name, iap->ifa_name) == 0) {
        VLOG(5) << "found network interface: " << iap->ifa_name;
        ret = if_nametoindex(iap->ifa_name);
        CHECK(ret) << "error: if_nametoindex failed";
        break;
      }
    }
  }

  freeifaddrs(addrs);
  return ret;
}

static inline std::string get_dev_ip(char const* dev_name) {
  struct ifaddrs* ifAddrStruct = NULL;
  struct ifaddrs* ifa = NULL;
  void* tmpAddrPtr = NULL;

  getifaddrs(&ifAddrStruct);

  for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr) {
      continue;
    }
    if (strncmp(ifa->ifa_name, dev_name, strlen(dev_name)) != 0) {
      continue;
    }
    if (ifa->ifa_addr->sa_family == AF_INET) {  // check it is IP4
      // is a valid IP4 Address
      tmpAddrPtr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
      char addressBuffer[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
      VLOG(5) << Format("%s IP Address %s\n", ifa->ifa_name, addressBuffer);
      return std::string(addressBuffer);
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {  // check it is IP6
      // is a valid IP6 Address
      tmpAddrPtr = &((struct sockaddr_in6*)ifa->ifa_addr)->sin6_addr;
      char addressBuffer[INET6_ADDRSTRLEN];
      inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
      VLOG(5) << Format("%s IP Address %s\n", ifa->ifa_name, addressBuffer);
      return std::string(addressBuffer);
    }
  }
  if (ifAddrStruct != NULL) freeifaddrs(ifAddrStruct);
  return std::string();
}

// Function to convert MAC string to hex char array
static inline bool str_to_mac(std::string const& macStr, char mac[6]) {
  if (macStr.length() != 17) {
    LOG(ERROR) << "Invalid MAC address format.";
    return false;
  }

  int values[6];  // Temp array to hold integer values
  if (sscanf(macStr.c_str(), "%x:%x:%x:%x:%x:%x", &values[0], &values[1],
             &values[2], &values[3], &values[4], &values[5]) == 6) {
    // Convert to char array
    for (int i = 0; i < 6; i++) {
      mac[i] = static_cast<char>(values[i]);
    }
    return true;
  } else {
    LOG(ERROR) << "Invalid MAC address format.";
    return false;
  }
}

// Function to convert hex char array back to MAC string
static inline std::string mac_to_str(char const mac[6]) {
  std::stringstream ss;
  for (int i = 0; i < 6; i++) {
    ss << std::setfill('0') << std::setw(2) << std::hex
       << static_cast<int>(0xFF & mac[i]);
    if (i != 5) {
      ss << ":";
    }
  }
  return ss.str();
}

static inline std::string get_dev_mac(char const* dev_name) {
  std::string mac;
  std::string cmd = Format("cat /sys/class/net/%s/address", dev_name);
  FILE* fp = popen(cmd.c_str(), "r");
  if (fp == nullptr) {
    LOG(ERROR) << "Failed to get MAC address.";
    return mac;
  }
  char buffer[18];
  if (fgets(buffer, sizeof(buffer), fp) != nullptr) {
    mac = std::string(buffer);
    mac.erase(std::remove(mac.begin(), mac.end(), '\n'), mac.end());
  }
  pclose(fp);
  return mac;
}

static inline int send_fd(int sockfd, int fd) {
  assert(sockfd >= 0);
  assert(fd >= 0);
  struct msghdr msg;
  struct cmsghdr* cmsg;
  struct iovec iov;
  char buf[CMSG_SPACE(sizeof(fd))];
  memset(&msg, 0, sizeof(msg));
  memset(buf, 0, sizeof(buf));
  char const* name = "fd";
  iov.iov_base = (void*)name;
  iov.iov_len = 4;
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  cmsg = CMSG_FIRSTHDR(&msg);

  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(fd));

  *((int*)CMSG_DATA(cmsg)) = fd;

  msg.msg_controllen = CMSG_SPACE(sizeof(fd));

  if (sendmsg(sockfd, &msg, 0) < 0) {
    fprintf(stderr, "sendmsg failed\n");
    return -1;
  }
  return 0;
}

static inline int receive_fd(int sockfd, int* fd) {
  assert(sockfd >= 0);
  struct msghdr msg;
  struct iovec iov;
  char buf[CMSG_SPACE(sizeof(int))];
  struct cmsghdr* cmsg;

  iov.iov_base = buf;
  iov.iov_len = sizeof(buf);

  msg.msg_name = 0;
  msg.msg_namelen = 0;

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);
  if (recvmsg(sockfd, &msg, 0) < 0) {
    perror("recvmsg failed\n");
    return -1;
  }
  cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg == NULL || cmsg->cmsg_type != SCM_RIGHTS) {
    perror("recvmsg failed\n");
    return -1;
  }
  *fd = *((int*)CMSG_DATA(cmsg));
  return 0;
}

static inline void* create_shm(char const* shm_name, size_t size) {
  int fd;
  void* addr;

  /* unlink it if we exit excpetionally before */
  shm_unlink(shm_name);

  fd = shm_open(shm_name, O_CREAT | O_RDWR | O_EXCL, 0666);
  if (fd == -1) {
    perror("shm_open");
    return MAP_FAILED;
  }

  if (ftruncate(fd, size) == -1) {
    perror("ftruncate");
    return MAP_FAILED;
  }

  addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    perror("mmap");
    return MAP_FAILED;
  }

  return addr;
}

static inline void destroy_shm(char const* shm_name, void* addr, size_t size) {
  munmap(addr, size);
  shm_unlink(shm_name);
}

static inline void* attach_shm(char const* shm_name, size_t size) {
  int fd;
  void* addr;

  fd = shm_open(shm_name, O_RDWR, 0);
  if (fd == -1) {
    perror("shm_open");
    return MAP_FAILED;
  }

  addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    perror("mmap");
    return MAP_FAILED;
  }

  return addr;
}

static inline void detach_shm(void* addr, size_t size) {
  if (munmap(addr, size) == -1) {
    perror("munmap");
    exit(EXIT_FAILURE);
  }
}

inline int IntRand(int const& min, int const& max) {
  static thread_local std::mt19937 generator(std::random_device{}());
  // Do not use "static thread_local" for distribution object, as this will
  // corrupt objects with different min/max values. Note that this object is
  // extremely cheap.
  std::uniform_int_distribution<int> distribution(min, max);
  return distribution(generator);
}

inline uint32_t U32Rand(uint32_t const& min, uint32_t const& max) {
  static thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<uint32_t> distribution(min, max);
  return distribution(generator);
}

inline uint64_t U64Rand(uint64_t const& min, uint64_t const& max) {
  static thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<uint64_t> distribution(min, max);
  return distribution(generator);
}

inline double FloatRand(double const& min, double const& max) {
  static thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<double> distribution(min, max);
  return distribution(generator);
}

inline std::string GetEnvVar(std::string const& key) {
  char* val = getenv(key.c_str());
  return val == NULL ? std::string("") : std::string(val);
}

inline uint64_t get_monotonic_time_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000LL + (uint64_t)ts.tv_nsec;
}

struct ib_dev {
  char prefix[64];
  int port;
};

static bool match_if(char const* string, char const* ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool match_port(int const port1, int const port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}

static bool match_if_list(char const* string, int port, struct ib_dev* ifList,
                          int listSize, bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i = 0; i < listSize; i++) {
    if (match_if(string, ifList[i].prefix, matchExact) &&
        match_port(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

static inline int parse_interfaces(char const* string, struct ib_dev* ifList,
                                   int maxList) {
  if (!string) return 0;

  char const* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr + 1);
        ifNum++;
        ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++;
        ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}
typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePoint;

#ifdef USE_CUDA
inline void checkMemoryLocation(void* ptr) {
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  if (err == cudaSuccess) {
    if (attributes.type == cudaMemoryTypeDevice) {
      LOG(INFO) << "Memory belongs to GPU " << attributes.device << std::endl;
    } else if (attributes.type == cudaMemoryTypeHost) {
      LOG(INFO) << "Memory is allocated on the Host (CPU)." << std::endl;
    } else {
      LOG(INFO) << "Unknown memory type." << std::endl;
    }
  } else {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
  }
}
#endif

}  // namespace uccl
