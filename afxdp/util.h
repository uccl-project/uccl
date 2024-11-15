#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <ifaddrs.h>
#include <linux/ethtool.h>
#include <linux/in.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "transport_config.h"
#include "util_jring.h"

namespace uccl {

template <class T>
static inline T Percentile(std::vector<T>& vectorIn, double percent) {
    if (vectorIn.size() == 0) return (T)0;
    auto nth = vectorIn.begin() + (percent * vectorIn.size()) / 100;
    std::nth_element(vectorIn.begin(), nth, vectorIn.end());
    return *nth;
}

static inline uint16_t ipv4_checksum(const void* data, size_t header_length) {
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
#define ACCESS_ONCE(x) (*(volatile decltype(x)*)&(x))
#define is_power_of_two(x) ((x) != 0 && !((x) & ((x) - 1)))

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

static inline jring_t* create_ring(size_t element_size, size_t element_count) {
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

static inline uint64_t rdtsc(void) {
    uint32_t a, d;
    asm volatile("rdtsc" : "=a"(a), "=d"(d));
    return ((uint64_t)a) | (((uint64_t)d) << 32);
}

static inline double rdtsc_to_us(uint64_t tsc) {
    static double ghz = 0;
    // TODO(yang): auto detect the CPU frequency
    if (unlikely(!ghz)) ghz = 3.0;
    return tsc / ghz / 1000.0;
}

// 0x04030201 -> 1.2.3.4
static inline std::string ip_to_str(uint32_t ip) {
    struct sockaddr_in sa;
    char str[INET_ADDRSTRLEN];
    sa.sin_addr.s_addr = ip;
    inet_ntop(AF_INET, &(sa.sin_addr), str, INET_ADDRSTRLEN);
    return std::string(str);
}

// 1.2.3.4 -> 0x04030201
static inline uint32_t str_to_ip(const std::string& ip) {
    struct sockaddr_in sa;
    DCHECK(inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr)) != 0);
    return sa.sin_addr.s_addr;
}

// Return -1 if not found
static inline int get_dev_index(const char* dev_name) {
    int ret = -1;
    struct ifaddrs* addrs;
    CHECK(getifaddrs(&addrs) == 0) << "error: getifaddrs failed";

    for (struct ifaddrs* iap = addrs; iap != NULL; iap = iap->ifa_next) {
        if (iap->ifa_addr && (iap->ifa_flags & IFF_UP) &&
            iap->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in* sa = (struct sockaddr_in*)iap->ifa_addr;
            if (strcmp(dev_name, iap->ifa_name) == 0) {
                LOG(INFO) << "found network interface: " << iap->ifa_name;
                ret = if_nametoindex(iap->ifa_name);
                CHECK(ret) << "error: if_nametoindex failed";
                break;
            }
        }
    }

    freeifaddrs(addrs);
    return ret;
}

static inline std::string get_dev_ip(const char* dev_name) {
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
            VLOG(3) << Format("%s IP Address %s\n", ifa->ifa_name,
                              addressBuffer);
            return std::string(addressBuffer);
        } else if (ifa->ifa_addr->sa_family == AF_INET6) {  // check it is IP6
            // is a valid IP6 Address
            tmpAddrPtr = &((struct sockaddr_in6*)ifa->ifa_addr)->sin6_addr;
            char addressBuffer[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
            VLOG(3) << Format("%s IP Address %s\n", ifa->ifa_name,
                              addressBuffer);
            return std::string(addressBuffer);
        }
    }
    if (ifAddrStruct != NULL) freeifaddrs(ifAddrStruct);
    return std::string();
}

// Function to convert MAC string to hex char array
static inline bool str_to_mac(const std::string& macStr, char mac[6]) {
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
static inline std::string mac_to_str(const char mac[6]) {
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

static inline std::string get_dev_mac(const char* dev_name) {
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
    const char* name = "fd";
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

static inline void* create_shm(const char* shm_name, size_t size) {
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

static inline void destroy_shm(const char* shm_name, void* addr, size_t size) {
    munmap(addr, size);
    shm_unlink(shm_name);
}

static inline void* attach_shm(const char* shm_name, size_t size) {
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

inline int IntRand(const int& min, const int& max) {
    static thread_local std::mt19937 generator(std::random_device{}());
    // Do not use "static thread_local" for distribution object, as this will
    // corrupt objects with different min/max values. Note that this object is
    // extremely cheap.
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

inline uint64_t U64Rand(const uint64_t& min, const uint64_t& max) {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<uint64_t> distribution(min, max);
    return distribution(generator);
}

inline double FloatRand(const double& min, const double& max) {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(generator);
}

inline std::string GetEnvVar(std::string const& key) {
    char* val = getenv(key.c_str());
    return val == NULL ? std::string("") : std::string(val);
}

// Function to retrieve the redirection table and RSS key
inline bool get_rss_config(const std::string& interface_name,
                           std::vector<uint32_t>& redir_table,
                           std::vector<uint8_t>& rss_key) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return false;
    }

    // Prepare structures for ioctl
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, interface_name.c_str(), IFNAMSIZ - 1);

    // Step 1: Query the sizes of the RSS key and redirection table
    struct ethtool_rxfh* rss_query =
        (struct ethtool_rxfh*)malloc(sizeof(struct ethtool_rxfh));
    memset(rss_query, 0, sizeof(struct ethtool_rxfh));
    rss_query->cmd = ETHTOOL_GRSSH;  // Get RSS configuration command

    ifr.ifr_data = reinterpret_cast<char*>(rss_query);

    if (ioctl(sockfd, SIOCETHTOOL, &ifr) < 0) {
        perror("Failed to query RSS configuration sizes");
        free(rss_query);
        close(sockfd);
        return false;
    }

    uint32_t indir_size = rss_query->indir_size;
    uint32_t key_size = rss_query->key_size;
    free(rss_query);

    LOG(INFO) << "Interface " << interface_name
              << ": RSS indirection table size " << indir_size
              << ", RSS key size " << key_size;

    if (indir_size == 0 && key_size == 0) {
        std::cerr << "RSS configuration not supported on this NIC."
                  << std::endl;
        close(sockfd);
        return false;
    }

    // Step 2: Allocate memory for the full RSS configuration
    size_t struct_size =
        sizeof(struct ethtool_rxfh) + indir_size * sizeof(uint32_t) + key_size;
    struct ethtool_rxfh* rss_config = (struct ethtool_rxfh*)malloc(struct_size);
    memset(rss_config, 0, struct_size);

    rss_config->cmd = ETHTOOL_GRSSH;
    rss_config->indir_size = indir_size;
    rss_config->key_size = key_size;

    ifr.ifr_data = reinterpret_cast<char*>(rss_config);

    // Step 3: Perform ioctl to retrieve the RSS configuration
    if (ioctl(sockfd, SIOCETHTOOL, &ifr) < 0) {
        perror("Failed to retrieve RSS configuration");
        free(rss_config);
        close(sockfd);
        return false;
    }

    // Copy the redirection table
    if (indir_size > 0) {
        uint32_t* indir_table_start = (uint32_t*)rss_config->rss_config;
        redir_table.assign(indir_table_start, indir_table_start + indir_size);
    }

    // Copy the RSS key
    if (key_size > 0) {
        uint8_t* key_start =
            (uint8_t*)((uint32_t*)rss_config->rss_config + indir_size);
        rss_key.assign(key_start, key_start + key_size);
    }

    free(rss_config);
    close(sockfd);
    return true;
}

// Function to compute Toeplitz hash
inline uint32_t compute_rss_hash(const std::vector<uint8_t>& tuple,
                                 const std::vector<uint8_t>& rss_key) {
    uint32_t hash = 0;

    // Iterate through each bit in the tuple
    for (size_t i = 0; i < tuple.size() * 8; ++i) {
        if (tuple[i / 8] & (1 << (7 - (i % 8)))) {
            hash ^=
                ((uint32_t*)
                     rss_key.data())[i % (rss_key.size() / sizeof(uint32_t))];
        }
    }

    return hash;
}

// Function to calculate queue ID directly from IPs and ports
inline uint32_t calculate_queue_id(uint32_t src_ip, uint32_t dst_ip,
                                   uint16_t src_port, uint16_t dst_port,
                                   const std::vector<uint8_t>& rss_key,
                                   const std::vector<uint32_t>& redir_table) {
    // Convert IPs and ports to network byte order
    src_ip = htonl(src_ip);
    dst_ip = htonl(dst_ip);
    src_port = htons(src_port);
    dst_port = htons(dst_port);

    // Combine the tuple fields into a single vector
    std::vector<uint8_t> tuple;
    tuple.insert(tuple.end(), (uint8_t*)&src_ip,
                 (uint8_t*)&src_ip + sizeof(src_ip));
    tuple.insert(tuple.end(), (uint8_t*)&dst_ip,
                 (uint8_t*)&dst_ip + sizeof(dst_ip));
    tuple.insert(tuple.end(), (uint8_t*)&src_port,
                 (uint8_t*)&src_port + sizeof(src_port));
    tuple.insert(tuple.end(), (uint8_t*)&dst_port,
                 (uint8_t*)&dst_port + sizeof(dst_port));

    // Step 1: Calculate the RSS hash
    uint32_t rss_hash = compute_rss_hash(tuple, rss_key);

    // Step 2: Map the hash to the redirection table
    uint32_t queue_id = redir_table[rss_hash % redir_table.size()];

    return queue_id;
}

inline bool get_dst_ports_with_target_queueid(
    uint32_t src_ip, uint32_t dst_ip, uint16_t src_port,
    uint32_t target_queue_id, const std::vector<uint8_t>& rss_key,
    const std::vector<uint32_t>& redir_table, int num_dst_ports,
    std::vector<uint16_t>& dst_ports) {
    for (int port = BASE_PORT; port < 65536; port++) {
        uint16_t dst_port = port;
        uint32_t queue_id = calculate_queue_id(src_ip, dst_ip, src_port,
                                               dst_port, rss_key, redir_table);
        if (queue_id == target_queue_id) {
            dst_ports.push_back(dst_port);
            if (dst_ports.size() == num_dst_ports) {
                return true;
            }
        }
    }
    return false;
}
}  // namespace uccl
