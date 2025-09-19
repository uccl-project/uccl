#pragma once
#include "util/util.h"
#include <cstdint>
#include <thread>

#define USE_MULTIPATH
#define PATH_SELECTION
#define REXMIT_SET_PATH
#define THREADED_MEMCPY
// #define EMULATE_ZC
// #define USE_TCP
// #define ENABLE_CSUM
// #define RTT_STATS

enum class CCType {
  kTimely,
  kTimelyPP,
  kCubic,
  kCubicPP,
};
static constexpr CCType kCCType = CCType::kCubicPP;

#if !defined(AWS_C5) && !defined(AWS_G4) && !defined(AWS_G4METAL) &&     \
    !defined(CLAB_XL170) && !defined(CLAB_D6515) && !defined(IBM_GX3) && \
    !defined(TPU_V6E8)
#define AWS_C5
#endif

#if defined(AWS_C5)
static const uint32_t AFXDP_MTU = 3498;
static char const* DEV_DEFAULT = "ens6";
static double const kLinkBandwidth = 100.0 * 1e9 / 8;
static const uint32_t NUM_QUEUES = 12;  // 5/12 for uni/dual.
static const uint32_t kMaxPath = 200;   // 256/200 for uni/dual.
static const uint32_t kMaxUnackedPktsPP = 1u;
#elif defined(AWS_G4)
static const uint32_t AFXDP_MTU = 3498;
static char const* DEV_DEFAULT = "ens6";
static double const kLinkBandwidth = 50.0 * 1e9 / 8;
static const uint32_t NUM_QUEUES = 8;
static const uint32_t kMaxPath = 256;
static const uint32_t kMaxUnackedPktsPP = 1u;
#elif defined(AWS_G4METAL)
static const uint32_t AFXDP_MTU = 3498;
static char const* DEV_DEFAULT = "enp199s0";
static double const kLinkBandwidth = 100.0 * 1e9 / 8;
static const uint32_t NUM_QUEUES = 12;
static const uint32_t kMaxPath = 128;
static const uint32_t kMaxUnackedPktsPP = 3u;
#elif defined(CLAB_XL170)
static const uint32_t AFXDP_MTU = 1500;
static char const* DEV_DEFAULT = "ens1f1np1";
static double const kLinkBandwidth = 25.0 * 1e9 / 8;
static const uint32_t NUM_QUEUES = 2;
static const uint32_t kMaxPath = 64;
static const uint32_t kMaxUnackedPktsPP = 8u;
#elif defined(CLAB_D6515)
static const uint32_t AFXDP_MTU = 3498;
static char const* DEV_DEFAULT = "enp65s0f0np0";
static double const kLinkBandwidth = 100.0 * 1e9 / 8;
static const uint32_t NUM_QUEUES = 4;
static const uint32_t kMaxPath = 64;
static const uint32_t kMaxUnackedPktsPP = 16u;
#elif defined(IBM_GX3)
static const uint32_t AFXDP_MTU = 1500;
static char const* DEV_DEFAULT = "ens3";
static double const kLinkBandwidth = 48.0 * 1e9 / 8;
static const uint32_t NUM_QUEUES = 6;
static const uint32_t kMaxPath = 64;
static const uint32_t kMaxUnackedPktsPP = 1u;
#elif defined(TPU_V6E8)
static const uint32_t AFXDP_MTU = 1460;
static char const* DEV_DEFAULT = "ens8";
static double const kLinkBandwidth = 200.0 * 1e9 / 8;
static const uint32_t NUM_QUEUES = 1;
static const uint32_t kMaxPath = 64;
static const uint32_t kMaxUnackedPktsPP = 1u;
#endif

static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
static const uint16_t BASE_PORT = 10000;
// 4GB frame pool in total; exceeding will cause crash.
static const uint64_t NUM_FRAMES = 1024 * 1024;
static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t SEND_BATCH_SIZE = 32;

// CC parameters.
static const std::size_t kSackBitmapSize = 1024;
static const std::size_t kFastRexmitDupAckThres = 10;
static const uint32_t kMaxTwPkts = 1024;
static double const kMaxBwPP = 5.0 * 1e9 / 8;
static const uint32_t kSwitchPathThres = 1u;
static const uint32_t kMaxUnackedPktsPerEngine = kMaxUnackedPktsPP * kMaxPath;
static const uint32_t kMaxPathHistoryPerEngine = 4096;

static_assert(kMaxPath <= 4096, "kMaxPath too large");
static_assert(NUM_QUEUES <= 16, "NUM_QUEUES too large");
static_assert(kMaxUnackedPktsPerEngine <= kMaxPathHistoryPerEngine,
              "kMaxUnackedPktsPerEngine too large");
static_assert(is_power_of_two(kMaxPathHistoryPerEngine),
              "kMaxPathHistoryPerEngine must be power of 2");

#ifdef CLAB_XL170
// TODO(yang): why XL170 would crash with 1x fill ring size?
#define FILL_RING_SIZE (XSK_RING_PROD__DEFAULT_NUM_DESCS * 2)
#else
// TODO(yang): why C5 would crash with 2x fill ring size?
#define FILL_RING_SIZE XSK_RING_PROD__DEFAULT_NUM_DESCS
#endif
#define COMP_RING_SIZE XSK_RING_CONS__DEFAULT_NUM_DESCS
#define TX_RING_SIZE XSK_RING_PROD__DEFAULT_NUM_DESCS
#define RX_RING_SIZE XSK_RING_CONS__DEFAULT_NUM_DESCS
