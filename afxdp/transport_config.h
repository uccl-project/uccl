#pragma once
#include <cstdint>
#include <thread>

#include "util.h"

#define USE_MULTIPATH
#define PATH_SELECTION
#define LATENCY_CC
// #define PERPATH_CUBIC
#define EMULATE_ZC
// #define USE_TCP
// #define ENABLE_CSUM
// #define RTT_STATS

static const uint32_t NUM_QUEUES = 4;
static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
static const uint16_t BASE_PORT = 10000;
// 4GB frame pool in total; exceeding will cause crash.
static const uint64_t NUM_FRAMES = 1024 * 1024;
static const uint32_t RECV_BATCH_SIZE = 32;

// CC parameters.
static const uint32_t kPortEntropy = 32;
static const std::size_t kSackBitmapSize = 1024;
static const std::size_t kFastRexmitDupAckThres = 5;
static const uint32_t kMaxUnackedPkts = 128 * kPortEntropy;
static const uint32_t kMaxTwPkts = 256 * kPortEntropy;

static_assert(is_power_of_two(kMaxUnackedPkts),
              "kMaxUnackedPkts must be power of 2");
static_assert(kPortEntropy <= 256, "kPortEntropy too large");
static_assert(is_power_of_two(kPortEntropy), "kPortEntropy must be power of 2");

#ifdef CLOUDLAB_XL170
// TODO(yang): why XL170 would crash with 1x fill ring size?
#define FILL_RING_SIZE (XSK_RING_PROD__DEFAULT_NUM_DESCS * 2)
#else
// TODO(yang): why C5 would crash with 2x fill ring size?
#define FILL_RING_SIZE (XSK_RING_PROD__DEFAULT_NUM_DESCS)
#endif
#define COMP_RING_SIZE XSK_RING_CONS__DEFAULT_NUM_DESCS
#define TX_RING_SIZE XSK_RING_PROD__DEFAULT_NUM_DESCS
#define RX_RING_SIZE XSK_RING_CONS__DEFAULT_NUM_DESCS

#if defined(AWS_C5)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "ens6";
static const double kLinkBandwidth = 100.0 * 1e9 / 8;
#elif defined(AWS_G4)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "ens6";
static const double kLinkBandwidth = 50.0 * 1e9 / 8;
#elif defined(AWS_G4METAL)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "enp199s0";
static const double kLinkBandwidth = 100.0 * 1e9 / 8;
#elif defined(CLOUDLAB_XL170)
static const uint32_t AFXDP_MTU = 1500;
static const char* DEV_DEFAULT = "ens1f1np1";
static const double kLinkBandwidth = 25.0 * 1e9 / 8;
#elif defined(CLOUDLAB_D6515)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "enp65s0f0np0";
static const double kLinkBandwidth = 100.0 * 1e9 / 8;
#else
#define CLOUDLAB_XL170
static const uint32_t AFXDP_MTU = 1500;
static const char* DEV_DEFAULT = "ens1f1np1";
static const double kLinkBandwidth = 25.0 * 1e9 / 8;
#endif
