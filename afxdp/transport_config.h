#pragma once
#include <cstdint>

// #define USE_TCP
// #define ENABLE_CSUM
// #define RTT_STATS
// #define TEST_ZC
#define USE_MULTIPATH

static const uint32_t NUM_QUEUES = 1;
static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t MAX_UNACKED_PKTS = 512;
static const uint32_t MAX_TIMING_WHEEL_PKTS = 1024;
// 4GB frame pool in total; exceeding will cause crash.
static const uint64_t NUM_FRAMES = 1024 * 1024;
static const uint32_t ENGINE_CPU_START = 0;
static const uint16_t BASE_PORT = 10000;
// CC parameters.
static const uint32_t kPortEntropy = 128;
static const std::size_t kSackBitmapSize = 1024;
static const std::size_t kFastRexmitDupAckThres = 5;

#if defined(AWS_C5)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "ens6";
static const double kLinkBandwidth = 10.0 * 1e9 / 8;
#elif defined(AWS_G4)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "ens6";
static const double kLinkBandwidth = 50.0 * 1e9 / 8;
#elif defined(AWS_G4_METAL)
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
