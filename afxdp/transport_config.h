#ifndef TRANSPORT_CONFIG_H
#define TRANSPORT_CONFIG_H

#include <cstdint>

// #define USING_TCP
// #define ENABLE_CSUM
#define USING_MULTIPATH

static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t SEND_BATCH_SIZE = 32;
static const uint32_t NUM_QUEUES = 4;
// 4GB frame pool in total; exceeding will cause crash.
static const uint64_t NUM_FRAMES = 1024 * 1024;
static const uint32_t ENGINE_CPU_START = 0;
static const uint16_t BASE_PORT = 10000;
static const uint32_t kPortEntropy = 128;
static const double kLinkBandwidth =
    100.0 * 1000 * 1000 * 1000 / 8;  // byte per second

#if defined(AWS_C5) || defined(AWS_G4)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "ens6";
#elif defined(AWS_G4_METAL)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "enp199s0";
#elif defined(CLOUDLAB_XL170)
static const uint32_t AFXDP_MTU = 1500;
static const char* DEV_DEFAULT = "ens1f1np1";
#elif defined(CLOUDLAB_D6515)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "enp65s0f0np0";
#else
#define CLOUDLAB_XL170
static const uint32_t AFXDP_MTU = 1500;
static const char* DEV_DEFAULT = "ens1f1np1";
#endif

#endif  // TRANSPORT_CONFIG_H