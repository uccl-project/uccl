#ifndef TRANSPORT_CONFIG_H
#define TRANSPORT_CONFIG_H

#include <cstdint>

// #define USING_TCP
#define USING_MULTIPATH

static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t SEND_BATCH_SIZE = 32;
static const uint32_t NUM_QUEUES = 1;
// 4GB frame pool in total; exceeding will cause crash.
static const uint64_t NUM_FRAMES = 1024 * 1024;
static const uint32_t ENGINE_CPU_START = 0;
static const uint16_t BASE_PORT = 10000;

#if !defined(AWS_ENA) && !defined(CLOUDLAB_MLX5)
#define CLOUDLAB_MLX5
#endif

#ifdef AWS_ENA
// On AWS ENA (eg, c5, g4dn)
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "ens6";
// static const char* DEV_DEFAULT = "enp199s0";
#elif defined(CLOUDLAB_MLX5)
// On Cloudlab (xl170-ubuntu24-v6.8 profile)
static const uint32_t AFXDP_MTU = 1500;
static const char* DEV_DEFAULT = "ens1f1np1";
#else
static_assert(false, "Please specify the platform in make");
#endif

#endif  // TRANSPORT_CONFIG_H