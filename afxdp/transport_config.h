#ifndef TRANSPORT_CONFIG_H
#define TRANSPORT_CONFIG_H

#include <cstdint>

static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t SEND_BATCH_SIZE = 32;
static const uint32_t QID_DEFAULT = 0;
static const uint32_t NUM_FRAMES = 4096 * 64;  // 1GB frame pool
static const uint32_t ENGINE_CPUID = 2;
static const uint16_t BASE_PORT = 40000;

// #define USING_TCP
#define USING_MULTIPATH

#ifdef AWS
// On AWS g4
#define AWS_ENA
static const uint32_t AFXDP_MTU = 3498;
static const char* DEV_DEFAULT = "ens5";
static const char* server_mac_str = "16:ff:d0:73:a9:cf";
static const char* client_mac_str = "16:ff:d9:ee:ab:47";
static const char* server_ip_str = "172.31.66.106";
static const char* client_ip_str = "172.31.72.149";
#elif defined(CLOUDLAB)
// On Cloudlab (xl170-ubuntu24-v6.8 profile)
#define CLOUDLAB_MLX5
static const uint32_t AFXDP_MTU = 1500;
static const char* DEV_DEFAULT = "ens1f1np1";
static const char* server_mac_str = "9c:dc:71:56:af:45";
static const char* client_mac_str = "9c:dc:71:5b:22:91";
static const char* server_ip_str = "192.168.6.1";
static const char* client_ip_str = "192.168.6.2";
#else
static_assert(false, "Please specify the platform in make");
#endif

#endif  // TRANSPORT_CONFIG_H