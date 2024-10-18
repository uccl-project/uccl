#ifndef TRANSPORT_CONFIG_H
#define TRANSPORT_CONFIG_H

#include <cstdint>

#define RECV_BATCH_SIZE 32
#define SEND_BATCH_SIZE 32

// #define USING_TCP
#define USING_MULTIPATH

// static const uint8_t server_ethernet_address[] = {0x0a, 0xff, 0xea,
//                                                   0x86, 0x04, 0xd9};
// static const uint8_t client_ethernet_address[] = {0x0a, 0xff, 0xdf,
//                                                   0x30, 0xe7, 0x59};
// static const char* server_addr_str = "172.31.22.249";
// static const char* client_addr_str = "172.31.16.198";

static const uint8_t server_ethernet_address[] = {0x0a, 0xff, 0xea,
                                                  0x11, 0x5e, 0xad};
static const uint8_t client_ethernet_address[] = {0x0a, 0xff, 0xc6,
                                                  0x7f, 0xc9, 0x35};
static const char* server_addr_str = "172.31.18.199";
static const char* client_addr_str = "172.31.25.5";

static const uint16_t server_port = 40000;
static const uint16_t client_port = 40000;

#endif  // TRANSPORT_CONFIG_H