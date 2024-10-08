#pragma once

#include <stdint.h>

namespace uccl {

const uint8_t SERVER_ETHERNET_ADDRESS[] = {0x0a, 0xff, 0xea, 0x86, 0x04, 0xd9};
const uint8_t CLIENT_ETHERNET_ADDRESS[] = {0x0a, 0xff, 0xdf, 0x30, 0xe7, 0x59};
const uint32_t SERVER_IPV4_ADDRESS = 0xac1f16f9;  // 172.31.22.249
const uint32_t CLIENT_IPV4_ADDRESS = 0xac1f10c6;  // 172.31.16.198
const uint16_t SERVER_PORT = 40000;
const uint16_t CLIENT_PORT = 40001;
const uint32_t RECV_BATCH_SIZE = 32;

}  // namespace uccl