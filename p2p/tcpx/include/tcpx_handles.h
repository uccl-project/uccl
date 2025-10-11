/**
 * @file tcpx_handles.h
 * @brief Shared TCPX handle definitions
 *
 * This header provides the canonical definition of ncclNetHandle_v7
 * to avoid ODR violations from per-file redefinitions.
 */

#pragma once

#include <cstdint>

/**
 * @brief NCCL network handle (version 7)
 *
 * Opaque 128-byte structure used by TCPX plugin for connection handshake.
 * Contains serialized connection information (IP, port, etc.).
 */
struct ncclNetHandle_v7 {
  char data[128];
};

// Ensure the handle is exactly 128 bytes as expected by TCPX
static_assert(sizeof(ncclNetHandle_v7) == 128,
              "ncclNetHandle_v7 must be exactly 128 bytes");
