/**
 * @file tcpx_interface.h
 * @brief C API wrapper for NCCL GPUDirect TCPX plugin
 *
 * Provides GPU-to-GPU data transfer over TCP with zero-copy devmem-tcp support.
 * Wraps NCCL plugin v7 API for connection management, memory registration,
 * and async send/receive operations.
 */

#pragma once

#include <cstddef>
#include <cstdint>

// NCCL memory type constants
#define NCCL_PTR_HOST 0x1  // Host (CPU) memory
#define NCCL_PTR_CUDA 0x2  // Device (GPU) memory

extern "C" {

// Plugin initialization
int tcpx_get_device_count();
int tcpx_load_plugin(char const* plugin_path);

// Connection management
int tcpx_listen(int dev, void* handle, void** listen_comm);
int tcpx_connect_v5(int dev, void* handle, void** send_comm,
                    void** send_dev_handle);
int tcpx_accept_v5(void* listen_comm, void** recv_comm, void** recv_dev_handle);

// Memory registration (for CUDA buffers, sets up devmem-tcp mappings)
int tcpx_reg_mr(void* comm, void* data, size_t size, int type, void** mhandle);
int tcpx_dereg_mr(void* comm, void* mhandle);

// Async data transfer
int tcpx_isend(void* send_comm, void* data, int size, int tag, void* mhandle,
               void** request);
int tcpx_irecv(void* recv_comm, int n, void** data, int* sizes, int* tags,
               void** mhandles, void** request);
int tcpx_test(void* request, int* done, int* size);  // Poll for completion

// Mark GPU receive as consumed (allows plugin to reuse bounce buffers)
int tcpx_irecv_consumed(void* comm, int n, void* request);

// Cleanup
int tcpx_close_send(void* send_comm);
int tcpx_close_recv(void* recv_comm);
int tcpx_close_listen(void* listen_comm);
}
