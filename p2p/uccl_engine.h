#pragma once

#include "common.h"
#include <vector>
#include <stddef.h>
#include <stdint.h>

#define MSG_SIZE 256
#define FIFO_SIZE 64
// Handle for the UCCL engine instance
typedef struct uccl_engine uccl_engine_t;

// Handle for a connection
typedef struct uccl_conn uccl_conn_t;

// Handle for a memory region
typedef uint64_t uccl_mr_t;

typedef struct notify_msg {
  char name[MSG_SIZE];
  char msg[MSG_SIZE];
} notify_msg_t;

typedef struct md {
  notify_msg_t notify_data;
} md_t;

/**
 * Create and initialize an engine instance.
 * @param num_cpus      The number of CPUs to use for the engine.
 * @param in_python     Whether the engine is being created in Python.
 * @return              Pointer to the engine instance, or NULL on failure.
 */
uccl_engine_t* uccl_engine_create(int num_cpus, bool in_python);

/**
 * Destroy the engine instance and free resources.
 * @param engine        The engine instance to destroy.
 */
void uccl_engine_destroy(uccl_engine_t* engine);

/**
 * Connect to a remote peer.
 * @param engine        The engine instance.
 * @param ip_addr       The IP address of the remote server.
 * @param remote_gpu_idx The GPU index of the remote server.
 * @param remote_port   The port of the remote server.
 * @return              Connection handle, or NULL on failure.
 */
uccl_conn_t* uccl_engine_connect(uccl_engine_t* engine, char const* ip_addr,
                                 int remote_gpu_idx, int remote_port);
/**
 * Start the listener thread for the connection.
 * @param conn          Connection handle.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_start_listener(uccl_conn_t* conn);
/**
 * Accept an incoming connection (blocking).
 * @param engine        The engine instance.
 * @param ip_addr_buf   Buffer to store the IP address of the remote server.
 * @param ip_addr_buf_len Length of the buffer.
 * @param remote_gpu_idx Pointer to store the GPU index of the remote server.
 * @return              Connection handle, or NULL on failure.
 */
uccl_conn_t* uccl_engine_accept(uccl_engine_t* engine, char* ip_addr_buf,
                                size_t ip_addr_buf_len, int* remote_gpu_idx);

/**
 * Register a memory region for RDMA.
 * @param engine        The engine instance.
 * @param data          Pointer to the data to register.
 * @param size          Size of the data.
 * @param mr_id         Pointer to store the memory region handle.
 * @return              Memory region handle, or NULL on failure.
 */
int uccl_engine_reg(uccl_engine_t* engine, uintptr_t data, size_t size,
                    uccl_mr_t& mr_id);

/**
 * Read data (Non blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @param fifo_item     Fifo item for RDMA read.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_read(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                     size_t size, FifoItem fifo_item, uint64_t* transfer_id);

/**
 * Read a vector of data chunks (Non blocking).
 * @param conn          Connection handle.
 * @param mr_ids        Vector of memory region handles.
 * @param dst_v         Vector of pointers to the data to receive.
 * @param size_v        Vector of sizes of the data to receive.
 * @param fifo_items    Vector of FifoItem structs for RDMA operations.
 * @param num_iovs      Number of IO vectors.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_read_vector(uccl_conn_t* conn, std::vector<uccl_mr_t> mr_ids,
                            std::vector<void*> dst_v,
                            std::vector<size_t> size_v,
                            std::vector<FifoItem> fifo_items, int num_iovs,
                            uint64_t* transfer_id);
/**
 * Send data (Non blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_send(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                      size_t size, uint64_t* transfer_id);

/**
 * Send a vector of data chunks (Non blocking).
 * @param conn          Connection handle.
 * @param mr_ids        Vector of memory region handles.
 * @param src_v         Vector of pointers to the data to write.
 * @param num_iovs      Number of IO vectors.
 * @param size_v        Vector of sizes of the data to write.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_send_vector(uccl_conn_t* conn, std::vector<uccl_mr_t> mr_ids,
                             std::vector<void const*> src_v,
                             std::vector<size_t> size_v, int num_iovs,
                             uint64_t* transfer_id);
/**
 * Send data with RC mode (Non blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @param fifo_item     Fifo item.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                         size_t size, FifoItem fifo_item,
                         uint64_t* transfer_id);
/**
 * Send a vector of data chunks with RC mode (Non blocking).
 * @param conn          Connection handle.
 * @param mr_ids        Vector of memory region handles.
 * @param dst_v         Vector of pointers to the data to write.
 * @param size_v        Vector of sizes of the data to write.
 * @param fifo_items    Vector of FifoItem structs for RDMA operations.
 * @param num_iovs      Number of IO vectors.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_write_vector(uccl_conn_t* conn,
                                std::vector<uccl_mr_t> mr_ids,
                                std::vector<void*> dst_v,
                                std::vector<size_t> size_v,
                                std::vector<FifoItem> fifo_items, int num_iovs,
                                uint64_t* transfer_id);
/**
 * Receive data (blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the buffer to receive data.
 * @param max_size      Maximum size of the buffer.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t mr, void* data,
                     size_t max_size);
/**
 * Check the status of a transfer.
 * @param conn          Connection handle.
 * @param transfer_id   Transfer ID.
 * @return              True if the transfer is done, false otherwise.
 */
bool uccl_engine_xfer_status(uccl_conn_t* conn, uint64_t transfer_id);
/**
 * Cleanup connection.
 * @param conn          Connection handle to destroy.
 */
void uccl_engine_conn_destroy(uccl_conn_t* conn);

/**
 * Deregister memory region.
 * @param engine        The engine instance.
 * @param mr            Memory region handle to destroy.
 */
void uccl_engine_mr_destroy(uccl_engine_t* engine, uccl_mr_t mr);

/**
 * Get endpoint metadata for connection establishment.
 * @param engine        The engine instance.
 * @param metadata      Pointer to store the metadata in string.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata_str);

/**
 * Get all notification messages and clear the list.
 * @return              Vector of notification messages.
 */
std::vector<notify_msg_t> uccl_engine_get_notifs();

/**
 * Send a notification message.
 * @param conn          Connection handle.
 * @param notify_msg    Notification message.
 * @return              Number of bytes sent, or -1 on failure.
 */
int uccl_engine_send_notif(uccl_conn_t* conn, notify_msg_t* notify_msg);

/**
 * Prepare FIFO metadata for a memory region without requiring a connection.
 * This can be called at memory registration time to pre-compute the fifo_item
 * for true one-sided RDMA operations.
 * @param engine        The engine instance.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data buffer.
 * @param size          Size of the data.
 * @param fifo_buf      Output buffer to store the serialized FifoItem (64
 * bytes).
 * @return              0 on success, -1 on failure.
 */
int uccl_engine_prepare_fifo(uccl_engine_t* engine, uccl_mr_t mr,
                             void const* data, size_t size, char* fifo_buf);

/**
 * Update the address and size in a FIFO item.
 * @param fifo_buf      Pointer to the FIFO item buffer (64 bytes).
 * @param remote_addr   New remote address to set.
 * @param size          New size to set.
 * @return              0 on success, -1 on failure.
 */
int uccl_engine_update_fifo(FifoItem& fifo_item, uint64_t remote_addr,
                            uint32_t size);