#pragma once

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
typedef struct uccl_mr uccl_mr_t;

// UCCL operation types
enum uccl_msg_type {
  UCCL_WRITE = 1,
  UCCL_NOTIFY = 2
};

typedef struct tx_msg {
  uint64_t data_ptr;  // Memory address for data reception
  size_t data_size;   // Size of data to receive
} tx_msg_t;

typedef struct notify_msg {
  char name[MSG_SIZE];
  char msg[MSG_SIZE];
} notify_msg_t;

typedef struct md {
  uccl_msg_type op;
  union {
    tx_msg_t tx_data;
    notify_msg_t notify_data;
  } data;
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
 * @return              Memory region handle, or NULL on failure.
 */
uccl_mr_t* uccl_engine_reg(uccl_engine_t* engine, uintptr_t data, size_t size);

/**
 * Read data (Non blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @param slot_item     Pointer to FifoItem for RDMA read.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_read(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                     size_t size, void* slot_item, uint64_t* transfer_id);

/**
 * Send data (Non blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                      size_t size, uint64_t* transfer_id);

/**
 * Send data with RC mode (Non blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @param slot_item_ptr Pointer to the slot item.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_write_rc(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                         size_t size, void* slot_item_ptr,
                         uint64_t* transfer_id);
/**
 * Receive data (blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the buffer to receive data.
 * @param max_size      Maximum size of the buffer.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t* mr, void* data,
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
 * @param mr            Memory region handle to destroy.
 */
void uccl_engine_mr_destroy(uccl_mr_t* mr);

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
 * @param fifo_buf      Output buffer to store the serialized FifoItem (64 bytes).
 * @return              0 on success, -1 on failure.
 */
int uccl_engine_prepare_fifo(uccl_engine_t* engine, uccl_mr_t* mr,
                                    void const* data, size_t size, char* fifo_buf);