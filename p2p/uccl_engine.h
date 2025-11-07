#pragma once

#include <vector>
#include <stddef.h>
#include <stdint.h>

#define MSG_SIZE 64
// Handle for the UCCL engine instance
typedef struct uccl_engine uccl_engine_t;

// Handle for a connection
typedef struct uccl_conn uccl_conn_t;

// Handle for a memory region
typedef uint64_t uccl_mr_t;

// UCCL operation types
enum uccl_msg_type {
  UCCL_READ = 0,
  UCCL_WRITE = 1,
  UCCL_VECTOR_READ = 2,
  UCCL_VECTOR_WRITE = 3,
  UCCL_FIFO = 4,
  UCCL_VECTOR_FIFO = 5,
  UCCL_NOTIFY = 6
};

typedef struct notify_msg {
  char name[MSG_SIZE];
  char msg[MSG_SIZE];
} notify_msg_t;

typedef struct fifo_msg {
  int id;
  char fifo_buf[MSG_SIZE];
} fifo_msg_t;

typedef struct fifo_v_msg {
  char fifo_buf[MSG_SIZE];
} fifo_v_msg_t;

typedef struct tx_msg {
  uint64_t data_ptr;  // Memory address for data reception
  size_t data_size;   // Size of data to receive
} tx_msg_t;

typedef struct vector_msg {
  size_t count;  // Number of items in the vector
  int id;        // optional ID for a vector
} vector_msg_t;

typedef struct md {
  uccl_msg_type op;
  union {
    tx_msg_t tx_data;
    fifo_msg_t fifo_data;
    notify_msg_t notify_data;
    vector_msg_t vector_data;
  } data;
} md_t;

typedef struct metadata {
  uccl_msg_type op;
  char fifo_buf[64];
  uint64_t data_ptr;
  size_t data_size;
} metadata_t;

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
 * @return              Memory region handle, or -1 on failure.
 */
uccl_mr_t uccl_engine_reg(uccl_engine_t* engine, uintptr_t data, size_t size);

/**
 * Read data (Non blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @param fifo_id       FIFO ID.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_read(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                     size_t size, int fifo_id, uint64_t* transfer_id);

/**
 * Read a vector of data chunks (Non blocking).
 * @param conn          Connection handle.
 * @param mr_ids        Vector of memory region handles.
 * @param dst_v         Vector of pointers to the data to receive.
 * @param size_v        Vector of sizes of the data to receive.
 * @param fifo_id       FIFO ID.
 * @param num_iovs      Number of IO vectors.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_read_vector(uccl_conn_t* conn, std::vector<uint64_t> mr_ids,
                            std::vector<void*> dst_v,
                            std::vector<size_t> size_v, int fifo_id,
                            int num_iovs, uint64_t* transfer_id);
/**
 * Wait for the FIFO vector to be available.
 * @param id            FIFO vector ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_wait_for_fifo(int id);
/**
 * Send data (Non blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                      size_t size, uint64_t* transfer_id);

/**
 * Read a vector of data chunks (Non blocking).
 * @param conn          Connection handle.
 * @param mr_ids        Vector of memory region handles.
 * @param src_v         Vector of pointers to the data to write.
 * @param num_iovs      Number of IO vectors.
 * @param size_v        Vector of sizes of the data to write.
 * @param transfer_id   Pointer to store the transfer ID.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_write_vector(uccl_conn_t* conn, std::vector<uint64_t> mr_ids,
                             std::vector<void const*> src_v,
                             std::vector<size_t> size_v, int num_iovs,
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
 * Receive a vector of data chunks (blocking).
 * @param conn          Connection handle.
 * @param mr_ids        Vector of memory region handles.
 * @param data_v        Vector of pointers to the buffer to receive data.
 * @param size_v        Vector of sizes of the data to receive.
 * @param num_iovs      Number of IO vectors.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_recv_vector(uccl_conn_t* conn, std::vector<uint64_t> mr_ids,
                            std::vector<void*> data_v,
                            std::vector<size_t> size_v, int num_iovs);
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
void uccl_engine_mr_destroy(uccl_mr_t mr);

/**
 * Get endpoint metadata for connection establishment.
 * @param engine        The engine instance.
 * @param metadata      Pointer to store the metadata in string.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata_str);

/**
 * Send the transfer metadata.
 * @param conn          Connection handle.
 * @param md            Transfer metadata.
 * @return              Number of bytes sent, or -1 on failure.
 */
int uccl_engine_send_tx_md(uccl_conn_t* conn, md_t* md);

/**
 * Send multiple transfer metadata as a vector.
 * @param conn          Connection handle.
 * @param md_array      Array of transfer metadata.
 * @param count         Number of metadata items in the array.
 * @return              Number of bytes sent, or -1 on failure.
 */
int uccl_engine_send_tx_md_vector(uccl_conn_t* conn, md_t* md_array,
                                  size_t count);

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
 * Get socket file descriptor for a connection.
 * @param conn          Connection handle.
 * @return              Socket file descriptor, or -1 on failure.
 */
int uccl_engine_get_sock_fd(uccl_conn_t* conn);

/**
 * Free endpoint metadata buffer.
 * @param metadata      The metadata buffer to free.
 */
void uccl_engine_free_endpoint_metadata(uint8_t* metadata);
