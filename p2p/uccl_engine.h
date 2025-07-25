// ... existing code ...
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Handle for the UCCL engine instance
typedef struct uccl_engine uccl_engine_t;

// Handle for a connection
typedef struct uccl_conn uccl_conn_t;

// Handle for a memory region
typedef struct uccl_mr uccl_mr_t;

/**
 * Create and initialize an engine instance.
 * @param local_gpu_idx The GPU index to use for the engine.
 * @param num_cpus      The number of CPUs to use for the engine.
 * @return              Pointer to the engine instance, or NULL on failure.
 */
uccl_engine_t* uccl_engine_create(int local_gpu_idx, int num_cpus);

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
uccl_mr_t* uccl_engine_reg(uccl_engine_t* engine, void* data, size_t size);

/**
 * Send data (blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data to send.
 * @param size          Size of the data.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_send(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                     size_t size);

/**
 * Receive data (blocking).
 * @param conn          Connection handle.
 * @param mr            Memory region handle.
 * @param data          Pointer to the buffer to receive data.
 * @param max_size      Maximum size of the buffer.
 * @param recv_size     Pointer to store the actual received size.
 * @return              0 on success, non-zero on failure.
 */
int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t* mr, void* data,
                     size_t max_size, size_t* recv_size);

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
 * Free endpoint metadata buffer.
 * @param metadata      The metadata buffer to free.
 */
void uccl_engine_free_endpoint_metadata(uint8_t* metadata);

#ifdef __cplusplus
}
#endif  // __cplusplus