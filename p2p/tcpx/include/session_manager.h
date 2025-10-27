/**
 * @file session_manager.h
 * @brief TCPX Session Management API
 *
 * This file defines the TcpxSession class, which manages connections to remote
 * nodes and memory registration for TCPX transfers.
 *
 * Design based on test_tcpx_perf_multi.cc handshake flow (lines 327-437).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tcpx {

// Forward declaration
class TcpxTransfer;

/**
 * @brief TCPX Session - manages connections and memory registration
 *
 * This class encapsulates the complete TCPX handshake flow:
 * - Server: listen() → accept() → register memory → create transfers
 * - Client: loadRemoteConnInfo() → connect() → register memory → create
 * transfers
 *
 * Lifecycle:
 * 1. Construction: Initialize CUDA context, create ChannelManager
 * 2. Handshake: Server listen/accept OR Client loadRemoteConnInfo/connect
 * 3. Memory registration: registerMemory() for send/recv buffers
 * 4. Transfer creation: createTransfer() for each remote peer
 * 5. Destruction: RAII cleanup (deregister memory, release CUDA context, close
 * channels)
 *
 * Thread-safety: NOT thread-safe. Caller must synchronize access.
 */
class TcpxSession {
 public:
  /**
   * @brief Construct a new TCPX session
   * @param gpu_id CUDA device ID (0-based)
   * @param num_channels Number of TCPX channels per connection
   *
   * This constructor:
   * - Initializes CUDA context (cuDevicePrimaryCtxRetain)
   * - Creates ChannelManager with num_channels
   * - Creates UnpackLauncher for GPU kernel unpack
   * - Creates CUDA stream for unpack kernels
   *
   * Based on test_tcpx_perf_multi.cc lines 391-456.
   */
  TcpxSession(int gpu_id, int num_channels);

  /**
   * @brief Destroy the session
   *
   * RAII cleanup:
   * - Deregister all memory
   * - Destroy all transfers
   * - Close all channels
   * - Destroy CUDA stream
   * - Destroy UnpackLauncher
   * - Release CUDA primary context
   */
  ~TcpxSession();

  // Disable copy and move
  TcpxSession(TcpxSession const&) = delete;
  TcpxSession& operator=(TcpxSession const&) = delete;
  TcpxSession(TcpxSession&&) = delete;
  TcpxSession& operator=(TcpxSession&&) = delete;

  // ============================================================================
  // Handshake Flow
  // ============================================================================

  /**
   * @brief Server: Create listen communicators and return serialized handles
   * @return Serialized connection info (JSON format)
   *
   * Server-side step 1: Create listen comms for all channels.
   * Returns a JSON string containing all ncclNetHandle_v7 handles.
   *
   * Based on test_tcpx_perf_multi.cc lines 334-347.
   */
  std::string listen();

  /**
   * @brief Server: Accept connection from a remote peer
   * @param remote_name Identifier for the remote peer (e.g., "client_0")
   * @return 0 on success, non-zero on error
   *
   * Server-side step 2: Accept connections on all channels.
   * Must be called after listen() and after client has called connect().
   *
   * Based on test_tcpx_perf_multi.cc lines 373-380.
   */
  int accept(std::string const& remote_name);

  /**
   * @brief Client: Load remote connection info from server
   * @param remote_name Identifier for the remote peer (e.g., "server_0")
   * @param conn_info Serialized connection info from server's listen()
   * @return 0 on success, non-zero on error
   *
   * Client-side step 1: Parse and store server's handles.
   * Must be called before connect().
   */
  int loadRemoteConnInfo(std::string const& remote_name,
                         std::string const& conn_info);

  /**
   * @brief Client: Connect to remote peer
   * @param remote_name Identifier for the remote peer (e.g., "server_0")
   * @return 0 on success, non-zero on error
   *
   * Client-side step 2: Connect to server using previously loaded handles.
   * Must be called after loadRemoteConnInfo().
   *
   * Based on test_tcpx_perf_multi.cc client path.
   */
  int connect(std::string const& remote_name);

  /**
   * @brief Disconnect from a remote peer
   * @param remote_name Identifier for the remote peer
   * @return 0 on success, non-zero on error
   *
   * Closes all channels to the specified peer.
   * Automatically called by destructor for all connected peers.
   */
  int disconnect(std::string const& remote_name);

  // ============================================================================
  // Memory Management
  // ============================================================================

  /**
   * @brief Memory handle returned by registerMemory()
   */
  struct MemoryHandle {
    void* buffer;   ///< Pointer to the registered buffer
    size_t size;    ///< Size of the buffer in bytes
    int ptr_type;   ///< NCCL_PTR_CUDA or NCCL_PTR_HOST
    bool is_recv;   ///< true for recv buffers, false for send buffers
    void* mhandle;  ///< TCPX memory handle (opaque)
    uint64_t id;    ///< Unique identifier for this registration
  };

  /**
   * @brief Register memory for TCPX transfers
   * @param buffer Pointer to the buffer (must be 4KB-aligned for CUDA buffers)
   * @param size Size of the buffer in bytes
   * @param ptr_type NCCL_PTR_CUDA or NCCL_PTR_HOST
   * @param is_recv true for recv buffers, false for send buffers
   * @return Unique memory ID (>0 on success, 0 on error)
   *
   * Registers the buffer with all channels in the session.
   * The returned mem_id can be used in TcpxTransfer::postSend/postRecv.
   *
   * Based on test_tcpx_perf_multi.cc lines 428-437.
   */
  uint64_t registerMemory(void* buffer, size_t size, int ptr_type,
                          bool is_recv);

  /**
   * @brief Deregister memory
   * @param mem_id Memory ID returned by registerMemory()
   * @return 0 on success, non-zero on error
   *
   * Deregisters the buffer from all channels.
   * Automatically called by destructor for all registered memory.
   */
  int deregisterMemory(uint64_t mem_id);

  /**
   * @brief Get memory handle by ID
   * @param mem_id Memory ID returned by registerMemory()
   * @return Pointer to MemoryHandle, or nullptr if not found
   */
  MemoryHandle* getMemoryHandle(uint64_t mem_id);

  // ============================================================================
  // Transfer Management
  // ============================================================================

  /**
   * @brief Create a new transfer object for a remote peer
   * @param remote_name Identifier for the remote peer
   * @return Pointer to TcpxTransfer object (owned by caller), or nullptr on
   * error
   *
   * The returned transfer object can be used to post send/recv operations.
   * Caller is responsible for deleting the transfer object.
   */
  TcpxTransfer* createTransfer(std::string const& remote_name);

  // ============================================================================
  // Accessors (for TcpxTransfer)
  // ============================================================================

  /**
   * @brief Get the number of channels
   * @return Number of channels in this session
   */
  int getNumChannels() const;

  /**
   * @brief Get the GPU ID
   * @return CUDA device ID
   */
  int getGpuId() const;

  // Internal accessors (for TcpxTransfer implementation)
  void* getChannelManager();  // Returns ChannelManager*
  void* getUnpackLauncher();  // Returns tcpx::device::UnpackLauncher*
  void* getUnpackStream();    // Returns cudaStream_t

 private:
  struct Impl;  // PIMPL pattern to hide implementation details
  std::unique_ptr<Impl> impl_;

  // Friend class to allow TcpxTransfer to access impl_
  friend class TcpxTransfer;
};

}  // namespace tcpx
