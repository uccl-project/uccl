/**
 * @file session_manager.cc
 * @brief TCPX Session Management Implementation
 *
 * This file implements the TcpxSession class based on the handshake flow
 * from test_tcpx_perf_multi.cc (lines 327-437 for server, client path similar).
 */

#include "session_manager.h"

#include "channel_manager.h"
#include "tcpx_logging.h"
#include "transfer_manager.h"
#include "device/unpack_launch.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace tcpx {

// ============================================================================
// TcpxSession::Impl - Private implementation
// ============================================================================

struct TcpxSession::Impl {
  // GPU and channel configuration
  int gpu_id_;
  int num_channels_;

  // CUDA resources
  CUdevice cu_dev_;
  CUcontext cu_ctx_;
  cudaStream_t unpack_stream_;

  // TCPX resources
  ChannelManager* mgr_;
  tcpx::device::UnpackLauncher* launcher_;

  // Connection state
  std::map<std::string, std::vector<ncclNetHandle_v7>> remote_handles_;
  std::map<std::string, bool> remote_accepted_;

  // Memory registration
  std::map<uint64_t, TcpxSession::MemoryHandle> registered_memory_;
  uint64_t next_mem_id_;

  // Constructor
  Impl(int gpu_id, int num_channels)
      : gpu_id_(gpu_id),
        num_channels_(num_channels),
        cu_dev_(0),
        cu_ctx_(nullptr),
        unpack_stream_(nullptr),
        mgr_(nullptr),
        launcher_(nullptr),
        next_mem_id_(1) {  // Start from 1 (0 reserved for error)
  }

  // Destructor
  ~Impl() {
    // Cleanup in reverse order of initialization
    cleanup();
  }

  void cleanup() {
    // Deregister all memory (per mem_id)
    if (mgr_) {
      for (auto& pair : registered_memory_) {
        mgr_->deregister_memory(pair.first, pair.second.is_recv);
      }
    }
    registered_memory_.clear();

    // Destroy UnpackLauncher
    if (launcher_) {
      delete launcher_;
      launcher_ = nullptr;
    }

    // Destroy CUDA stream
    if (unpack_stream_) {
      cudaStreamDestroy(unpack_stream_);
      unpack_stream_ = nullptr;
    }

    // Close all channels before destroying ChannelManager
    if (mgr_) {
      mgr_->close_all(true);   // Close recv/listen comms
      mgr_->close_all(false);  // Close send comms
      delete mgr_;
      mgr_ = nullptr;
    }

    // Release CUDA primary context
    if (cu_ctx_) {
      cuDevicePrimaryCtxRelease(cu_dev_);
      cu_ctx_ = nullptr;
    }
  }
};

// ============================================================================
// TcpxSession - Public API
// ============================================================================

TcpxSession::TcpxSession(int gpu_id, int num_channels)
    : impl_(new Impl(gpu_id, num_channels)) {
  // Based on test_tcpx_perf_multi.cc lines 391-456

  // Step 1: Initialize CUDA context
  CUresult cu_rc = cuInit(0);
  if (cu_rc != CUDA_SUCCESS) {
    LOG_ERROR("cuInit failed: %d", cu_rc);
    throw std::runtime_error("cuInit failed");
  }

  cu_rc = cuDeviceGet(&impl_->cu_dev_, gpu_id);
  if (cu_rc != CUDA_SUCCESS) {
    LOG_ERROR("cuDeviceGet failed: %d", cu_rc);
    throw std::runtime_error("cuDeviceGet failed");
  }

  cu_rc = cuDevicePrimaryCtxRetain(&impl_->cu_ctx_, impl_->cu_dev_);
  if (cu_rc != CUDA_SUCCESS) {
    LOG_ERROR("cuDevicePrimaryCtxRetain failed: %d", cu_rc);
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed");
  }

  cu_rc = cuCtxSetCurrent(impl_->cu_ctx_);
  if (cu_rc != CUDA_SUCCESS) {
    LOG_ERROR("cuCtxSetCurrent failed: %d", cu_rc);
    cuDevicePrimaryCtxRelease(impl_->cu_dev_);
    throw std::runtime_error("cuCtxSetCurrent failed");
  }

  cudaError_t cuda_rc = cudaSetDevice(gpu_id);
  if (cuda_rc != cudaSuccess) {
    LOG_ERROR("cudaSetDevice failed: %s", cudaGetErrorString(cuda_rc));
    cuDevicePrimaryCtxRelease(impl_->cu_dev_);
    throw std::runtime_error("cudaSetDevice failed");
  }

  // Step 2: Create ChannelManager
  impl_->mgr_ = new ChannelManager(num_channels, gpu_id);
  impl_->num_channels_ = impl_->mgr_->get_num_channels();  // May be clamped

  LOG_INFO("TcpxSession created: GPU %d, %d channels", gpu_id,
           impl_->num_channels_);

  // Step 3: Create CUDA stream for unpack kernels
  cuda_rc = cudaStreamCreate(&impl_->unpack_stream_);
  if (cuda_rc != cudaSuccess) {
    LOG_ERROR("cudaStreamCreate failed: %s", cudaGetErrorString(cuda_rc));
    delete impl_->mgr_;
    impl_->mgr_ = nullptr;
    cuDevicePrimaryCtxRelease(impl_->cu_dev_);
    throw std::runtime_error("cudaStreamCreate failed");
  }

  // Step 4: Create UnpackLauncher
  tcpx::device::UnpackLaunchConfig launcher_config;
  launcher_config.stream = impl_->unpack_stream_;
  launcher_config.use_small_kernel = true;
  launcher_config.enable_profiling = false;
  impl_->launcher_ = new tcpx::device::UnpackLauncher(launcher_config);

  LOG_INFO("TcpxSession initialized successfully");
}

TcpxSession::~TcpxSession() {
  LOG_INFO("TcpxSession destroyed");
}

// ============================================================================
// Handshake Flow
// ============================================================================

std::string TcpxSession::listen() {
  // Based on test_tcpx_perf_multi.cc lines 334-347

  std::vector<ncclNetHandle_v7> handles;
  int rc = impl_->mgr_->server_listen_all(handles);
  if (rc != 0) {
    LOG_ERROR("server_listen_all failed: %d", rc);
    return "";
  }

  LOG_INFO("Listening on %d channels", impl_->num_channels_);

  // Serialize handles to JSON
  std::ostringstream oss;
  oss << "{\"num_channels\":" << handles.size() << ",\"handles\":[";
  for (size_t i = 0; i < handles.size(); ++i) {
    if (i > 0) oss << ",";
    oss << "\"";
    // Encode handle as hex string
    const unsigned char* data =
        reinterpret_cast<const unsigned char*>(&handles[i]);
    for (size_t j = 0; j < sizeof(ncclNetHandle_v7); ++j) {
      char hex[3];
      snprintf(hex, sizeof(hex), "%02x", data[j]);
      oss << hex;
    }
    oss << "\"";
  }
  oss << "]}";

  return oss.str();
}

int TcpxSession::accept(const std::string& remote_name) {
  // Based on test_tcpx_perf_multi.cc lines 373-380

  int rc = impl_->mgr_->server_accept_all();
  if (rc != 0) {
    LOG_ERROR("server_accept_all failed: %d", rc);
    return rc;
  }

  impl_->remote_accepted_[remote_name] = true;

  // Add remote to handles map so createTransfer() can find it
  // Server doesn't need actual handles (already has accept comms)
  impl_->remote_handles_[remote_name] = std::vector<ncclNetHandle_v7>();

  LOG_INFO("Accepted connection from %s", remote_name.c_str());

  return 0;
}

int TcpxSession::loadRemoteConnInfo(const std::string& remote_name,
                                    const std::string& conn_info) {
  // Parse JSON to extract handles
  // Simple parser (assumes valid JSON from listen())

  // Find num_channels
  size_t num_pos = conn_info.find("\"num_channels\":");
  if (num_pos == std::string::npos) {
    LOG_ERROR("Invalid conn_info: missing num_channels");
    return -1;
  }

  int num_channels = 0;
  if (sscanf(conn_info.c_str() + num_pos + 15, "%d", &num_channels) != 1) {
    LOG_ERROR("Invalid conn_info: failed to parse num_channels");
    return -1;
  }

  // Find handles array
  size_t handles_pos = conn_info.find("\"handles\":[");
  if (handles_pos == std::string::npos) {
    LOG_ERROR("Invalid conn_info: missing handles");
    return -1;
  }

  std::vector<ncclNetHandle_v7> handles;
  handles.resize(num_channels);

  // Parse each handle (hex string)
  const char* p = conn_info.c_str() + handles_pos + 11;  // Skip "handles":["
  for (int i = 0; i < num_channels; ++i) {
    // Skip to next quote
    while (*p && *p != '"') p++;
    if (!*p) {
      LOG_ERROR("Invalid conn_info: unexpected end of string");
      return -1;
    }
    p++;  // Skip opening quote

    // Decode hex string
    unsigned char* data = reinterpret_cast<unsigned char*>(&handles[i]);
    for (size_t j = 0; j < sizeof(ncclNetHandle_v7); ++j) {
      unsigned int byte;
      if (sscanf(p, "%2x", &byte) != 1) {
        LOG_ERROR("Invalid conn_info: failed to parse handle %d byte %zu", i,
                  j);
        return -1;
      }
      data[j] = static_cast<unsigned char>(byte);
      p += 2;
    }

    // Skip closing quote and comma
    while (*p && *p != '"') p++;
    if (*p) p++;  // Skip closing quote
    while (*p && (*p == ',' || *p == ' ')) p++;
  }

  impl_->remote_handles_[remote_name] = handles;
  LOG_INFO("Loaded connection info for %s: %d channels", remote_name.c_str(),
           num_channels);

  return 0;
}

int TcpxSession::connect(const std::string& remote_name) {
  // Find handles for this remote
  auto it = impl_->remote_handles_.find(remote_name);
  if (it == impl_->remote_handles_.end()) {
    LOG_ERROR("No connection info loaded for %s", remote_name.c_str());
    return -1;
  }

  int rc = impl_->mgr_->client_connect_all(it->second);
  if (rc != 0) {
    LOG_ERROR("client_connect_all failed: %d", rc);
    return rc;
  }

  LOG_INFO("Connected to %s", remote_name.c_str());
  return 0;
}

int TcpxSession::disconnect(const std::string& remote_name) {
  // Close all channels (both recv and send)
  if (impl_->mgr_) {
    impl_->mgr_->close_all(true);   // Close recv/listen comms
    impl_->mgr_->close_all(false);  // Close send comms
  }

  // Remove from tracking maps
  impl_->remote_handles_.erase(remote_name);
  impl_->remote_accepted_.erase(remote_name);

  LOG_INFO("Disconnected from %s", remote_name.c_str());
  return 0;
}

// ============================================================================
// Memory Management
// ============================================================================

uint64_t TcpxSession::registerMemory(void* buffer, size_t size, int ptr_type,
                                     bool is_recv) {
  // Based on test_tcpx_perf_multi.cc lines 428-437

  if (!buffer || size == 0) {
    LOG_ERROR("Invalid buffer or size");
    return 0;
  }

  // Allocate mem_id first
  uint64_t mem_id = impl_->next_mem_id_++;

  // Register with ChannelManager (pass mem_id)
  int rc = impl_->mgr_->register_memory(mem_id, buffer, size, ptr_type, is_recv);
  if (rc != 0) {
    LOG_ERROR("register_memory failed: %d", rc);
    return 0;
  }

  // Create memory handle
  MemoryHandle handle;
  handle.buffer = buffer;
  handle.size = size;
  handle.ptr_type = ptr_type;
  handle.is_recv = is_recv;
  handle.mhandle = nullptr;  // Not exposed at session level
  handle.id = mem_id;

  impl_->registered_memory_[mem_id] = handle;

  LOG_INFO("Registered memory: id=%lu, buffer=%p, size=%zu, is_recv=%d",
           mem_id, buffer, size, is_recv);

  return mem_id;
}

int TcpxSession::deregisterMemory(uint64_t mem_id) {
  auto it = impl_->registered_memory_.find(mem_id);
  if (it == impl_->registered_memory_.end()) {
    LOG_ERROR("Memory ID %lu not found", mem_id);
    return -1;
  }

  // Deregister from ChannelManager (pass mem_id)
  int rc = impl_->mgr_->deregister_memory(mem_id, it->second.is_recv);
  if (rc != 0) {
    LOG_ERROR("deregister_memory failed: %d", rc);
    return rc;
  }

  impl_->registered_memory_.erase(it);
  LOG_INFO("Deregistered memory: id=%lu", mem_id);

  return 0;
}

TcpxSession::MemoryHandle* TcpxSession::getMemoryHandle(uint64_t mem_id) {
  auto it = impl_->registered_memory_.find(mem_id);
  if (it == impl_->registered_memory_.end()) {
    return nullptr;
  }
  return &it->second;
}

// ============================================================================
// Transfer Management
// ============================================================================

TcpxTransfer* TcpxSession::createTransfer(const std::string& remote_name) {
  // Check if connected to this remote
  auto it = impl_->remote_handles_.find(remote_name);
  if (it == impl_->remote_handles_.end()) {
    LOG_ERROR("Not connected to %s", remote_name.c_str());
    return nullptr;
  }

  // Create transfer object
  TcpxTransfer* transfer = new TcpxTransfer(this, remote_name);
  LOG_INFO("Created transfer for remote: %s", remote_name.c_str());
  return transfer;
}

// ============================================================================
// Accessors
// ============================================================================

int TcpxSession::getNumChannels() const { return impl_->num_channels_; }

int TcpxSession::getGpuId() const { return impl_->gpu_id_; }

void* TcpxSession::getChannelManager() { return impl_->mgr_; }

void* TcpxSession::getUnpackLauncher() { return impl_->launcher_; }

void* TcpxSession::getUnpackStream() { return impl_->unpack_stream_; }

}  // namespace tcpx
