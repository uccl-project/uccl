#ifndef MSCCLPP_NCCL_LITE_COMMON_H_
#define MSCCLPP_NCCL_LITE_COMMON_H_

#include "env.hpp"
#include "gpu_utils.hpp"
#include "ib.hpp"
#include "native_collectives.hpp"
#include "numa.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <numa.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

inline void* operator new(std::size_t, std::align_val_t, void* ptr) noexcept {
  return ptr;
}

namespace mscclpp {
namespace lite {

struct InitStatus {
  int result = static_cast<int>(ncclSuccess);
  char message[160] = {};
};

template <typename Context>
class InitGuard {
 public:
  explicit InitGuard(Context* ctx) : ctx_(ctx) {}
  ~InitGuard() {
    if (committed_) return;
    auto failure = std::current_exception();
    if (!failure) {
      failure = std::make_exception_ptr(mscclpp::Error(
          "native collective context initialization did not complete",
          mscclpp::ErrorCode::InternalError));
    }
    {
      std::lock_guard<std::mutex> lock(ctx_->initMutex);
      ctx_->initialized = false;
      ctx_->initializing = false;
      ctx_->initException = failure;
    }
    ctx_->initCv.notify_all();
  }

  void commit() {
    {
      std::lock_guard<std::mutex> lock(ctx_->initMutex);
      ctx_->initialized = true;
      ctx_->initializing = false;
      ctx_->initException = nullptr;
    }
    committed_ = true;
    ctx_->initCv.notify_all();
  }

 private:
  Context* ctx_;
  bool committed_ = false;
};

inline ncclResult_t cudaResult(cudaError_t error, char const* operation) {
  if (error == cudaSuccess) return ncclSuccess;
  std::fprintf(stderr, "WARN: %s failed with CUDA error: %s\n", operation,
               cudaGetErrorString(error));
  if (error == cudaErrorInvalidValue) return ncclInvalidArgument;
  return ncclUnhandledCudaError;
}

inline ncclResult_t mapException(std::exception const& ex) {
  if (auto const* err = dynamic_cast<mscclpp::Error const*>(&ex)) {
    switch (err->getErrorCode()) {
      case mscclpp::ErrorCode::InvalidUsage:
        return ncclInvalidUsage;
      case mscclpp::ErrorCode::Timeout:
      case mscclpp::ErrorCode::SystemError:
        return ncclSystemError;
      default:
        return ncclInternalError;
    }
  }
  if (dynamic_cast<mscclpp::CudaError const*>(&ex) != nullptr ||
      dynamic_cast<mscclpp::CuError const*>(&ex) != nullptr) {
    return ncclUnhandledCudaError;
  }
  return ncclInternalError;
}

inline mscclpp::ErrorCode initErrorCode(ncclResult_t result) {
  switch (result) {
    case ncclInvalidArgument:
    case ncclInvalidUsage:
      return mscclpp::ErrorCode::InvalidUsage;
    case ncclSystemError:
      return mscclpp::ErrorCode::SystemError;
    default:
      return mscclpp::ErrorCode::InternalError;
  }
}

inline void publishInitStatus(std::shared_ptr<Communicator> bootstrapComm,
                              int rank, int nRanks, ncclResult_t result,
                              std::string const& message, char const* stage) {
  std::vector<InitStatus> statuses(nRanks);
  auto& local = statuses[rank];
  local.result = static_cast<int>(result);
  if (!message.empty()) {
    std::snprintf(local.message, sizeof(local.message), "%s", message.c_str());
  }
  bootstrapComm->bootstrap()->allGather(statuses.data(), sizeof(InitStatus));

  for (int peer = 0; peer < nRanks; ++peer) {
    auto peerResult = static_cast<ncclResult_t>(statuses[peer].result);
    if (peerResult != ncclSuccess) {
      std::string detail(statuses[peer].message);
      if (detail.empty()) detail = "unknown initialization error";
      throw mscclpp::Error(std::string(stage) + " failed on rank " +
                               std::to_string(peer) + ": " + detail,
                           initErrorCode(peerResult));
    }
  }
}

inline int getIBDeviceNumaNode(std::string const& ibDevName) {
  std::string path = "/sys/class/infiniband/" + ibDevName + "/device/numa_node";
  std::ifstream f(path);
  int node = -1;
  if (f.is_open()) f >> node;
  return node;
}

inline std::vector<mscclpp::Transport> getAvailableIBTransports() {
  static const mscclpp::Transport transports[] = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1,
      mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5,
      mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  int count = 0;
  std::string hcaEnv = mscclpp::env()->hcaDevices;
  if (!hcaEnv.empty()) {
    std::stringstream ss(hcaEnv);
    std::string tok;
    while (std::getline(ss, tok, ',')) ++count;
  } else {
    count = mscclpp::getIBDeviceCount();
  }
  count = std::min(count,
                   static_cast<int>(sizeof(transports) / sizeof(transports[0])));
  std::vector<mscclpp::Transport> result;
  result.reserve(count);
  for (int i = 0; i < count; ++i) result.push_back(transports[i]);
  return result;
}

inline mscclpp::Transport selectIBTransportForGpu(int cudaDeviceId) {
  auto available = getAvailableIBTransports();
  if (available.empty()) return mscclpp::Transport::Unknown;

  int gpuNuma = -1;
  try {
    gpuNuma = mscclpp::getDeviceNumaNode(cudaDeviceId);
  } catch (...) {
  }

  std::vector<mscclpp::Transport> sameNuma;
  for (auto transport : available) {
    try {
      std::string name = mscclpp::getIBDeviceName(transport);
      if (gpuNuma >= 0 && getIBDeviceNumaNode(name) == gpuNuma) {
        sameNuma.push_back(transport);
      }
    } catch (...) {
    }
  }
  auto const& choices = sameNuma.empty() ? available : sameNuma;
  return choices[static_cast<size_t>(cudaDeviceId) % choices.size()];
}

inline void placeOnNuma(void* mapping, size_t size, int numaNode,
                        char const* name) {
  if (mapping == nullptr || size == 0 || numaNode < 0) return;
  if (numa_available() < 0) {
    std::fprintf(stderr, "WARN: NUMA placement unavailable for %s\n", name);
    return;
  }
  std::memset(mapping, 0, size);
  numa_tonode_memory(mapping, size, numaNode);
}

inline void createOwnedShm(std::string const& name, size_t size) {
  shm_unlink(name.c_str());
  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) {
    throw mscclpp::Error("shm_open failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  if (ftruncate(fd, size) < 0) {
    close(fd);
    shm_unlink(name.c_str());
    throw mscclpp::Error("ftruncate failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  close(fd);
}

inline void* mapShm(std::string const& name, size_t size) {
  int fd = shm_open(name.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    throw mscclpp::Error("shm_open failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  void* mapping =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (mapping == MAP_FAILED) {
    throw mscclpp::Error("mmap failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  return mapping;
}

inline void waitForEpoch(std::atomic<uint64_t> const& value, uint64_t epoch) {
  int spins = 0;
  while (value.load(std::memory_order_acquire) < epoch) {
    if (spins++ < 65536) {
      asm volatile("pause" ::: "memory");
    } else {
      std::this_thread::yield();
    }
  }
}

inline void waitForCudaEvent(cudaEvent_t event) {
  while (true) {
    cudaError_t result = cudaEventQuery(event);
    if (result == cudaSuccess) return;
    if (result != cudaErrorNotReady) MSCCLPP_CUDATHROW(result);
    std::this_thread::yield();
  }
}

inline bool needsFallback(ncclResult_t result) {
  return result == ncclInvalidUsage || result == ncclInvalidArgument;
}

inline bool forceFallback(bool ncclLoaded, char const* opName,
                          std::string const& fallbackList) {
  if (!ncclLoaded) return false;
  if (fallbackList == "all") return true;
  std::stringstream ss(fallbackList);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token == opName) return true;
  }
  return false;
}

}  // namespace lite
}  // namespace mscclpp

#endif  // MSCCLPP_NCCL_LITE_COMMON_H_
