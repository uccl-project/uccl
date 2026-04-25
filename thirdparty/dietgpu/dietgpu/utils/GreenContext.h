/**
 * Green Context support for limiting SM usage in dietgpu.
 *
 * When the environment variable UCCL_DIETGPU_SM_USAGE is set to a positive
 * integer, dietgpu will create a CUDA green context that restricts kernel
 * execution to at most that many SMs.
 *
 * Requires CUDA >= 12.4 (driver API green context support).
 * On older CUDA versions or when the env var is unset, this is a no-op.
 */

#pragma once

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace dietgpu {

// Green context APIs require CUDA 12.4+
#if CUDA_VERSION >= 12040

class GreenContextManager {
 public:
  static GreenContextManager& instance() {
    static GreenContextManager inst;
    return inst;
  }

  /// Create the green context once (thread-safe). Does NOT push/activate it.
  void create(int device) {
    std::lock_guard<std::mutex> lock(mu_);
    if (created_) return;
    created_ = true;

    const char* envVal = std::getenv("UCCL_DIETGPU_SM_USAGE");
    if (!envVal) return;

    int requestedSMs = std::atoi(envVal);
    if (requestedSMs <= 0) return;

    CUresult err;

    err = cuInit(0);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "[GreenContext] cuInit failed (%d), skipping\n", (int)err);
      return;
    }

    CUdevice cuDev;
    err = cuDeviceGet(&cuDev, device);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "[GreenContext] cuDeviceGet(%d) failed (%d), skipping\n",
              device, (int)err);
      return;
    }

    int totalSMs = 0;
    err = cuDeviceGetAttribute(
        &totalSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDev);
    if (err != CUDA_SUCCESS || totalSMs <= 0) {
      fprintf(stderr, "[GreenContext] Cannot query SM count, skipping\n");
      return;
    }

    unsigned int smCount = static_cast<unsigned int>(requestedSMs);
    if (smCount >= static_cast<unsigned int>(totalSMs)) {
      fprintf(stderr,
              "[GreenContext] Requested %u SMs >= total %d SMs, skipping\n",
              smCount, totalSMs);
      return;
    }

    // SM 9.0+ requires multiples of 8; align down.
    smCount = (smCount / 8) * 8;
    if (smCount == 0) smCount = 8;

    CUdevResource devResource;
    err = cuDeviceGetDevResource(cuDev, &devResource, CU_DEV_RESOURCE_TYPE_SM);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr,
              "[GreenContext] cuDeviceGetDevResource failed (%d), skipping\n",
              (int)err);
      return;
    }

    unsigned int nbGroups = 1;
    CUdevResource splitResult;
    CUdevResource remaining;
    err = cuDevSmResourceSplitByCount(
        &splitResult, &nbGroups, &devResource, &remaining, 0, smCount);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr,
              "[GreenContext] cuDevSmResourceSplitByCount failed (%d), "
              "skipping\n",
              (int)err);
      return;
    }

    CUdevResourceDesc resDesc;
    err = cuDevResourceGenerateDesc(&resDesc, &splitResult, 1);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr,
              "[GreenContext] cuDevResourceGenerateDesc failed (%d), "
              "skipping\n",
              (int)err);
      return;
    }

    err = cuGreenCtxCreate(
        &greenCtx_, resDesc, cuDev, CU_GREEN_CTX_DEFAULT_STREAM);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr,
              "[GreenContext] cuGreenCtxCreate failed (%d), skipping\n",
              (int)err);
      return;
    }

    err = cuCtxFromGreenCtx(&ctx_, greenCtx_);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr,
              "[GreenContext] cuCtxFromGreenCtx failed (%d), skipping\n",
              (int)err);
      cuGreenCtxDestroy(greenCtx_);
      greenCtx_ = nullptr;
      return;
    }

    active_ = true;
    fprintf(stderr,
            "[GreenContext] Created: %u / %d SMs on device %d "
            "(requested %d, aligned to %u)\n",
            splitResult.sm.smCount, totalSMs, device, requestedSMs, smCount);
  }

  /// Ensure the green context is current on the CALLING thread.
  /// Must be called before every CUDA operation.
  /// cuCtxSetCurrent replaces the current context (no stack buildup),
  /// and is idempotent if already set.
  void ensureCurrent() {
    if (!active_) return;

    CUcontext cur = nullptr;
    cuCtxGetCurrent(&cur);
    if (cur == ctx_) return;  // already active on this thread

    CUresult err = cuCtxSetCurrent(ctx_);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr,
              "[GreenContext] cuCtxSetCurrent failed (%d) on thread, "
              "SM limiting may not apply\n",
              (int)err);
    }
  }

  bool isActive() const { return active_; }

  ~GreenContextManager() {
    if (active_) {
      cuGreenCtxDestroy(greenCtx_);
    }
  }

 private:
  GreenContextManager() = default;
  GreenContextManager(const GreenContextManager&) = delete;
  GreenContextManager& operator=(const GreenContextManager&) = delete;

  std::mutex mu_;
  bool created_ = false;
  bool active_ = false;
  CUgreenCtx greenCtx_ = nullptr;
  CUcontext ctx_ = nullptr;
};

/// Call before every CUDA operation to ensure SM limiting is active.
/// First call creates the green context; every call ensures it is current
/// on the calling thread (handles multi-threading and cudaSetDevice resets).
inline void initGreenContextIfNeeded(int device) {
  auto& mgr = GreenContextManager::instance();
  mgr.create(device);
  mgr.ensureCurrent();
}

#else // CUDA_VERSION < 12040

inline void initGreenContextIfNeeded(int /*device*/) {
  // Green context not available on CUDA < 12.4
}

#endif // CUDA_VERSION >= 12040

} // namespace dietgpu
