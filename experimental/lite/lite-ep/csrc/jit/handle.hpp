#pragma once

#include "../utils/lazy_driver.hpp"
#include <deep_ep/common/exception.cuh>
#include <filesystem>
#include <cuda.h>

namespace deep_ep::jit {

#if CUDART_VERSION >= 12080 and defined(EP_JIT_USE_RUNTIME_API)

// Use CUDA runtime API
using LibraryHandle = cudaLibrary_t;
using KernelHandle = cudaKernel_t;
using LaunchConfigHandle = cudaLaunchConfig_t;
using LaunchAttrHandle = cudaLaunchAttribute;

#define EP_CUDA_UNIFIED_CHECK CUDA_RUNTIME_CHECK

static KernelHandle load_kernel(std::filesystem::path const& cubin_path,
                                std::string const& func_name,
                                LibraryHandle* library_opt = nullptr) {
  LibraryHandle library;
  KernelHandle kernel{};
  CUDA_RUNTIME_CHECK(cudaLibraryLoadFromFile(
      &library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  CUDA_RUNTIME_CHECK(cudaLibraryGetKernel(&kernel, library, func_name.c_str()));

  if (library_opt != nullptr) *library_opt = library;
  return kernel;
}

static void unload_library(LibraryHandle const& library) {
  auto const error = cudaLibraryUnload(library);
  EP_HOST_ASSERT(error == cudaSuccess or error == cudaErrorCudartUnloading);
}

static LaunchConfigHandle construct_launch_config(
    KernelHandle const& kernel, cudaStream_t const& stream,
    int const& smem_size, dim3 const& grid_dim, dim3 const& block_dim,
    int const& cluster_dim, bool const& cooperative, bool const& enable_pdl) {
  if (smem_size > 0)
    CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  LaunchConfigHandle config;
  config.gridDim = grid_dim;
  config.blockDim = block_dim;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  config.numAttrs = 0;
  config.attrs = nullptr;

  // TODO: support cooperative and dependent kernel launch
  (void)cooperative;
  (void)enable_pdl;
  // NOTES: must use `static` or the `attr` will be deconstructed
  static LaunchAttrHandle attr;
  if (cluster_dim > 1) {
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {static_cast<unsigned>(cluster_dim), 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;
  }
  return config;
}

template <typename... ActTypes>
static auto launch_kernel(KernelHandle const& kernel,
                          LaunchConfigHandle const& config,
                          ActTypes&&... args) {
  void* ptr_args[] = {&args...};
  return cudaLaunchKernelExC(&config, kernel, ptr_args);
}

#else

// Use CUDA driver API
using LibraryHandle = CUmodule;
using KernelHandle = CUfunction;
using LaunchConfigHandle = CUlaunchConfig;
using LaunchAttrHandle = CUlaunchAttribute;

#define EP_CUDA_UNIFIED_CHECK CUDA_DRIVER_CHECK

static KernelHandle load_kernel(std::filesystem::path const& cubin_path,
                                std::string const& func_name,
                                LibraryHandle* library_opt = nullptr) {
  LibraryHandle library;
  KernelHandle kernel;
  CUDA_DRIVER_CHECK(lazy_cuModuleLoad(&library, cubin_path.c_str()));
  CUDA_DRIVER_CHECK(
      lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()));

  if (library_opt != nullptr) *library_opt = library;
  return kernel;
}

static void unload_library(LibraryHandle const& library) {
  auto const error = lazy_cuModuleUnload(library);
  EP_HOST_ASSERT(error == CUDA_SUCCESS or error == CUDA_ERROR_DEINITIALIZED);
}

static LaunchConfigHandle construct_launch_config(
    KernelHandle const& kernel, cudaStream_t const& stream,
    int const& smem_size, dim3 const& grid_dim, dim3 const& block_dim,
    int const& cluster_dim, bool const& cooperative, bool const& enable_pdl) {
  if (smem_size > 0)
    CUDA_DRIVER_CHECK(lazy_cuFuncSetAttribute(
        kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));

  LaunchConfigHandle config;
  config.gridDimX = grid_dim.x;
  config.gridDimY = grid_dim.y;
  config.gridDimZ = grid_dim.z;
  config.blockDimX = block_dim.x;
  config.blockDimY = block_dim.y;
  config.blockDimZ = block_dim.z;
  config.sharedMemBytes = smem_size;
  config.hStream = stream;

  // Create attributes
  static LaunchAttrHandle attrs[3];
  config.attrs = attrs;
  config.numAttrs = 0;

  // Cooperative launch
  if (cooperative) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attr.value.cooperative = 1;
  }

  // Cluster size
  // NOTES: must use `static` or the `attr` will be deconstructed
  if (cluster_dim > 1) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr.value.clusterDim.x = cluster_dim;
    attr.value.clusterDim.y = 1;
    attr.value.clusterDim.z = 1;
  }

  // Dependent kernel launch
  if (enable_pdl) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attr.value.programmaticStreamSerializationAllowed = 1;
  }
  return config;
}

template <typename... ActTypes>
static auto launch_kernel(KernelHandle const& kernel,
                          LaunchConfigHandle const& config,
                          ActTypes&&... args) {
  void* ptr_args[] = {&args...};
  return lazy_cuLaunchKernelEx(&config, kernel, ptr_args, nullptr);
}

#endif

}  // namespace deep_ep::jit
