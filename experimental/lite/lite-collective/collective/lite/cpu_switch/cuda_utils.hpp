// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace mscclpp::lite {

/* A distinct runtime_error subtype lets NCCL adapters preserve CUDA errors. */
class CudaOperationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Throws CudaOperationError with CUDA's diagnostic string on failure.
void throwCudaError(cudaError_t result, char const* operation);

}  // namespace mscclpp::lite
