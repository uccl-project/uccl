// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cuda_runtime.h>

namespace mscclpp::lite {

// Throws std::runtime_error with CUDA's diagnostic string on failure.
void throwCudaError(cudaError_t result, char const* operation);

}  // namespace mscclpp::lite
