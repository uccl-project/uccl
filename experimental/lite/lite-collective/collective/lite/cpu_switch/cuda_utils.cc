// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "cuda_utils.hpp"

#include <sstream>
#include <stdexcept>

namespace mscclpp::lite {

void throwCudaError(cudaError_t result, char const* operation) {
  if (result == cudaSuccess) return;
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(result);
  throw std::runtime_error(message.str());
}

}  // namespace mscclpp::lite
