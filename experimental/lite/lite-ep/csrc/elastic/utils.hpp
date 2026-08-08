#pragma once

#include <deep_ep/common/exception.cuh>
#include <torch/python.h>

namespace deep_ep::elastic {

static at::cuda::CUDAStream get_global_comm_stream() {
  static std::optional<at::cuda::CUDAStream> comm_stream = std::nullopt;
  if (not comm_stream.has_value())
    comm_stream = at::cuda::getStreamFromPool(true);
  return comm_stream.value();
}

template <int kNumDims>
static auto get_shape(torch::Tensor const& t) {
  EP_HOST_ASSERT(t.dim() == kNumDims);
  return [&t]<size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
  }
  (std::make_index_sequence<kNumDims>());
}

template <typename dtype_t = void>
static dtype_t* get_data_ptr(std::optional<torch::Tensor> const& t) {
  return t.has_value() ? t->data_ptr<dtype_t>() : nullptr;
}

}  // namespace deep_ep::elastic
