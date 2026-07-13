// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONTEXT_HPP_
#define MSCCLPP_CONTEXT_HPP_

#include "core.hpp"
#include "gpu_utils.hpp"
#include "ib.hpp"
#include <unordered_map>
#include <vector>

namespace mscclpp {

class CudaIpcStream {
 private:
  std::shared_ptr<CudaStreamWithFlags> stream_;
  int deviceId_;
  bool dirty_;

  void setStreamIfNeeded();

 public:
  CudaIpcStream(int deviceId);

  void memcpyD2D(void* dst, void const* src, size_t nbytes);

  void memcpyH2D(void* dst, void const* src, size_t nbytes);

  void sync();

  operator cudaStream_t() const { return *stream_; }

  int deviceId() const { return deviceId_; }
};

class TokenPool;
struct Context::Impl {
  std::unordered_map<Transport, std::unique_ptr<IbCtx>> ibContexts_;
  std::vector<std::shared_ptr<CudaIpcStream>> ipcStreams_;
  std::shared_ptr<TokenPool> tokenPool_;
  const size_t maxNumTokens_ = 1 << 15;  // 32K tokens

  Impl();

  IbCtx* getIbContext(Transport ibTransport);
  std::shared_ptr<uint64_t> getToken();
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONTEXT_HPP_
