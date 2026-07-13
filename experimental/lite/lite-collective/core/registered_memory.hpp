// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include "communicator.hpp"
#include "core.hpp"
#include "errors.hpp"
#include "gpu.hpp"
#include "gpu_ipc_mem.hpp"
#include "ib.hpp"

namespace mscclpp {

struct TransportInfo {
  Transport transport;

  // TODO: rewrite this using std::variant or something
  bool ibLocal;
  union {
    GpuIpcMemHandle gpuIpcMemHandle;
    struct {
      IbMr const* ibMr;
      IbMrInfo ibMrInfo;
    };
  };
};

struct RegisteredMemory::Impl {
  // This is the data pointer returned by RegisteredMemory::data(), which may be
  // different from the original data pointer for deserialized remote memory.
  void* data;
  // This is the original data pointer the RegisteredMemory was created with.
  void* originalDataPtr;
  size_t size;
  uint64_t hostHash;
  uint64_t pidHash;
  TransportFlags transports;
  std::vector<TransportInfo> transportInfos;
  std::shared_ptr<void> peerMemHandle;

  UniqueGpuIpcMemHandle localGpuIpcMemHandle;
  std::shared_ptr<void> remoteMemMap;

  // Only used for IB transport
  std::unordered_map<Transport, std::unique_ptr<const IbMr>> ibMrMap;

  Impl(void* data, size_t size, TransportFlags transports,
       Context::Impl& contextImpl);
  Impl(const std::vector<char>::const_iterator& begin,
       const std::vector<char>::const_iterator& end);
  /// Constructs a RegisteredMemory::Impl from a vector of data. The constructor
  /// should only be used for the remote memory.
  Impl(std::vector<char> const& data);

  TransportInfo const& getTransportInfo(Transport transport) const;
};

}  // namespace mscclpp

#endif  // MSCCLPP_REGISTERED_MEMORY_HPP_
