// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENDPOINT_HPP_
#define MSCCLPP_ENDPOINT_HPP_

#include "core.hpp"
#include "ib.hpp"
#include "socket.h"
#include <memory>
#include <vector>

#define MAX_IF_NAME_SIZE 16

namespace mscclpp {

struct Endpoint::Impl {
  Impl(EndpointConfig const& config, Context::Impl& contextImpl);
  Impl(std::vector<char> const& serialization);

  EndpointConfig config_;
  uint64_t hostHash_;
  uint64_t pidHash_;

  // The following are only used for IB and are undefined for other transports.
  bool ibLocal_;
  bool ibNoAtomic_;
  std::shared_ptr<IbQp> ibQp_;
  IbQpInfo ibQpInfo_;

  // The following are only used for Ethernet and are undefined for other
  // transports.
  std::unique_ptr<Socket> socket_;
  SocketAddress socketAddress_;
  uint32_t volatile* abortFlag_;
  char netIfName_[MAX_IF_NAME_SIZE + 1];
};

}  // namespace mscclpp

#endif  // MSCCLPP_ENDPOINT_HPP_
