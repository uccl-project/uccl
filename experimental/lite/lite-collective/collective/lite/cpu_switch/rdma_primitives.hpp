// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>

namespace mscclpp::lite {

class RdmaPrimitives {
 public:
  template <typename Port>
  void rdmaWrite(Port& port, size_t localOffset, size_t remoteOffset,
                 size_t bytes) const {
    port.write(localOffset, remoteOffset, bytes);
  }

  template <typename Port>
  void rdmaWriteAndFlush(Port& port, size_t localOffset, size_t remoteOffset,
                         size_t bytes) const {
    port.writeAndFlush(localOffset, remoteOffset, bytes);
  }

  template <typename Port>
  void signal(Port& port, uint64_t epoch) const {
    port.signal(epoch);
  }

  template <typename Port>
  void wait(Port const& port, uint64_t epoch) const {
    port.wait(epoch);
  }
};

}  // namespace mscclpp::lite
