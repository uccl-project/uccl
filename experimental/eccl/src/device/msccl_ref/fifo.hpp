// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_HPP_
#define MSCCLPP_FIFO_HPP_

// TODO: delete later

#include "fifo_device.hpp"
#include <memory>

namespace mscclpp {

constexpr size_t DEFAULT_FIFO_SIZE = 512;

/// Host-side proxy FIFO for device-produced work elements.
class Fifo {
 public:
  /// Constructor.
  /// @param size Number of entries (default: DEFAULT_FIFO_SIZE).
  Fifo(int size = DEFAULT_FIFO_SIZE);

  /// Destructor.
  ~Fifo();

  /// Poll and get the trigger at the head.
  /// @return ProxyTrigger at the head of the FIFO.
  ProxyTrigger poll();

  /// Remove the head trigger.
  void pop();

  /// Get FIFO size.
  /// @return Number of entries in the FIFO.
  int size() const;

  /// Get device-side FIFO handle.
  /// @return FifoDeviceHandle for device access.
  FifoDeviceHandle deviceHandle() const;

  [[deprecated(
      "flushTail() is now no-op and no longer needed. This will be removed in "
      "a future release.")]] void
  flushTail([[maybe_unused]] bool sync = false) {}

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_HPP_