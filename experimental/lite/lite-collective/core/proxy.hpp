// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_HPP_
#define MSCCLPP_PROXY_HPP_

#include "fifo.hpp"
#include <functional>
#include <memory>

namespace mscclpp {

/// Return values for ProxyHandler.
enum class ProxyHandlerResult {
  /// Move to next trigger in FIFO.
  Continue,
  /// Stop and exit proxy.
  Stop,
};

class Proxy;

/// Handler function type for proxy.
using ProxyHandler = std::function<ProxyHandlerResult(ProxyTrigger)>;

/// Host-side proxy for PortChannels.
class Proxy {
 public:
  /// Constructor.
  /// @param handler Handler for each FIFO trigger.
  /// @param threadInit Optional function run once in the proxy thread before
  /// FIFO consumption.
  ///        The function should initialize thread runtime context before any
  ///        CUDA API call in that thread (for example, set CUDA device and
  ///        optionally bind NUMA affinity).
  /// @param fifoSize FIFO size (default: DEFAULT_FIFO_SIZE).
  Proxy(ProxyHandler handler, std::function<void()> threadInit,
        int fifoSize = DEFAULT_FIFO_SIZE);

  /// Constructor.
  /// @param handler Handler for each FIFO trigger.
  /// @param fifoSize FIFO size (default: DEFAULT_FIFO_SIZE).
  Proxy(ProxyHandler handler, int fifoSize = DEFAULT_FIFO_SIZE);

  /// Destructor. Stops proxy if running.
  ~Proxy();

  /// Start proxy.
  void start(bool blocking = false);

  /// Stop proxy.
  void stop();

  /// Get reference to FIFO used by proxy.
  /// @return Shared pointer to FIFO.
  std::shared_ptr<Fifo> fifo();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_HPP_
