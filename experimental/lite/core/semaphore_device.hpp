// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_DEVICE_HPP_
#define MSCCLPP_SEMAPHORE_DEVICE_HPP_

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#include "poll_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

/// Device-side handle for Host2DeviceSemaphore.
struct Host2DeviceSemaphoreDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Poll if the host has signaled.
  /// @return true if the host has signaled.
  MSCCLPP_DEVICE_INLINE bool poll() {
    bool signaled = (loadInbound() > loadExpectedInbound());
    if (signaled) incExpectedInbound();
    return signaled;
  }

  /// Wait for the host to signal.
  MSCCLPP_DEVICE_INLINE void wait(
      [[maybe_unused]] int64_t maxSpinCount = 100000000) {
    auto expected = incExpectedInbound();
    POLL_MAYBE_JAILBREAK((loadInbound() < expected), maxSpinCount);
  }

  /// Thread-safe read of expected inbound value.
  /// @return The expected inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadExpectedInbound() {
    return atomicLoad<uint64_t, scopeDevice>(expectedInboundToken,
                                             memoryOrderRelaxed);
  }

  /// Thread-safe increment of expected inbound value.
  /// @return The incremented expected inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t incExpectedInbound() {
    return atomicFetchAdd<uint64_t, scopeDevice>(expectedInboundToken, 1,
                                                 memoryOrderRelaxed) +
           1;
  }

  /// Thread-safe read of inbound value.
  /// @return The inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadInbound() {
    return atomicLoad<uint64_t, scopeSystem>(inboundToken, memoryOrderAcquire);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  /// A local memory space where a host thread (on behalf of the remote device)
  /// will write its semaphore value and the local device will read it.
  uint64_t* inboundToken;

  /// A local memory space where the local device stores the expected value of
  /// the inboundToken to wait for.
  uint64_t* expectedInboundToken;
};

/// Device-side handle for MemoryDevice2DeviceSemaphore.
struct MemoryDevice2DeviceSemaphoreDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Poll if remote device has signaled.
  /// @return true if remote device has signaled.
  MSCCLPP_DEVICE_INLINE bool poll() {
    bool signaled = (loadInbound() > loadExpectedInbound());
    if (signaled) incExpectedInbound();
    return signaled;
  }

  /// Wait for remote device to signal.
  MSCCLPP_DEVICE_INLINE void wait(
      [[maybe_unused]] int64_t maxSpinCount = 100000000) {
    auto expected = incExpectedInbound();
    POLL_MAYBE_JAILBREAK((loadInbound() < expected), maxSpinCount);
  }

  /// Relaxed wait; no memory completion guarantee. Use it only for
  /// synchronizing execution, not data.
  MSCCLPP_DEVICE_INLINE void relaxedWait(
      [[maybe_unused]] int64_t maxSpinCount = 100000000) {
    auto expected = incExpectedInbound();
    POLL_MAYBE_JAILBREAK((loadInboundRelaxed() < expected), maxSpinCount);
  }

  /// Signal remote device, ensures prior memory ops complete.
  MSCCLPP_DEVICE_INLINE void signal() {
    auto outbound = incOutbound();
#if defined(MSCCLPP_DEVICE_CUDA) && (__CUDA_ARCH__ == 800)
    // Using memoryOrderSeqCst is faster for A100.
    atomicStore(remoteInboundToken, outbound, memoryOrderSeqCst);
#else
    atomicStore(remoteInboundToken, outbound, memoryOrderRelease);
#endif
  }

  /// Relaxed signal; no memory completion guarantee. Use it only for
  /// synchronizing execution, not data.
  MSCCLPP_DEVICE_INLINE void relaxedSignal() {
    auto outbound = incOutbound();
    atomicStore(remoteInboundToken, outbound, memoryOrderRelaxed);
  }

  /// Thread-safe read of expected inbound value.
  /// @return The expected inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadExpectedInbound() {
    return atomicLoad<uint64_t, scopeDevice>(expectedInboundToken,
                                             memoryOrderRelaxed);
  }

  /// Thread-safe increment of expected inbound value.
  /// @return The incremented expected inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t incExpectedInbound() {
    return atomicFetchAdd<uint64_t, scopeDevice>(expectedInboundToken, 1,
                                                 memoryOrderRelaxed) +
           1;
  }

  /// Thread-safe read of inbound value.
  /// @return The inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadInbound() {
    return atomicLoad<uint64_t, scopeSystem>(inboundToken, memoryOrderAcquire);
  }

  /// Thread-safe read of inbound value without memory completion guarantee.
  /// @return The inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadInboundRelaxed() {
    return atomicLoad<uint64_t, scopeSystem>(inboundToken, memoryOrderRelaxed);
  }

  /// Thread-safe read of outbound value.
  /// @return The outbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadOutbound() {
    return atomicLoad<uint64_t, scopeDevice>(outboundToken, memoryOrderRelaxed);
  }

  /// Thread-safe increment of outbound value.
  /// @return The incremented outbound value.
  MSCCLPP_DEVICE_INLINE uint64_t incOutbound() {
    return atomicFetchAdd<uint64_t, scopeDevice>(outboundToken, 1,
                                                 memoryOrderRelaxed) +
           1;
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  /// A local memory space where the remote device will write its semaphore
  /// value and the local device will read it.
  uint64_t* inboundToken;

  /// A local memory space where the local device stores the semaphore value to
  /// be written to the remote device.
  uint64_t* outboundToken;

  /// A remote memory space where the local device writes its outboundToken on.
  /// This is inboundToken of the remote device.
  uint64_t* remoteInboundToken;

  /// A local memory space where the local device stores the expected value of
  /// the inboundToken to wait for.
  uint64_t* expectedInboundToken;
};

/// @deprecated Use MemoryDevice2DeviceSemaphoreDeviceHandle instead.
[[deprecated(
    "Use MemoryDevice2DeviceSemaphoreDeviceHandle "
    "instead.")]] typedef MemoryDevice2DeviceSemaphoreDeviceHandle
    SmDevice2DeviceSemaphoreDeviceHandle;

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_DEVICE_HPP_
