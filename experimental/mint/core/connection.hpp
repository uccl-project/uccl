// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include "communicator.hpp"
#include "context.hpp"
#include "core.hpp"
#include "endpoint.hpp"
#include "gpu_utils.hpp"
#include "ib.hpp"
#include "registered_memory.hpp"
#include "socket.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace mscclpp {

/// Internal base class for connection implementations between two processes.
class BaseConnection {
 public:
  BaseConnection(std::shared_ptr<Context> context,
                 Endpoint const& localEndpoint);

  virtual ~BaseConnection() = default;

  virtual void write(RegisteredMemory dst, uint64_t dstOffset,
                     RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;

  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                             uint64_t* src, uint64_t newValue) = 0;

  virtual void flush(int64_t timeoutUsec = -1) = 0;

  /// Set the local address where remote updateAndSync operations should write.
  /// This is called by the receiver to specify where incoming signals should be
  /// written. Default implementation is a no-op for connections that don't need
  /// it.
  /// @param addr The local address for incoming writes.
  virtual void setRemoteUpdateDstAddr(uint64_t /*addr*/) {}

  virtual Transport transport() const = 0;

  virtual Transport remoteTransport() const = 0;

  std::shared_ptr<Context> context() const;

  Device const& localDevice() const;

  int getMaxWriteQueueSize() const;

  static std::shared_ptr<BaseConnection>& getImpl(Connection& conn) {
    return conn.impl_;
  }

 protected:
  friend class Context;
  friend class CudaIpcConnection;
  friend class IBConnection;
  friend class EthernetConnection;

  static Endpoint::Impl const& getImpl(Endpoint const& endpoint);
  static RegisteredMemory::Impl const& getImpl(RegisteredMemory const& memory);
  static Context::Impl& getImpl(Context& context);

  std::shared_ptr<Context> context_;
  Endpoint localEndpoint_;
  int maxWriteQueueSize_;
};

class CudaIpcConnection : public BaseConnection {
 private:
  std::shared_ptr<CudaIpcStream> stream_;

 public:
  CudaIpcConnection(std::shared_ptr<Context> context,
                    Endpoint const& localEndpoint,
                    Endpoint const& remoteEndpoint);

  Transport transport() const override;

  Transport remoteTransport() const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
             uint64_t srcOffset, uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src,
                     uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

class IBConnection : public BaseConnection {
 private:
  Transport transport_;
  Transport remoteTransport_;
  std::weak_ptr<IbQp> qp_;
  std::unique_ptr<uint64_t>
      dummyAtomicSource_;  // not used anywhere but IB needs a source
  RegisteredMemory dummyAtomicSourceMem_;
  mscclpp::TransportInfo dstTransportInfo_;

  // For write-with-imm mode (HostNoAtomic): uses RDMA write-with-imm to signal
  // instead of atomic operations, with a host thread forwarding to GPU for
  // memory consistency.
  bool ibNoAtomic_;
  std::thread recvThread_;
  std::atomic<bool> stopRecvThread_;
  int localGpuDeviceId_;  // Local GPU device ID for setting CUDA context in
                          // recv thread
  cudaStream_t signalStream_;

  // Write-with-imm design:
  // - Sender: 0-byte RDMA write-with-imm to dst MR, newValue in imm_data
  // (32-bit)
  // - Receiver: uses remoteUpdateDstAddr_ (set via setRemoteUpdateDstAddr) to
  // know where to write
  uint64_t remoteUpdateDstAddr_;

  void recvThreadFunc();

 public:
  IBConnection(std::shared_ptr<Context> context, Endpoint const& localEndpoint,
               Endpoint const& remoteEndpoint);
  ~IBConnection();

  /// Set the local address where remote updateAndSync operations will write.
  /// Must be called before the remote sends any updateAndSync in host-no-atomic
  /// mode.
  /// @param addr The local address for incoming writes.
  void setRemoteUpdateDstAddr(uint64_t addr) override;

  Transport transport() const override;

  Transport remoteTransport() const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
             uint64_t srcOffset, uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src,
                     uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

class EthernetConnection : public BaseConnection {
 private:
  std::unique_ptr<Socket> sendSocket_;
  std::unique_ptr<Socket> recvSocket_;
  std::thread threadRecvMessages_;
  uint32_t volatile* abortFlag_;
  const uint64_t sendBufferSize_;
  const uint64_t recvBufferSize_;
  std::vector<char> sendBuffer_;
  std::vector<char> recvBuffer_;

  void recvMessages();
  void sendMessage();

 public:
  EthernetConnection(std::shared_ptr<Context> context,
                     Endpoint const& localEndpoint,
                     Endpoint const& remoteEndpoint,
                     uint64_t sendBufferSize = 256 * 1024 * 1024,
                     uint64_t recvBufferSize = 256 * 1024 * 1024);

  ~EthernetConnection();

  Transport transport() const override;

  Transport remoteTransport() const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
             uint64_t srcOffset, uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src,
                     uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_HPP_
