// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_HPP_
#define MSCCLPP_EXECUTOR_HPP_

#include "core.hpp"
#include "gpu.hpp"
#include <memory>
#include <unordered_map>

namespace mscclpp {

/// Packet formats used by low-latency transport.
enum class PacketType {
  LL8,   // 8-byte low-latency packet.
  LL16,  // 16-byte low-latency packet.
};

/// Represents a compiled execution plan loaded from disk.
///
/// An ExecutionPlan encapsulates metadata about a collective algorithm such as
/// its name, the collective it implements, and the supported message-size
/// range. The concrete implementation is hidden behind the PIMPL pointer.
class ExecutionPlan {
 public:
  /// Construct an ExecutionPlan by loading the plan file at `planPath`.
  /// @param planPath Filesystem path to the serialized plan.
  /// @param rank The rank of the current process.
  ExecutionPlan(std::string const& planPath, int rank);

  /// Destructor.
  ~ExecutionPlan() = default;

  /// Return the human-readable name of the plan.
  std::string const& name() const;

  /// Return the collective implemented by this plan (e.g., "allreduce",
  /// "allgather").
  std::string const& collective() const;

  /// Minimum message size (in bytes) for which this plan is valid.
  size_t minMessageSize() const;

  /// Maximum message size (in bytes) for which this plan is valid.
  size_t maxMessageSize() const;

  /// Whether this plan performs the operation in-place.
  bool isInPlace() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class Executor;
};

/// High-level executor responsible for invoking execution plans on a
/// communicator.
class Executor {
 public:
  /// Construct an Executor using the provided communicator.
  /// @param comm Communicator instance used for underlying communication.
  /// @param defaultScratchBuffer Optional scratch buffer used by some plans
  /// (may be nullptr).
  Executor(std::shared_ptr<Communicator> comm,
           std::shared_ptr<char> defaultScratchBuffer = nullptr);

  /// Copy construction is disabled for Executor.
  Executor(Executor const&) = delete;

  /// Copy assignment is disabled for Executor.
  Executor& operator=(Executor const&) = delete;

  /// Destructor. Cleans up internal resources held by the Executor.
  ~Executor();

  /// Execute a plan.
  ///
  /// This method dispatches the given plan on the provided CUDA stream.
  ///
  /// @param rank Rank of the calling process.
  /// @param sendbuff Pointer to the send buffer.
  /// @param recvBuff Pointer to the receive buffer.
  /// @param sendBuffSize Size of the send buffer in bytes.
  /// @param recvBuffSize Size of the receive buffer in bytes.
  /// @param dataType Data type of elements in the buffers.
  /// @param plan The execution plan to run.
  /// @param stream CUDA stream to execute kernels/operations on.
  /// @param packetType Packet type used for low-latency transports (default:
  /// LL16).
  void execute(int rank, void* sendbuff, void* recvBuff, size_t sendBuffSize,
               size_t recvBuffSize, DataType dataType,
               ExecutionPlan const& plan, cudaStream_t stream,
               PacketType packetType = PacketType::LL16);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_HPP_
