#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <mccl.h>
#include <mpi.h>
#include <musa_runtime.h>

namespace uccl::musa_ep {
struct SessionConfig {
  size_t experts, hidden, topk;
  size_t max_routes = 65536;
  size_t byte_budget = size_t{1} << 30;
};

class RouteHandle {
 public:
  RouteHandle() = default;

 private:
  friend class Session;
  uint64_t owner_root_ = 0, owner_ = 0, generation_ = 0;
};

class DeviceTensor {
 public:
  DeviceTensor() = default;
  DeviceTensor(size_t rows, size_t hidden);
  ~DeviceTensor();
  DeviceTensor(DeviceTensor&& other) noexcept;
  DeviceTensor& operator=(DeviceTensor&& other) noexcept;
  DeviceTensor(DeviceTensor const&) = delete;
  DeviceTensor& operator=(DeviceTensor const&) = delete;
  float* data() const { return data_; }
  size_t rows() const { return rows_; }
  size_t hidden() const { return hidden_; }

 private:
  float* data_ = nullptr;
  size_t rows_ = 0, hidden_ = 0;
};

struct DispatchResult {
  // Borrowed until combine/cancel; rows are grouped by local expert.
  float const* data;
  size_t rows, hidden;
  RouteHandle handle;
};

// Experimental synchronous FP32 EP. One host thread and one round per Session.
// MPI/MCCL communicators are borrowed, with identical rank order; caller keeps
// them alive and the current MUSA device unchanged through close/destruction.
// Every rank must call the same operation. Peer failure requires a bounded
// external launcher: this class does not recover communicators or cancel WRs.
class Session {
 public:
  Session(SessionConfig config, MPI_Comm control, mcclComm_t comm);
  ~Session();
  Session(Session const&) = delete;
  Session& operator=(Session const&) = delete;

  DispatchResult dispatch(float const* x, size_t tokens,
                          std::vector<int64_t> const& indices,
                          std::vector<float> const& weights,
                          musaEvent_t input_ready);
  DeviceTensor combine(float const* expert_output, size_t rows, size_t hidden,
                       RouteHandle const& handle, musaEvent_t expert_ready);
  std::vector<size_t> const& expert_offsets() const;
  // If rows are nonempty, release_ready must cover all caller expert work
  // that may still read the borrowed dispatch buffer.
  void cancel_completed_round(RouteHandle const& handle,
                              musaEvent_t release_ready);
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uccl::musa_ep
