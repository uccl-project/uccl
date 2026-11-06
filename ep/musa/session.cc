#include "session.h"
#include "exchange_plan.h"
#include "kernels.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uccl::musa_ep {
namespace {
std::atomic<uint64_t> next_session{1};
enum class State : uint64_t { kIdle, kDispatched, kClosed };
enum class Operation : uint64_t { kDispatch, kCombine, kCancel, kClose };
}  // namespace

DeviceTensor::DeviceTensor(size_t rows, size_t hidden)
    : rows_(rows), hidden_(hidden) {
  size_t const bytes =
      checked_product(checked_product(rows, hidden), sizeof(float));
  if (bytes &&
      musaMalloc(reinterpret_cast<void**>(&data_), bytes) != musaSuccess)
    throw std::bad_alloc();
}
DeviceTensor::~DeviceTensor() {
  if (data_) musaFree(data_);
}
DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept {
  std::swap(data_, other.data_);
  std::swap(rows_, other.rows_);
  std::swap(hidden_, other.hidden_);
}
DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
  std::swap(data_, other.data_);
  std::swap(rows_, other.rows_);
  std::swap(hidden_, other.hidden_);
  return *this;
}

struct Session::Impl {
  struct Round {
    std::unique_ptr<RoutePlan> plan;
    ExchangePlan layout;
    ExpertGrouping grouping;
    std::vector<int64_t> send_meta, recv_meta;
    std::vector<uint64_t> send_counts, recv_counts, shapes;
    std::vector<size_t> token_map;
    std::vector<float> weights;
    DeviceTensor output;
  };

  SessionConfig config;
  MPI_Comm control;
  mcclComm_t comm;
  int rank = 0, ranks = 0;
  uint64_t origin_rank = 0, id = next_session.fetch_add(1), generation = 0;
  State state = State::kIdle;
  musaStream_t stream = nullptr;
  void* arena = nullptr;
  size_t capacity = 0;
  std::vector<uint64_t> consensus;
  std::unique_ptr<Round> round;

  [[noreturn]] void fatal(char const* call, int code) {
    std::fprintf(stderr, "MUSA EP rank=%d generation=%llu: %s failed (%d)\n",
                 rank, static_cast<unsigned long long>(generation), call, code);
    MPI_Abort(control, code ? code : 1);
    std::abort();
  }
  void mpi(int result, char const* call) {
    if (result != MPI_SUCCESS) fatal(call, result);
  }
  void gpu(musaError_t result, char const* call) {
    if (result != musaSuccess) fatal(call, int(result));
  }
  void net(int result, char const* call) {
    if (result != mcclSuccess) fatal(call, result);
  }
  void agree(bool valid, char const* phase) {
    int const local = valid ? 1 : 0;
    int all = 0;
    mpi(MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, control),
        "status agreement");
    if (!all) throw std::invalid_argument(phase);
  }
  template <class F>
  void prepare(char const* phase, F f) {
    bool valid = true;
    try {
      f();
    } catch (std::exception const& error) {
      std::fprintf(stderr, "MUSA EP rank=%d %s: %s\n", rank, phase,
                   error.what());
      valid = false;
    }
    agree(valid, phase);
  }
  void operation(Operation op, bool valid) {
    uint64_t local[] = {origin_rank, id, uint64_t(op), uint64_t(state),
                        generation};
    mpi(MPI_Allgather(local, 5, MPI_UINT64_T, consensus.data(), 5, MPI_UINT64_T,
                      control),
        "operation agreement");
    for (int peer = 0; peer < ranks; ++peer)
      for (size_t i = 0; i < 5; ++i)
        valid = valid && consensus[size_t(peer) * 5 + i] == local[i];
    agree(valid, "EP operation/state/handle mismatch");
  }
  bool owns(RouteHandle const& handle) const {
    return round && handle.owner_root_ == origin_rank && handle.owner_ == id &&
           handle.generation_ == generation;
  }
  template <class T>
  T* buffer(ExchangePlan const& layout, BufferIndex index) const {
    if (!arena) return nullptr;
    return reinterpret_cast<T*>(static_cast<char*>(arena) +
                                layout.buffers[index]);
  }
  void upload(void* dst, void const* src, size_t bytes) {
    if (bytes)
      gpu(musaMemcpyAsync(dst, src, bytes, musaMemcpyHostToDevice, stream),
          "upload");
  }
  void transfer(float const* send, float* recv,
                std::vector<size_t> const& send_counts,
                std::vector<size_t> const& send_offsets,
                std::vector<size_t> const& recv_counts,
                std::vector<size_t> const& recv_offsets) {
    size_t const hidden = config.hidden;
    if (send_counts[rank])
      gpu(musaMemcpyAsync(recv + recv_offsets[rank] * hidden,
                          send + send_offsets[rank] * hidden,
                          send_counts[rank] * hidden * sizeof(float),
                          musaMemcpyDeviceToDevice, stream),
          "self copy");
    net(mcclGroupStart(), "group start");
    for (int peer = 0; peer < ranks; ++peer) {
      if (peer == rank) continue;
      if (recv_counts[peer])
        net(mcclRecv(recv + recv_offsets[peer] * hidden,
                     recv_counts[peer] * hidden, mcclFloat, peer, comm, stream),
            "recv");
      if (send_counts[peer])
        net(mcclSend(send + send_offsets[peer] * hidden,
                     send_counts[peer] * hidden, mcclFloat, peer, comm, stream),
            "send");
    }
    net(mcclGroupEnd(), "group end");
  }
};

Session::Session(SessionConfig config, MPI_Comm control, mcclComm_t comm)
    : impl_(std::make_unique<Impl>()) {
  auto& s = *impl_;
  s.config = config;
  s.control = control;
  s.comm = comm;
  s.mpi(MPI_Comm_rank(control, &s.rank), "rank");
  s.mpi(MPI_Comm_size(control, &s.ranks), "size");
  int world_rank = 0;
  s.mpi(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), "world rank");
  s.prepare("session configuration", [&] {
    if (!s.id) throw std::overflow_error("EP Session identity exhausted");
    make_plan(s.rank, s.ranks, config.experts, 0, config.hidden, config.topk,
              {}, {}, config.max_routes, config.byte_budget);
    s.consensus.resize(checked_product(size_t(s.ranks), 6));
  });
  uint64_t local[] = {1,           config.experts, config.hidden,
                      config.topk, s.id,           uint64_t(world_rank)};
  s.mpi(MPI_Allgather(local, 6, MPI_UINT64_T, s.consensus.data(), 6,
                      MPI_UINT64_T, control),
        "configuration agreement");
  bool matches = true;
  for (int peer = 0; peer < s.ranks; ++peer)
    for (size_t i = 0; i < 4; ++i)
      matches = matches && s.consensus[size_t(peer) * 6 + i] == local[i];
  s.agree(matches, "EP ranks have different configurations");
  s.id = s.consensus[4];  // Rank zero assigns one identity for this Session.
  s.origin_rank = s.consensus[5];
  bool const created = musaStreamCreate(&s.stream) == musaSuccess;
  try {
    s.agree(created, "EP stream allocation");
  } catch (...) {
    if (created) musaStreamDestroy(s.stream);
    throw;
  }
}

Session::~Session() {
  // Public operations return synchronously. A live handle must be explicitly
  // combined/cancelled; the destructor cannot perform a hidden collective.
  if (impl_->round) std::terminate();
  if (impl_->arena) musaFree(impl_->arena);
  if (impl_->stream) musaStreamDestroy(impl_->stream);
}

DispatchResult Session::dispatch(float const* x, size_t tokens,
                                 std::vector<int64_t> const& indices,
                                 std::vector<float> const& weights,
                                 musaEvent_t input_ready) {
  auto& s = *impl_;
  s.operation(Operation::kDispatch,
              s.state == State::kIdle &&
                  s.generation != std::numeric_limits<uint64_t>::max());
  std::unique_ptr<Impl::Round> pending;
  s.prepare("dispatch input/plan", [&] {
    if (tokens && (!x || !input_ready))
      throw std::invalid_argument(
          "nonempty input requires a buffer and ready event");
    pending = std::make_unique<Impl::Round>();
    auto& r = *pending;
    r.plan = std::make_unique<RoutePlan>(
        make_plan(s.rank, s.ranks, s.config.experts, tokens, s.config.hidden,
                  s.config.topk, indices, weights, s.config.max_routes,
                  s.config.byte_budget));
    r.send_meta = route_metadata(*r.plan);
    r.send_counts.assign(r.plan->send_counts.begin(),
                         r.plan->send_counts.end());
    r.recv_counts.resize(s.ranks);
    r.shapes.resize(size_t(s.ranks) * 2);
  });
  auto& r = *pending;
  s.mpi(MPI_Alltoall(r.send_counts.data(), 1, MPI_UINT64_T,
                     r.recv_counts.data(), 1, MPI_UINT64_T, s.control),
        "route counts");
  uint64_t shape[] = {tokens, s.config.topk};
  s.mpi(MPI_Allgather(shape, 2, MPI_UINT64_T, r.shapes.data(), 2, MPI_UINT64_T,
                      s.control),
        "source shapes");
  s.prepare("dispatch receive capacity", [&] {
    r.layout = make_exchange_plan(
        *r.plan, {r.recv_counts.begin(), r.recv_counts.end()}, s.config.experts,
        s.config.max_routes, s.config.byte_budget);
    r.recv_meta.resize(checked_product(r.layout.recv_offsets.back(), 4));
  });
  s.mpi(
      MPI_Alltoallv(r.send_meta.data(), r.layout.send_meta_counts.data(),
                    r.layout.send_meta_offsets.data(), MPI_INT64_T,
                    r.recv_meta.data(), r.layout.recv_meta_counts.data(),
                    r.layout.recv_meta_offsets.data(), MPI_INT64_T, s.control),
      "route metadata");
  s.prepare("dispatch metadata/buffers", [&] {
    std::vector<std::pair<size_t, size_t>> shapes;
    for (int peer = 0; peer < s.ranks; ++peer)
      shapes.emplace_back(r.shapes[peer * 2], r.shapes[peer * 2 + 1]);
    r.grouping =
        group_metadata(r.recv_meta, r.layout, s.rank, s.config.experts, shapes);
    r.token_map.reserve(r.plan->routes.size());
    r.weights.resize(checked_product(tokens, s.config.topk));
    for (auto const& route : r.plan->routes) {
      r.token_map.push_back(route.token);
      r.weights[route.token * s.config.topk + route.lane] = route.weight;
    }
    // Keep a high-watermark arena across rounds. Shrink if retained capacity
    // plus this round's output would exceed the configured GPU budget.
    if (s.capacity < r.layout.arena_bytes ||
        (s.capacity > r.layout.arena_bytes &&
         checked_sum(r.layout.required_bytes,
                     s.capacity - r.layout.arena_bytes) >
             s.config.byte_budget)) {
      if (s.arena) {
        s.gpu(musaFree(s.arena), "arena release");
        s.arena = nullptr;
        s.capacity = 0;
      }
      if (r.layout.arena_bytes) {
        if (musaMalloc(&s.arena, r.layout.arena_bytes) != musaSuccess)
          throw std::bad_alloc();
        s.capacity = r.layout.arena_bytes;
      }
    }
    r.output = DeviceTensor(tokens, s.config.hidden);
  });
  auto const& l = r.layout;
  auto* send = s.buffer<float>(l, kSend);
  auto* recv = s.buffer<float>(l, kRecv);
  auto* expert = s.buffer<float>(l, kExpert);
  auto* token_map = s.buffer<size_t>(l, kTokens);
  s.upload(token_map, r.token_map.data(), r.token_map.size() * sizeof(size_t));
  s.upload(s.buffer<size_t>(l, kPermutation), r.grouping.expert_to_recv.data(),
           r.grouping.expert_to_recv.size() * sizeof(size_t));
  s.upload(s.buffer<size_t>(l, kLanes), r.plan->lane_to_route.data(),
           r.plan->lane_to_route.size() * sizeof(size_t));
  s.upload(s.buffer<float>(l, kWeights), r.weights.data(),
           r.weights.size() * sizeof(float));
  if (tokens)
    s.gpu(musaStreamWaitEvent(s.stream, input_ready, 0), "input ready event");
  s.gpu(gather_rows(x, send, token_map, r.plan->routes.size(), s.config.hidden,
                    s.stream),
        "pack");
  s.gpu(musaStreamSynchronize(s.stream), "pack completion");
  s.transfer(send, recv, r.plan->send_counts, r.plan->send_offsets,
             l.recv_counts, l.recv_offsets);
  s.gpu(gather_rows(recv, expert, s.buffer<size_t>(l, kPermutation),
                    l.recv_offsets.back(), s.config.hidden, s.stream),
        "expert gather");
  s.gpu(musaStreamSynchronize(s.stream), "dispatch completion");
  ++s.generation;
  s.state = State::kDispatched;
  RouteHandle handle;
  handle.owner_root_ = s.origin_rank;
  handle.owner_ = s.id;
  handle.generation_ = s.generation;
  size_t const rows = l.recv_offsets.back();
  s.round = std::move(pending);
  return {expert, rows, s.config.hidden, handle};
}

DeviceTensor Session::combine(float const* expert_output, size_t rows,
                              size_t hidden, RouteHandle const& handle,
                              musaEvent_t expert_ready) {
  auto& s = *impl_;
  s.operation(Operation::kCombine,
              s.state == State::kDispatched && s.owns(handle) &&
                  hidden == s.config.hidden &&
                  rows == (s.round ? s.round->layout.recv_offsets.back() : 0) &&
                  (!rows || (expert_output && expert_ready)));
  auto& r = *s.round;
  auto const& l = r.layout;
  if (rows)
    s.gpu(musaStreamWaitEvent(s.stream, expert_ready, 0), "expert ready event");
  auto* recv = s.buffer<float>(l, kRecv);
  auto* send = s.buffer<float>(l, kSend);
  s.gpu(scatter_rows(expert_output, recv, s.buffer<size_t>(l, kPermutation),
                     rows, hidden, s.stream),
        "inverse permutation");
  s.gpu(musaStreamSynchronize(s.stream), "inverse completion");
  s.transfer(recv, send, l.recv_counts, l.recv_offsets, r.plan->send_counts,
             r.plan->send_offsets);
  s.gpu(weighted_combine(send, r.output.data(), s.buffer<size_t>(l, kLanes),
                         s.buffer<float>(l, kWeights), r.plan->tokens, hidden,
                         s.config.topk, s.stream),
        "weighted combine");
  s.gpu(musaStreamSynchronize(s.stream), "combine completion");
  DeviceTensor output = std::move(r.output);
  s.round.reset();
  s.state = State::kIdle;
  return output;
}

std::vector<size_t> const& Session::expert_offsets() const {
  if (!impl_->round) throw std::logic_error("no dispatched EP round");
  return impl_->round->grouping.expert_offsets;
}
void Session::cancel_completed_round(RouteHandle const& handle,
                                     musaEvent_t release_ready) {
  auto& s = *impl_;
  s.operation(
      Operation::kCancel,
      s.state == State::kDispatched && s.owns(handle) &&
          (release_ready || (s.round && !s.round->layout.recv_offsets.back())));
  if (release_ready)
    s.gpu(musaStreamWaitEvent(s.stream, release_ready, 0),
          "cancel release event");
  s.gpu(musaStreamSynchronize(s.stream), "cancel release completion");
  s.round.reset();
  s.state = State::kIdle;
}
void Session::close() {
  auto& s = *impl_;
  s.operation(Operation::kClose, s.state == State::kIdle);
  if (s.arena) s.gpu(musaFree(s.arena), "arena close");
  s.arena = nullptr;
  s.capacity = 0;
  s.state = State::kClosed;
}
}  // namespace uccl::musa_ep
