#include "../include/config.h"
#include "../include/gpu_rt.h"
#include "../src/transport/communicator.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace UKernel {
namespace Transport {
namespace Python {

namespace {

torch::Tensor tensor_from_python(nb::handle obj, char const* arg_name) {
  PyObject* py_obj = obj.ptr();
  if (!THPVariable_Check(py_obj)) {
    throw std::invalid_argument(std::string(arg_name) +
                                " must be a torch.Tensor");
  }
  return THPVariable_Unpack(py_obj);
}

PreferredTransport parse_transport(std::string const& value) {
  if (value == "auto") return PreferredTransport::Auto;
  if (value == "ipc") return PreferredTransport::Ipc;
  if (value == "rdma") return PreferredTransport::Rdma;
  if (value == "uccl") return PreferredTransport::Uccl;
  if (value == "tcp") return PreferredTransport::Tcp;
  throw std::invalid_argument("unsupported transport: " + value);
}

}  // namespace

class Communicator {
 public:
  Communicator(int gpu_id, int rank, int world_size, std::string exchanger_ip,
               int exchanger_port, std::string transport = "auto",
               int local_id = -1)
      : comm_(std::make_shared<UKernel::Transport::Communicator>(
            gpu_id, rank, world_size,
            std::make_shared<UKernel::Transport::CommunicatorConfig>(
                UKernel::Transport::CommunicatorConfig{
                    exchanger_ip,
                    exchanger_port,
                    local_id,
                    parse_transport(transport),
                }))) {
    GPU_RT_CHECK(gpuSetDevice(gpu_id));
  }

  ~Communicator() {
    std::vector<void*> pinned_ptrs;
    {
      std::lock_guard<std::mutex> lk(mu_);
      pinned_ptrs.reserve(pinned_tensors_.size());
      for (auto const& it : pinned_tensors_) {
        pinned_ptrs.push_back(it.first);
      }
      pinned_tensors_.clear();
      pending_requests_.clear();
    }
    for (void* ptr : pinned_ptrs) {
      if (ptr != nullptr) {
        comm_->dereg_mr(ptr);
      }
    }
  }

  int rank() const { return comm_->rank(); }
  int world_size() const { return comm_->world_size(); }

  bool connect_peer(int peer_rank) { return comm_->connect(peer_rank); }
  bool accept_peer(int peer_rank) { return comm_->accept(peer_rank); }

  uint32_t pin_tensor(nb::handle tensor) {
    torch::Tensor t = tensor_from_python(tensor, "tensor");
    if (!t.is_cuda()) {
      throw std::invalid_argument("pin_tensor requires a CUDA tensor");
    }
    if (!t.is_contiguous()) {
      throw std::invalid_argument("pin_tensor requires a contiguous tensor");
    }
    size_t total_bytes =
        static_cast<size_t>(t.numel()) * static_cast<size_t>(t.element_size());
    if (total_bytes == 0) {
      throw std::invalid_argument("pin_tensor requires non-empty tensor");
    }

    void* ptr = t.data_ptr();
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = pinned_tensors_.find(ptr);
      if (it != pinned_tensors_.end()) {
        if (it->second.bytes != total_bytes) {
          throw std::runtime_error(
              "pinned tensor size changed for the same pointer");
        }
        return it->second.mr_id;
      }
    }

    auto mr = comm_->reg_mr(ptr, total_bytes);
    if (mr.id == 0) {
      throw std::runtime_error("pin_tensor failed to register MR");
    }

    std::lock_guard<std::mutex> lk(mu_);
    pinned_tensors_[ptr] = PinnedTensor{std::move(t), mr.id, total_bytes};
    return mr.id;
  }

  bool unpin_tensor(nb::handle tensor) {
    torch::Tensor t = tensor_from_python(tensor, "tensor");
    void* ptr = t.data_ptr();
    PinnedTensor pinned;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = pinned_tensors_.find(ptr);
      if (it == pinned_tensors_.end()) return false;
      pinned = std::move(it->second);
      pinned_tensors_.erase(it);
    }
    return comm_->dereg_mr(ptr);
  }

  bool publish_mr(int peer_rank, uint32_t buffer_id, uint32_t mr_id,
                  uint64_t generation = 0) {
    // Serialize publishes so auto-generation stays deterministic while
    // avoiding holding mr_mu_ during OOB/network publish.
    std::lock_guard<std::mutex> pub_lk(publish_mu_);
    MR local = comm_->get_local_mr(mr_id);
    auto const key = make_peer_buffer_key(peer_rank, buffer_id);
    // generation=0 means auto-version by (peer, buffer_id). We only advance
    // local publish cursor after notify succeeds to avoid version gaps.
    uint64_t effective_generation = generation;
    {
      std::lock_guard<std::mutex> lk(mr_mu_);
      if (effective_generation == 0) {
        auto it = local_published_generation_.find(key);
        effective_generation =
            (it == local_published_generation_.end()) ? 1 : (it->second + 1);
      }
    }
    NamedMRInfos infos{};
    infos.generation = effective_generation;
    infos.entries.push_back(NamedMR{buffer_id, local});
    if (!comm_->notify_named_mrs(peer_rank, effective_generation, infos)) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lk(mr_mu_);
      local_published_generation_[key] =
          std::max(local_published_generation_[key], effective_generation);
    }
    return true;
  }

  uint32_t wait_mr(int peer_rank, uint32_t buffer_id, uint64_t generation = 0) {
    auto const key = make_peer_buffer_key(peer_rank, buffer_id);
    uint64_t target_generation = generation;
    if (target_generation == 0) {
      // generation=0 means "wait for next version" of this (peer, buffer_id).
      std::lock_guard<std::mutex> lk(mr_mu_);
      auto it = remote_seen_generation_.find(key);
      uint64_t const last_seen =
          (it == remote_seen_generation_.end()) ? 0 : it->second;
      target_generation = last_seen + 1;
    } else {
      // Explicit generation can be satisfied from cache.
      std::lock_guard<std::mutex> lk(mr_mu_);
      auto it = remote_mr_cache_.find(key);
      if (it != remote_mr_cache_.end() && it->second.generation >= generation) {
        return it->second.mr_id;
      }
    }

    for (;;) {
      NamedMRInfos infos{};
      if (!comm_->wait_named_mrs(peer_rank, target_generation, infos)) {
        throw std::runtime_error(
            "wait_mr failed to fetch remote named MR table");
      }
      bool found = false;
      {
        std::lock_guard<std::mutex> lk(mr_mu_);
        // Cache every entry from this generation for future lookup.
        for (auto const& entry : infos.entries) {
          auto const entry_key =
              make_peer_buffer_key(peer_rank, entry.buffer_id);
          remote_mr_cache_[entry_key] =
              RemoteMRCacheItem{entry.mr.id, infos.generation};
          remote_seen_generation_[entry_key] =
              std::max(remote_seen_generation_[entry_key], infos.generation);
          if (entry.buffer_id == buffer_id) {
            found = true;
          }
        }
        if (found) {
          return remote_mr_cache_.at(key).mr_id;
        }
      }
      // If caller requires explicit generation, missing buffer is a hard error.
      if (generation != 0) {
        throw std::runtime_error("wait_mr missing requested buffer_id");
      }
      // Auto mode: continue scanning next generation until this buffer is published.
      ++target_generation;
    }
  }

  uint32_t remote_mr(int peer_rank, uint32_t buffer_id) const {
    std::lock_guard<std::mutex> lk(mr_mu_);
    auto const key = make_peer_buffer_key(peer_rank, buffer_id);
    auto it = remote_mr_cache_.find(key);
    if (it == remote_mr_cache_.end()) {
      throw std::runtime_error("remote_mr not found for requested buffer_id");
    }
    return it->second.mr_id;
  }

  uint64_t isend(int peer_rank, nb::handle tensor, size_t offset = 0,
                 size_t len = 0, uint32_t remote_mr_id = 0,
                 size_t remote_offset = 0) {
    torch::Tensor t = tensor_from_python(tensor, "tensor");
    if (!t.is_cuda()) {
      throw std::invalid_argument("isend requires a CUDA tensor");
    }
    if (!t.is_contiguous()) {
      throw std::invalid_argument("isend requires a contiguous tensor");
    }
    size_t elem_bytes = static_cast<size_t>(t.element_size());
    size_t total_bytes = static_cast<size_t>(t.numel()) * elem_bytes;
    if (len == 0) len = total_bytes;
    if (offset + len > total_bytes) {
      throw std::invalid_argument("isend offset+len exceeds tensor size");
    }
    auto pinned_mr = find_pinned_mr_id(t.data_ptr(), total_bytes);
    uint32_t mr_id = pinned_mr.has_value()
                         ? *pinned_mr
                         : comm_->reg_mr(t.data_ptr(), total_bytes).id;
    std::optional<UKernel::Transport::RemoteSlice> dst_hint = std::nullopt;
    if (remote_mr_id != 0) {
      // Optional explicit destination hint for one-sided fast path.
      dst_hint =
          UKernel::Transport::RemoteSlice{remote_mr_id, remote_offset, {}, 0};
    }
    uint64_t req =
        comm_->isend(peer_rank, UKernel::Transport::LocalSlice{mr_id, offset, len},
                     dst_hint);
    if (req == 0) {
      if (!pinned_mr.has_value()) {
        comm_->dereg_mr(t.data_ptr());
      }
      return 0;
    }
    track_request(req, std::move(t),
                  pinned_mr.has_value() ? nullptr : t.data_ptr());
    return req;
  }

  uint64_t isend_buffer(int peer_rank, nb::handle tensor, uint32_t buffer_id,
                        size_t offset = 0, size_t len = 0,
                        size_t remote_offset = 0,
                        uint64_t min_generation = 0) {
    uint32_t remote_mr_id = 0;
    if (min_generation == 0) {
      // Fast path for steady-state traffic: reuse cached latest remote MR when
      // available, and avoid waiting for a "next generation" update.
      try {
        remote_mr_id = remote_mr(peer_rank, buffer_id);
      } catch (std::exception const&) {
        // Cold start: no cache yet, wait for at least generation 1.
        remote_mr_id = wait_mr(peer_rank, buffer_id, 1);
      }
    } else {
      // Caller requires at least a specific generation.
      remote_mr_id = wait_mr(peer_rank, buffer_id, min_generation);
    }
    return isend(peer_rank, tensor, offset, len, remote_mr_id, remote_offset);
  }

  uint64_t irecv(int peer_rank, nb::handle tensor, size_t offset = 0,
                 size_t len = 0) {
    torch::Tensor t = tensor_from_python(tensor, "tensor");
    if (!t.is_cuda()) {
      throw std::invalid_argument("irecv requires a CUDA tensor");
    }
    if (!t.is_contiguous()) {
      throw std::invalid_argument("irecv requires a contiguous tensor");
    }
    size_t elem_bytes = static_cast<size_t>(t.element_size());
    size_t total_bytes = static_cast<size_t>(t.numel()) * elem_bytes;
    if (len == 0) len = total_bytes;
    if (offset + len > total_bytes) {
      throw std::invalid_argument("irecv offset+len exceeds tensor size");
    }
    auto pinned_mr = find_pinned_mr_id(t.data_ptr(), total_bytes);
    uint32_t mr_id = pinned_mr.has_value()
                         ? *pinned_mr
                         : comm_->reg_mr(t.data_ptr(), total_bytes).id;
    uint64_t req = comm_->irecv(
        peer_rank, UKernel::Transport::LocalSlice{mr_id, offset, len});
    if (req == 0) {
      if (!pinned_mr.has_value()) {
        comm_->dereg_mr(t.data_ptr());
      }
      return 0;
    }
    track_request(req, std::move(t),
                  pinned_mr.has_value() ? nullptr : t.data_ptr());
    return req;
  }

  bool poll(uint64_t req) {
    try {
      bool done = comm_->poll(static_cast<unsigned>(req));
      if (done) cleanup_request(req);
      return done;
    } catch (...) {
      cleanup_request(req);
      throw;
    }
  }
  void release(uint64_t req) {
    comm_->release(static_cast<unsigned>(req));
    cleanup_request(req);
  }
  bool wait_finish(uint64_t req) {
    try {
      bool ok = comm_->wait_finish(static_cast<unsigned>(req));
      cleanup_request(req);
      return ok;
    } catch (...) {
      cleanup_request(req);
      throw;
    }
  }

  bool wait_finish_multi(std::vector<uint64_t> reqs) {
    std::vector<unsigned> unsigned_reqs(reqs.size());
    for (size_t i = 0; i < reqs.size(); ++i) {
      unsigned_reqs[i] = static_cast<unsigned>(reqs[i]);
    }
    try {
      bool ok = comm_->wait_finish(unsigned_reqs);
      for (uint64_t req : reqs) cleanup_request(req);
      return ok;
    } catch (...) {
      for (uint64_t req : reqs) cleanup_request(req);
      throw;
    }
  }

  std::string peer_transport(int peer_rank) const {
    switch (comm_->peer_transport_kind(peer_rank)) {
      case PeerTransportKind::Ipc:
        return "ipc";
      case PeerTransportKind::Rdma:
        return "rdma";
      case PeerTransportKind::Uccl:
        return "uccl";
      case PeerTransportKind::Tcp:
        return "tcp";
      default:
        return "unknown";
    }
  }

  bool same_host(int peer_rank) const { return comm_->same_host(peer_rank); }

  void send(int peer_rank, nb::handle tensor, uint32_t remote_mr_id = 0,
            size_t remote_offset = 0) {
    uint64_t req = isend(peer_rank, tensor, 0, 0, remote_mr_id, remote_offset);
    wait_finish(req);
  }

  void send_buffer(int peer_rank, nb::handle tensor, uint32_t buffer_id,
                   size_t remote_offset = 0, uint64_t min_generation = 0) {
    uint64_t req =
        isend_buffer(peer_rank, tensor, buffer_id, 0, 0, remote_offset,
                     min_generation);
    wait_finish(req);
  }

  void recv(int peer_rank, nb::handle tensor) {
    uint64_t req = irecv(peer_rank, tensor);
    wait_finish(req);
  }

 private:
  struct PendingTensorRequest {
    torch::Tensor tensor;
    void* local_buf = nullptr;
  };

  struct PinnedTensor {
    torch::Tensor tensor;
    uint32_t mr_id = 0;
    size_t bytes = 0;
  };

  struct RemoteMRCacheItem {
    uint32_t mr_id = 0;
    uint64_t generation = 0;
  };

  static uint64_t make_peer_buffer_key(int peer_rank, uint32_t buffer_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(peer_rank)) << 32) |
           static_cast<uint64_t>(buffer_id);
  }

  std::optional<uint32_t> find_pinned_mr_id(void* ptr, size_t bytes) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pinned_tensors_.find(ptr);
    if (it == pinned_tensors_.end()) return std::nullopt;
    if (it->second.bytes != bytes) {
      throw std::runtime_error("pinned tensor size mismatch");
    }
    return it->second.mr_id;
  }

  void track_request(uint64_t req, torch::Tensor tensor, void* local_buf) {
    std::lock_guard<std::mutex> lk(mu_);
    pending_requests_[req] = PendingTensorRequest{std::move(tensor), local_buf};
  }

  void cleanup_request(uint64_t req) {
    PendingTensorRequest pending;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = pending_requests_.find(req);
      if (it == pending_requests_.end()) return;
      pending = std::move(it->second);
      pending_requests_.erase(it);
    }
    if (pending.local_buf != nullptr) {
      comm_->dereg_mr(pending.local_buf);
    }
  }

  std::shared_ptr<UKernel::Transport::Communicator> comm_;
  std::unordered_map<void*, PinnedTensor> pinned_tensors_;
  std::unordered_map<uint64_t, PendingTensorRequest> pending_requests_;
  // Tracks local published generation for (peer, buffer_id).
  std::unordered_map<uint64_t, uint64_t> local_published_generation_;
  // Tracks latest seen remote generation for (peer, buffer_id).
  std::unordered_map<uint64_t, uint64_t> remote_seen_generation_;
  // Caches latest remote MR id for (peer, buffer_id).
  std::unordered_map<uint64_t, RemoteMRCacheItem> remote_mr_cache_;
  mutable std::mutex publish_mu_;
  mutable std::mutex mr_mu_;
  mutable std::mutex mu_;
};

}  // namespace Python
}  // namespace Transport
}  // namespace UKernel

NB_MODULE(TORCH_EXTENSION_NAME, m) {
  using UKernel::Transport::Python::Communicator;

  nb::class_<Communicator>(m, "Communicator")
      .def(nb::init<int, int, int, std::string, int, std::string, int>(),
           nb::arg("gpu_id"), nb::arg("rank"), nb::arg("world_size"),
           nb::arg("exchanger_ip") = "127.0.0.1",
           nb::arg("exchanger_port") = 6979, nb::arg("transport") = "auto",
           nb::arg("local_id") = -1)
      .def_prop_ro("rank", &Communicator::rank)
      .def_prop_ro("world_size", &Communicator::world_size)
      .def("connect_peer", &Communicator::connect_peer, nb::arg("peer_rank"))
      .def("accept_peer", &Communicator::accept_peer, nb::arg("peer_rank"))
      .def("pin_tensor", &Communicator::pin_tensor, nb::arg("tensor"))
      .def("unpin_tensor", &Communicator::unpin_tensor, nb::arg("tensor"))
      .def("publish_mr", &Communicator::publish_mr, nb::arg("peer_rank"),
           nb::arg("buffer_id"), nb::arg("mr_id"),
           nb::arg("generation") = 0)
      .def("wait_mr", &Communicator::wait_mr, nb::arg("peer_rank"),
           nb::arg("buffer_id"), nb::arg("generation") = 0)
      .def("isend", &Communicator::isend, nb::arg("peer_rank"),
           nb::arg("tensor"), nb::arg("offset") = 0, nb::arg("len") = 0,
           nb::arg("remote_mr_id") = 0, nb::arg("remote_offset") = 0)
      .def("isend_buffer", &Communicator::isend_buffer, nb::arg("peer_rank"),
           nb::arg("tensor"), nb::arg("buffer_id"), nb::arg("offset") = 0,
           nb::arg("len") = 0, nb::arg("remote_offset") = 0,
           nb::arg("min_generation") = 0)
      .def("irecv", &Communicator::irecv, nb::arg("peer_rank"),
           nb::arg("tensor"), nb::arg("offset") = 0, nb::arg("len") = 0)
      .def("poll", &Communicator::poll, nb::arg("req"))
      .def("release", &Communicator::release, nb::arg("req"))
      .def("wait_finish", &Communicator::wait_finish, nb::arg("req"))
      .def("wait_finish_multi", &Communicator::wait_finish_multi,
           nb::arg("reqs"))
      .def("peer_transport", &Communicator::peer_transport,
           nb::arg("peer_rank"))
      .def("same_host", &Communicator::same_host, nb::arg("peer_rank"))
      .def("send", &Communicator::send, nb::arg("peer_rank"), nb::arg("tensor"),
           nb::arg("remote_mr_id") = 0, nb::arg("remote_offset") = 0)
      .def("send_buffer", &Communicator::send_buffer, nb::arg("peer_rank"),
           nb::arg("tensor"), nb::arg("buffer_id"),
           nb::arg("remote_offset") = 0, nb::arg("min_generation") = 0)
      .def("recv", &Communicator::recv, nb::arg("peer_rank"),
           nb::arg("tensor"));
}
