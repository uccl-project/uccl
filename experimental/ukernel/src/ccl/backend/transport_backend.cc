#include "transport_backend.h"
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace UKernel {
namespace CCL {

namespace {

void validate_span(char const* what, size_t offset, size_t bytes, size_t capacity) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::invalid_argument(std::string(what) + " out of range");
  }
}

}  // namespace

CommunicatorTransportBackend::CommunicatorTransportBackend(
    UKernel::Transport::Communicator& comm, CollectiveMemory memory)
    : comm_(comm), memory_(std::move(memory)) {
  completion_notifier_ = comm_.register_completion_notifier(
      [this](unsigned request_id, std::chrono::steady_clock::time_point) {
        on_transport_completion(request_id);
      });
}

char const* CommunicatorTransportBackend::name() const {
  return "communicator-transport";
}

bool CommunicatorTransportBackend::supports(ExecutionOpKind kind) const {
  return kind == ExecutionOpKind::Send || kind == ExecutionOpKind::Recv;
}

BackendToken CommunicatorTransportBackend::submit(ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument(
        "unsupported op kind for communicator transport backend");
  }
  if (op.peer_rank < 0) {
    throw std::invalid_argument("transport op requires a valid peer rank");
  }
  if (op.kind == ExecutionOpKind::Send &&
      op.src.slot == MemorySlot::SymmetricTensor &&
      op.src.rank >= 0 &&
      op.src.rank != memory_.tensor.local_rank) {
    throw std::invalid_argument("transport send must source from the local tensor view");
  }
  if (op.kind == ExecutionOpKind::Recv &&
      op.dst.slot == MemorySlot::SymmetricTensor &&
      op.dst.rank >= 0 &&
      op.dst.rank != memory_.tensor.local_rank) {
    throw std::invalid_argument("transport recv must target the local tensor view");
  }

  BackendToken token{next_token_++};
  unsigned request_id = 0;
  if (op.kind == ExecutionOpKind::Send) {
    void const* src = resolve_const(op.src, op.chunk.size_bytes);
    request_id = comm_.isend(op.peer_rank, const_cast<void*>(src), 0,
                             op.chunk.size_bytes,
                             resolve_local_mr_id(op.src, op.chunk.size_bytes),
                             resolve_remote_mr_id(op.peer_rank), true);
  } else {
    void* dst = resolve_mutable(op.dst, op.chunk.size_bytes);
    request_id = comm_.irecv(op.peer_rank, dst, 0, op.chunk.size_bytes, true);
  }

  if (request_id == 0) {
    throw std::runtime_error("communicator transport request submission failed");
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    pending_[token.value] = PendingRequest{request_id, false};
    request_to_token_[request_id] = token.value;
  }
  return token;
}

bool CommunicatorTransportBackend::poll(BackendToken token) {
  unsigned request_id = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(token.value);
    if (it == pending_.end()) return true;
    if (it->second.completed) return true;
    request_id = it->second.request_id;
  }

  if (!comm_.poll(request_id)) return false;
  on_transport_completion(request_id);
  return true;
}

bool CommunicatorTransportBackend::try_pop_completed(BackendToken& token) {
  std::lock_guard<std::mutex> lk(mu_);
  if (completed_tokens_.empty()) return false;
  token.value = completed_tokens_.front();
  completed_tokens_.pop_front();
  return true;
}

void CommunicatorTransportBackend::release(BackendToken token) {
  unsigned request_id = 0;
  bool should_release = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(token.value);
    if (it == pending_.end()) return;
    request_id = it->second.request_id;
    it->second.released = true;
    if (it->second.completed) {
      should_release = true;
      pending_.erase(it);
      request_to_token_.erase(request_id);
    }
  }
  if (should_release) {
    comm_.release(request_id);
  }
}

void* CommunicatorTransportBackend::resolve_mutable(MemoryRef const& ref,
                                                    size_t bytes) const {
  switch (ref.slot) {
    case MemorySlot::RecvStaging:
      if (memory_.recv_staging == nullptr) {
        throw std::invalid_argument("transport recv staging is missing");
      }
      validate_span("transport recv staging", ref.offset_bytes, bytes,
                    memory_.recv_staging_bytes);
      return byte_offset(memory_.recv_staging, ref.offset_bytes);
    case MemorySlot::SymmetricTensor:
      if (ref.rank == -1 || ref.rank == memory_.tensor.local_rank) {
        if (memory_.tensor.local_ptr == nullptr) {
          throw std::invalid_argument("transport local tensor is missing");
        }
        validate_span("transport local tensor", ref.offset_bytes, bytes,
                      memory_.tensor.bytes);
        return byte_offset(memory_.tensor.local_ptr, ref.offset_bytes);
      }
      break;
  }
  throw std::invalid_argument("transport backend cannot write remote tensor");
}

void const* CommunicatorTransportBackend::resolve_const(MemoryRef const& ref,
                                                        size_t bytes) const {
  switch (ref.slot) {
    case MemorySlot::RecvStaging:
      if (memory_.recv_staging == nullptr) {
        throw std::invalid_argument("transport recv staging is missing");
      }
      validate_span("transport recv staging", ref.offset_bytes, bytes,
                    memory_.recv_staging_bytes);
      return byte_offset(memory_.recv_staging, ref.offset_bytes);
    case MemorySlot::SymmetricTensor:
      if (ref.rank == -1 || ref.rank == memory_.tensor.local_rank) {
        if (memory_.tensor.local_ptr == nullptr) {
          throw std::invalid_argument("transport local tensor is missing");
        }
        validate_span("transport local tensor", ref.offset_bytes, bytes,
                      memory_.tensor.bytes);
        return byte_offset(memory_.tensor.local_ptr, ref.offset_bytes);
      }
      if (ref.rank >= 0 &&
          static_cast<size_t>(ref.rank) < memory_.tensor.peers.size()) {
        auto const& peer = memory_.tensor.peers[static_cast<size_t>(ref.rank)];
        if (peer.ptr == nullptr) {
          throw std::invalid_argument("transport peer tensor pointer unavailable");
        }
        validate_span("transport peer tensor", ref.offset_bytes, bytes,
                      memory_.tensor.bytes);
        return byte_offset(peer.ptr, ref.offset_bytes);
      }
      break;
  }
  throw std::invalid_argument("transport invalid source reference");
}

uint32_t CommunicatorTransportBackend::resolve_local_mr_id(
    MemoryRef const& ref, size_t bytes) const {
  if (ref.slot == MemorySlot::SymmetricTensor &&
      (ref.rank == -1 || ref.rank == memory_.tensor.local_rank)) {
    if (memory_.tensor.local_mr_id == 0) {
      throw std::invalid_argument("transport local tensor MR id is missing");
    }
    return memory_.tensor.local_mr_id;
  }

  void* ptr = resolve_mutable(ref, bytes);
  return comm_.get_local_mr(ptr).id;
}

uint32_t CommunicatorTransportBackend::resolve_remote_mr_id(int peer_rank) const {
  if (peer_rank < 0 ||
      static_cast<size_t>(peer_rank) >= memory_.tensor.peers.size()) {
    throw std::invalid_argument("transport peer rank out of range");
  }
  uint32_t mr_id = memory_.tensor.peers[static_cast<size_t>(peer_rank)].mr_id;
  if (mr_id == 0) {
    throw std::invalid_argument("transport remote MR id is missing");
  }
  return mr_id;
}

void CommunicatorTransportBackend::on_transport_completion(unsigned request_id) {
  bool should_release = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = request_to_token_.find(request_id);
    if (it == request_to_token_.end()) return;

    auto pending_it = pending_.find(it->second);
    if (pending_it == pending_.end() || pending_it->second.completed) return;

    pending_it->second.completed = true;
    if (pending_it->second.released) {
      should_release = true;
      pending_.erase(pending_it);
      request_to_token_.erase(it);
    } else {
      completed_tokens_.push_back(it->second);
    }
  }
  if (should_release) {
    comm_.release(request_id);
  }
}

void* CommunicatorTransportBackend::byte_offset(void* base, size_t offset) {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* CommunicatorTransportBackend::byte_offset(void const* base,
                                                      size_t offset) {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

}  // namespace CCL
}  // namespace UKernel
