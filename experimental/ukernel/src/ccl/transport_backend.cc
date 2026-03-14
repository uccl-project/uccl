#include "transport_backend.h"
#include <stdexcept>

namespace UKernel {
namespace CCL {

namespace {

template <typename RegisteredBuffer>
constexpr RegisteredBuffer kRegisteredOrder[] = {
    RegisteredBuffer::LocalInput,
    RegisteredBuffer::RemoteInput,
    RegisteredBuffer::RemoteReduced,
    RegisteredBuffer::FinalOutput,
    RegisteredBuffer::RecvStaging,
};

}  // namespace

CommunicatorTransportBackend::CommunicatorTransportBackend(
    UKernel::Transport::Communicator& comm, int peer_rank,
    CollectiveBuffers buffers)
    : comm_(comm), peer_rank_(peer_rank), buffers_(buffers) {
  if (comm_.world_size() != 2) {
    throw std::invalid_argument(
        "communicator transport backend currently supports world_size == 2 only");
  }
  ensure_registered();
}

CommunicatorTransportBackend::~CommunicatorTransportBackend() {
  deregister_registered();
}

char const* CommunicatorTransportBackend::name() const {
  return "communicator-transport";
}

bool CommunicatorTransportBackend::supports(
    ExecutionOpKind kind) const {
  return kind == ExecutionOpKind::RdmaSend ||
         kind == ExecutionOpKind::RdmaRecv;
}

BackendToken CommunicatorTransportBackend::submit(ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument(
        "unsupported op kind for communicator transport backend");
  }
  validate_topology(op);

  BackendToken token{next_token_++};
  RegisteredMr mr = resolve_mr(op.kind == ExecutionOpKind::RdmaSend
                                   ? op.src_role
                                   : op.dst_role);
  unsigned request_id = 0;
  if (op.kind == ExecutionOpKind::RdmaSend) {
    void const* src = resolve_src(op.src_role, op.chunk.offset_bytes);
    request_id = comm_.isend(peer_rank_, const_cast<void*>(src), 0,
                             op.chunk.size_bytes, mr.local_mr_id,
                             mr.remote_mr_id, true);
  } else {
    void* dst = resolve_dst(op.dst_role, op.chunk.offset_bytes);
    request_id = comm_.irecv(peer_rank_, dst, 0, op.chunk.size_bytes, true);
  }

  if (request_id == 0) {
    throw std::runtime_error("communicator transport request submission failed");
  }

  pending_[token.value] = PendingRequest{request_id, false};
  return token;
}

bool CommunicatorTransportBackend::poll(BackendToken token) {
  auto it = pending_.find(token.value);
  if (it == pending_.end()) return true;
  if (it->second.completed) return true;

  if (!comm_.poll(it->second.request_id)) return false;
  it->second.completed = true;
  return true;
}

void CommunicatorTransportBackend::release(BackendToken token) {
  auto it = pending_.find(token.value);
  if (it != pending_.end() && it->second.completed) {
    comm_.release(it->second.request_id);
  }
  pending_.erase(token.value);
}

void CommunicatorTransportBackend::validate_topology(
    ExecutionOp const& op) const {
  int expected_remote_rank =
      op.kind == ExecutionOpKind::RdmaSend ? op.dst_rank : op.src_rank;
  if (expected_remote_rank != peer_rank_) {
    throw std::invalid_argument(
        "communicator transport backend only supports a single fixed remote peer");
  }
}

void CommunicatorTransportBackend::ensure_registered() {
  if (registered_) return;
  if (buffers_.registration_bytes == 0) {
    throw std::invalid_argument(
        "communicator transport backend requires registration_bytes > 0");
  }

  register_buffer(RegisteredBuffer::LocalInput,
                  const_cast<void*>(buffers_.local_input));
  register_buffer(RegisteredBuffer::RemoteInput,
                  const_cast<void*>(buffers_.remote_input));
  register_buffer(RegisteredBuffer::RemoteReduced,
                  const_cast<void*>(buffers_.remote_reduced));
  register_buffer(RegisteredBuffer::FinalOutput, buffers_.final_output);
  register_buffer(RegisteredBuffer::RecvStaging, buffers_.recv_staging);
  exchange_mrs();
  registered_ = true;
}

void CommunicatorTransportBackend::exchange_mrs() {
  for (auto id : kRegisteredOrder<RegisteredBuffer>) {
    auto it = registered_mrs_.find(static_cast<uint64_t>(id));
    if (it == registered_mrs_.end()) continue;

    UKernel::Transport::MR local =
        comm_.get_local_mr(it->second.local_mr_id);
    if (!comm_.notify_mr(peer_rank_, local)) {
      throw std::runtime_error("communicator transport notify_mr failed");
    }
  }

  for (auto id : kRegisteredOrder<RegisteredBuffer>) {
    auto it = registered_mrs_.find(static_cast<uint64_t>(id));
    if (it == registered_mrs_.end()) continue;

    UKernel::Transport::MR remote{};
    if (!comm_.wait_mr_notify(peer_rank_, remote)) {
      throw std::runtime_error("communicator transport wait_mr_notify failed");
    }
    it->second.remote_mr_id = remote.id;
  }
}

void CommunicatorTransportBackend::register_buffer(RegisteredBuffer id, void* ptr) {
  if (ptr == nullptr) return;
  UKernel::Transport::MR mr = comm_.reg_mr(ptr, buffers_.registration_bytes);
  registered_mrs_[static_cast<uint64_t>(id)] =
      RegisteredMr{mr.id, 0};
}

void CommunicatorTransportBackend::deregister_registered() {
  if (!registered_) return;
  if (buffers_.local_input != nullptr)
    comm_.dereg_mr(const_cast<void*>(buffers_.local_input));
  if (buffers_.remote_input != nullptr)
    comm_.dereg_mr(const_cast<void*>(buffers_.remote_input));
  if (buffers_.remote_reduced != nullptr)
    comm_.dereg_mr(const_cast<void*>(buffers_.remote_reduced));
  if (buffers_.final_output != nullptr) comm_.dereg_mr(buffers_.final_output);
  if (buffers_.recv_staging != nullptr) comm_.dereg_mr(buffers_.recv_staging);
  registered_ = false;
}

void* CommunicatorTransportBackend::resolve_dst(BufferRole role,
                                                size_t offset) const {
  switch (role) {
    case BufferRole::FinalOutput:
      return byte_offset(buffers_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(buffers_.recv_staging, offset);
    case BufferRole::None:
    case BufferRole::LocalInput:
    case BufferRole::RemoteInput:
    case BufferRole::RemoteReduced:
      break;
  }
  throw std::invalid_argument("invalid communicator transport dst role");
}

void const* CommunicatorTransportBackend::resolve_src(
    BufferRole role, size_t offset) const {
  switch (role) {
    case BufferRole::LocalInput:
      return byte_offset(buffers_.local_input, offset);
    case BufferRole::RemoteInput:
      return byte_offset(buffers_.remote_input, offset);
    case BufferRole::RemoteReduced:
      return byte_offset(buffers_.remote_reduced, offset);
    case BufferRole::FinalOutput:
      return byte_offset(buffers_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(buffers_.recv_staging, offset);
    case BufferRole::None:
      break;
  }
  throw std::invalid_argument("invalid communicator transport src role");
}

CommunicatorTransportBackend::RegisteredMr
CommunicatorTransportBackend::resolve_mr(BufferRole role) const {
  RegisteredBuffer id;
  switch (role) {
    case BufferRole::LocalInput:
      id = RegisteredBuffer::LocalInput;
      break;
    case BufferRole::RemoteInput:
      id = RegisteredBuffer::RemoteInput;
      break;
    case BufferRole::RemoteReduced:
      id = RegisteredBuffer::RemoteReduced;
      break;
    case BufferRole::FinalOutput:
      id = RegisteredBuffer::FinalOutput;
      break;
    case BufferRole::RecvStaging:
      id = RegisteredBuffer::RecvStaging;
      break;
    case BufferRole::None:
      throw std::invalid_argument("invalid communicator transport mr role");
  }
  auto it = registered_mrs_.find(static_cast<uint64_t>(id));
  if (it == registered_mrs_.end()) {
    throw std::invalid_argument(
        "communicator transport mr not registered for role");
  }
  return it->second;
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
