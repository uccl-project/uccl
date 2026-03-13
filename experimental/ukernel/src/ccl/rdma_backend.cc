#include "rdma_backend.h"
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

RdmaBackend::RdmaBackend(UKernel::Transport::UcclTransportAdapter& adapter,
                         BufferBindings bindings)
    : adapter_(adapter), bindings_(bindings) {
  ensure_registered();
}

RdmaBackend::~RdmaBackend() { deregister_registered(); }

char const* RdmaBackend::name() const { return "rdma"; }

bool RdmaBackend::supports(ExecutionOpKind kind) const {
  return kind == ExecutionOpKind::RdmaSend || kind == ExecutionOpKind::RdmaRecv;
}

BackendToken RdmaBackend::submit(ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for rdma backend");
  }

  BackendToken token{next_request_id_++};
  uint64_t request_id = token.value;

  if (op.kind == ExecutionOpKind::RdmaSend) {
    void const* src = resolve_src(op.src_role, op.chunk.offset_bytes);
    int rc = adapter_.send_async(op.dst_rank, const_cast<void*>(src),
                                 op.chunk.size_bytes,
                                 resolve_mr_id_for_src(op.src_role), 0,
                                 request_id);
    if (rc != 0) {
      throw std::runtime_error("rdma send_async failed");
    }
  } else {
    void* dst = resolve_dst(op.dst_role, op.chunk.offset_bytes);
    int rc = adapter_.recv_async(op.src_rank, dst, op.chunk.size_bytes,
                                 resolve_mr_id_for_dst(op.dst_role),
                                 request_id);
    if (rc != 0) {
      throw std::runtime_error("rdma recv_async failed");
    }
  }

  pending_[token.value] = PendingRequest{request_id, false};
  return token;
}

bool RdmaBackend::poll(BackendToken token) {
  auto it = pending_.find(token.value);
  if (it == pending_.end()) return true;
  if (it->second.completed) return true;

  // Current adapter exposes only a blocking completion wait. That is enough
  // for the first end-to-end RDMA path while we keep executor semantics
  // uniform.
  if (!adapter_.wait_completion(it->second.request_id)) {
    throw std::runtime_error("rdma wait_completion failed");
  }
  it->second.completed = true;
  return true;
}

void RdmaBackend::release(BackendToken token) { pending_.erase(token.value); }

void RdmaBackend::ensure_registered() {
  if (registered_) return;
  if (bindings_.registration_bytes == 0) {
    throw std::invalid_argument("rdma backend requires registration_bytes > 0");
  }

  register_buffer(RegisteredBuffer::LocalInput,
                  const_cast<void*>(bindings_.local_input));
  register_buffer(RegisteredBuffer::RemoteInput,
                  const_cast<void*>(bindings_.remote_input));
  register_buffer(RegisteredBuffer::RemoteReduced,
                  const_cast<void*>(bindings_.remote_reduced));
  register_buffer(RegisteredBuffer::FinalOutput, bindings_.final_output);
  register_buffer(RegisteredBuffer::RecvStaging, bindings_.recv_staging);
  registered_ = true;
}

void RdmaBackend::register_buffer(RegisteredBuffer id, void* ptr) {
  if (ptr == nullptr) return;
  if (!adapter_.register_memory(static_cast<uint64_t>(id), ptr,
                                bindings_.registration_bytes)) {
    throw std::runtime_error("rdma register_memory failed");
  }
}

void RdmaBackend::deregister_registered() {
  if (!registered_) return;
  adapter_.deregister_memory(static_cast<uint64_t>(RegisteredBuffer::LocalInput));
  adapter_.deregister_memory(static_cast<uint64_t>(RegisteredBuffer::RemoteInput));
  adapter_.deregister_memory(static_cast<uint64_t>(RegisteredBuffer::RemoteReduced));
  adapter_.deregister_memory(static_cast<uint64_t>(RegisteredBuffer::FinalOutput));
  adapter_.deregister_memory(static_cast<uint64_t>(RegisteredBuffer::RecvStaging));
  registered_ = false;
}

void* RdmaBackend::resolve_dst(BufferRole role, size_t offset) const {
  switch (role) {
    case BufferRole::FinalOutput:
      return byte_offset(bindings_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(bindings_.recv_staging, offset);
    case BufferRole::None:
    case BufferRole::LocalInput:
    case BufferRole::RemoteInput:
    case BufferRole::RemoteReduced:
      throw std::invalid_argument("invalid rdma dst role");
  }
  throw std::invalid_argument("unknown rdma dst role");
}

void const* RdmaBackend::resolve_src(BufferRole role, size_t offset) const {
  switch (role) {
    case BufferRole::LocalInput:
      return byte_offset(bindings_.local_input, offset);
    case BufferRole::RemoteInput:
      return byte_offset(bindings_.remote_input, offset);
    case BufferRole::RemoteReduced:
      return byte_offset(bindings_.remote_reduced, offset);
    case BufferRole::FinalOutput:
      return byte_offset(bindings_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(bindings_.recv_staging, offset);
    case BufferRole::None:
      throw std::invalid_argument("invalid rdma src role");
  }
  throw std::invalid_argument("unknown rdma src role");
}

uint64_t RdmaBackend::resolve_mr_id_for_src(BufferRole role) const {
  switch (role) {
    case BufferRole::LocalInput:
      return static_cast<uint64_t>(RegisteredBuffer::LocalInput);
    case BufferRole::RemoteInput:
      return static_cast<uint64_t>(RegisteredBuffer::RemoteInput);
    case BufferRole::RemoteReduced:
      return static_cast<uint64_t>(RegisteredBuffer::RemoteReduced);
    case BufferRole::FinalOutput:
      return static_cast<uint64_t>(RegisteredBuffer::FinalOutput);
    case BufferRole::RecvStaging:
      return static_cast<uint64_t>(RegisteredBuffer::RecvStaging);
    case BufferRole::None:
      break;
  }
  throw std::invalid_argument("invalid rdma src mr role");
}

uint64_t RdmaBackend::resolve_mr_id_for_dst(BufferRole role) const {
  switch (role) {
    case BufferRole::FinalOutput:
      return static_cast<uint64_t>(RegisteredBuffer::FinalOutput);
    case BufferRole::RecvStaging:
      return static_cast<uint64_t>(RegisteredBuffer::RecvStaging);
    case BufferRole::None:
    case BufferRole::LocalInput:
    case BufferRole::RemoteInput:
    case BufferRole::RemoteReduced:
      break;
  }
  throw std::invalid_argument("invalid rdma dst mr role");
}

void* RdmaBackend::byte_offset(void* base, size_t offset) {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* RdmaBackend::byte_offset(void const* base, size_t offset) {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

CommunicatorRdmaBackend::CommunicatorRdmaBackend(
    UKernel::Transport::Communicator& comm, int peer_rank, BufferBindings bindings)
    : comm_(comm), peer_rank_(peer_rank), bindings_(bindings) {
  ensure_registered();
}

CommunicatorRdmaBackend::~CommunicatorRdmaBackend() { deregister_registered(); }

char const* CommunicatorRdmaBackend::name() const { return "communicator-rdma"; }

bool CommunicatorRdmaBackend::supports(ExecutionOpKind kind) const {
  return kind == ExecutionOpKind::RdmaSend || kind == ExecutionOpKind::RdmaRecv;
}

BackendToken CommunicatorRdmaBackend::submit(ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for communicator rdma backend");
  }

  BackendToken token{next_token_++};
  RegisteredMr mr = resolve_mr(op.kind == ExecutionOpKind::RdmaSend ? op.src_role
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
    throw std::runtime_error("communicator rdma request submission failed");
  }

  pending_[token.value] = PendingRequest{request_id, false};
  return token;
}

bool CommunicatorRdmaBackend::poll(BackendToken token) {
  auto it = pending_.find(token.value);
  if (it == pending_.end()) return true;
  if (it->second.completed) return true;

  if (!comm_.wait_finish(it->second.request_id)) {
    throw std::runtime_error("communicator rdma wait_finish failed");
  }
  it->second.completed = true;
  return true;
}

void CommunicatorRdmaBackend::release(BackendToken token) {
  pending_.erase(token.value);
}

void CommunicatorRdmaBackend::ensure_registered() {
  if (registered_) return;
  if (bindings_.registration_bytes == 0) {
    throw std::invalid_argument(
        "communicator rdma backend requires registration_bytes > 0");
  }

  register_buffer(RegisteredBuffer::LocalInput,
                  const_cast<void*>(bindings_.local_input));
  register_buffer(RegisteredBuffer::RemoteInput,
                  const_cast<void*>(bindings_.remote_input));
  register_buffer(RegisteredBuffer::RemoteReduced,
                  const_cast<void*>(bindings_.remote_reduced));
  register_buffer(RegisteredBuffer::FinalOutput, bindings_.final_output);
  register_buffer(RegisteredBuffer::RecvStaging, bindings_.recv_staging);
  exchange_mrs();
  registered_ = true;
}

void CommunicatorRdmaBackend::exchange_mrs() {
  for (auto id : kRegisteredOrder<RegisteredBuffer>) {
    auto it = registered_mrs_.find(static_cast<uint64_t>(id));
    if (it == registered_mrs_.end()) continue;

    UKernel::Transport::MR local =
        comm_.get_local_mr(static_cast<uint16_t>(it->second.local_mr_id));
    if (!comm_.notify_mr(peer_rank_, local)) {
      throw std::runtime_error("communicator rdma notify_mr failed");
    }
  }

  for (auto id : kRegisteredOrder<RegisteredBuffer>) {
    auto it = registered_mrs_.find(static_cast<uint64_t>(id));
    if (it == registered_mrs_.end()) continue;

    UKernel::Transport::MR remote{};
    if (!comm_.wait_mr_notify(peer_rank_, remote)) {
      throw std::runtime_error("communicator rdma wait_mr_notify failed");
    }
    it->second.remote_mr_id = static_cast<uint16_t>(remote.id);
  }
}

void CommunicatorRdmaBackend::register_buffer(RegisteredBuffer id, void* ptr) {
  if (ptr == nullptr) return;
  UKernel::Transport::MR mr = comm_.reg_mr(ptr, bindings_.registration_bytes);
  registered_mrs_[static_cast<uint64_t>(id)] =
      RegisteredMr{static_cast<uint16_t>(mr.id), 0};
}

void CommunicatorRdmaBackend::deregister_registered() {
  if (!registered_) return;
  if (bindings_.local_input != nullptr)
    comm_.dereg_mr(const_cast<void*>(bindings_.local_input));
  if (bindings_.remote_input != nullptr)
    comm_.dereg_mr(const_cast<void*>(bindings_.remote_input));
  if (bindings_.remote_reduced != nullptr)
    comm_.dereg_mr(const_cast<void*>(bindings_.remote_reduced));
  if (bindings_.final_output != nullptr) comm_.dereg_mr(bindings_.final_output);
  if (bindings_.recv_staging != nullptr)
    comm_.dereg_mr(bindings_.recv_staging);
  registered_ = false;
}

void* CommunicatorRdmaBackend::resolve_dst(BufferRole role, size_t offset) const {
  switch (role) {
    case BufferRole::FinalOutput:
      return byte_offset(bindings_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(bindings_.recv_staging, offset);
    case BufferRole::None:
    case BufferRole::LocalInput:
    case BufferRole::RemoteInput:
    case BufferRole::RemoteReduced:
      break;
  }
  throw std::invalid_argument("invalid communicator rdma dst role");
}

void const* CommunicatorRdmaBackend::resolve_src(BufferRole role,
                                                 size_t offset) const {
  switch (role) {
    case BufferRole::LocalInput:
      return byte_offset(bindings_.local_input, offset);
    case BufferRole::RemoteInput:
      return byte_offset(bindings_.remote_input, offset);
    case BufferRole::RemoteReduced:
      return byte_offset(bindings_.remote_reduced, offset);
    case BufferRole::FinalOutput:
      return byte_offset(bindings_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(bindings_.recv_staging, offset);
    case BufferRole::None:
      break;
  }
  throw std::invalid_argument("invalid communicator rdma src role");
}

CommunicatorRdmaBackend::RegisteredMr CommunicatorRdmaBackend::resolve_mr(
    BufferRole role) const {
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
      throw std::invalid_argument("invalid communicator rdma mr role");
  }
  auto it = registered_mrs_.find(static_cast<uint64_t>(id));
  if (it == registered_mrs_.end()) {
    throw std::invalid_argument("communicator rdma mr not registered for role");
  }
  return it->second;
}

void* CommunicatorRdmaBackend::byte_offset(void* base, size_t offset) {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* CommunicatorRdmaBackend::byte_offset(void const* base,
                                                 size_t offset) {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

}  // namespace CCL
}  // namespace UKernel
