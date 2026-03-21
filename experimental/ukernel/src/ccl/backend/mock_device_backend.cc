#include "mock_device_backend.h"
#include <stdexcept>

namespace UKernel {
namespace CCL {

MockDeviceBackend::MockDeviceBackend(
    void* workerPool,
    CollectiveMemory memory,
    int dtype,
    int reduce_type,
    uint32_t num_blocks)
    : workerPool_(workerPool),
      memory_(std::move(memory)),
      dtype_(dtype),
      reduce_type_(reduce_type),
      num_blocks_(num_blocks == 0 ? 1 : num_blocks) {}

char const* MockDeviceBackend::name() const {
  return "mock-device";
}

bool MockDeviceBackend::supports(ExecutionOpKind kind) const {
  switch (kind) {
    case ExecutionOpKind::Copy:
    case ExecutionOpKind::Reduce:
      return true;
    case ExecutionOpKind::Send:
    case ExecutionOpKind::Recv:
    case ExecutionOpKind::EventWait:
    case ExecutionOpKind::Barrier:
      return false;
  }
  return false;
}

BackendToken MockDeviceBackend::submit(ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for mock-device backend");
  }

  // Simulate task submission by just marking it as completed immediately
  BackendToken token{next_token_++};
  submitted_[token.value] = {0, 0}; // Mock values
  completed_tokens_.push_back(token.value);
  return token;
}

bool MockDeviceBackend::poll(BackendToken token) {
  // Simulate that all tasks are completed immediately
  return true;
}

bool MockDeviceBackend::try_pop_completed(BackendToken& token) {
  if (completed_tokens_.empty()) return false;
  token.value = completed_tokens_.front();
  completed_tokens_.pop_front();
  return true;
}

void MockDeviceBackend::release(BackendToken token) {
  submitted_.erase(token.value);
}

void* MockDeviceBackend::byte_offset(void* base, size_t offset) const {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* MockDeviceBackend::byte_offset(void const* base,
                                         size_t offset) const {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

void* MockDeviceBackend::resolve_mutable(MemoryRef const& ref) const {
  switch (ref.slot) {
    case MemorySlot::RecvStaging:
      return byte_offset(memory_.recv_staging, ref.offset_bytes);
    case MemorySlot::SymmetricTensor:
      if (ref.rank == -1 || ref.rank == memory_.tensor.local_rank) {
        return byte_offset(memory_.tensor.local_ptr, ref.offset_bytes);
      }
      break;
  }
  throw std::invalid_argument("invalid dst ref for mock-device backend");
}

void const* MockDeviceBackend::resolve_const(MemoryRef const& ref) const {
  switch (ref.slot) {
    case MemorySlot::RecvStaging:
      return byte_offset(memory_.recv_staging, ref.offset_bytes);
    case MemorySlot::SymmetricTensor:
      if (ref.rank == -1 || ref.rank == memory_.tensor.local_rank) {
        return byte_offset(memory_.tensor.local_ptr, ref.offset_bytes);
      }
      if (ref.rank >= 0 &&
          static_cast<size_t>(ref.rank) < memory_.tensor.peers.size()) {
        return byte_offset(memory_.tensor.peers[static_cast<size_t>(ref.rank)].ptr,
                           ref.offset_bytes);
      }
      break;
  }
  throw std::invalid_argument("invalid src ref for mock-device backend");
}

}  // namespace CCL
}  // namespace UKernel
