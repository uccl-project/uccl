#include "mock_device_backend.h"
#include <stdexcept>

namespace UKernel {
namespace CCL {

MockDeviceBackend::MockDeviceBackend(
    void* workerPool,
    CollectiveBuffers buffers, 
    int dtype,
    int reduce_type,
    uint32_t num_blocks)
    : workerPool_(workerPool),
      buffers_(buffers),
      dtype_(dtype),
      reduce_type_(reduce_type),
      num_blocks_(num_blocks == 0 ? 1 : num_blocks) {}

char const* MockDeviceBackend::name() const {
  return "mock-device";
}

bool MockDeviceBackend::supports(ExecutionOpKind kind) const {
  switch (kind) {
    case ExecutionOpKind::PkCopy:
    case ExecutionOpKind::PkReduce:
      return true;
    case ExecutionOpKind::RdmaSend:
    case ExecutionOpKind::RdmaRecv:
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
  return token;
}

bool MockDeviceBackend::poll(BackendToken token) {
  // Simulate that all tasks are completed immediately
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

void* MockDeviceBackend::resolve_dst(BufferRole role, size_t offset) const {
  switch (role) {
    case BufferRole::FinalOutput:
      return byte_offset(buffers_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(buffers_.recv_staging, offset);
    case BufferRole::None:
    case BufferRole::LocalInput:
    case BufferRole::RemoteInput:
    case BufferRole::RemoteReduced:
      throw std::invalid_argument("invalid dst buffer role for mock-device backend");
  }
  throw std::invalid_argument("unknown dst buffer role");
}

void const* MockDeviceBackend::resolve_src(BufferRole role,
                                         size_t offset) const {
  switch (role) {
    case BufferRole::LocalInput:
      return byte_offset(buffers_.local_input, offset);
    case BufferRole::RemoteInput:
      return byte_offset(buffers_.remote_input, offset);
    case BufferRole::RemoteReduced:
      return byte_offset(buffers_.remote_reduced, offset);
    case BufferRole::RecvStaging:
      return byte_offset(buffers_.recv_staging, offset);
    case BufferRole::FinalOutput:
      return byte_offset(buffers_.final_output, offset);
    case BufferRole::None:
      throw std::invalid_argument("invalid src buffer role for mock-device backend");
  }
  throw std::invalid_argument("unknown src buffer role");
}

}  // namespace CCL
}  // namespace UKernel