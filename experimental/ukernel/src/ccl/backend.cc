#include "backend.h"
#include <stdexcept>

namespace UKernel {
namespace CCL {

namespace {

BackendToken submit_token(uint64_t& next_token, uint64_t& submissions,
                          uint32_t polls_before_ready,
                          std::unordered_map<uint64_t, uint32_t>& pending_polls) {
  BackendToken token{next_token++};
  ++submissions;
  pending_polls.emplace(token.value, polls_before_ready);
  return token;
}

bool poll_token(BackendToken token,
                std::unordered_map<uint64_t, uint32_t>& pending_polls) {
  auto it = pending_polls.find(token.value);
  if (it == pending_polls.end()) return true;
  if (it->second == 0) return true;
  --it->second;
  return it->second == 0;
}

void release_token(BackendToken token,
                   std::unordered_map<uint64_t, uint32_t>& pending_polls) {
  pending_polls.erase(token.value);
}

}  // namespace

MockBackend::MockBackend(uint32_t polls_before_ready)
    : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

char const* MockBackend::name() const { return "mock"; }

bool MockBackend::supports(ExecutionOpKind) const { return true; }

BackendToken MockBackend::submit(ExecutionOp const&) {
  return submit_token(next_token_, submissions_, polls_before_ready_,
                      pending_polls_);
}

bool MockBackend::poll(BackendToken token) {
  return poll_token(token, pending_polls_);
}

void MockBackend::release(BackendToken token) {
  release_token(token, pending_polls_);
}

PersistentKernelBackend::PersistentKernelBackend(uint32_t polls_before_ready)
    : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

char const* PersistentKernelBackend::name() const { return "persistent"; }

bool PersistentKernelBackend::supports(ExecutionOpKind kind) const {
  switch (kind) {
    case ExecutionOpKind::PkCopy:
    case ExecutionOpKind::PkReduce:
    case ExecutionOpKind::EventWait:
    case ExecutionOpKind::Barrier:
      return true;
    case ExecutionOpKind::RdmaSend:
    case ExecutionOpKind::RdmaRecv:
    case ExecutionOpKind::CeCopy:
      return false;
  }
  return false;
}

BackendToken PersistentKernelBackend::submit(ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("persistent backend does not support this op");
  }
  return submit_token(next_token_, submissions_, polls_before_ready_,
                      pending_polls_);
}

bool PersistentKernelBackend::poll(BackendToken token) {
  return poll_token(token, pending_polls_);
}

void PersistentKernelBackend::release(BackendToken token) {
  release_token(token, pending_polls_);
}

}  // namespace CCL
}  // namespace UKernel
