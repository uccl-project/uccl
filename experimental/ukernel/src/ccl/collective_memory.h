#pragma once

#include "collective_types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace UKernel {
namespace CCL {

using BufferId = uint32_t;

inline constexpr BufferId kInvalidBufferId = UINT32_MAX;
// Buffer id 0 is reserved as "invalid/unset" in transport resource APIs.
// Use non-zero defaults for CCL role buffers so default collectives can
// register MR/IPC metadata without requiring user-specified ids.
inline constexpr BufferId kDefaultInputBufferId = 1;
inline constexpr BufferId kDefaultOutputBufferId = kDefaultInputBufferId;
inline constexpr BufferId kDefaultScratchBufferId = 2;

struct TensorLayout {
  // Torch-style tensor metadata: sizes/strides/storage_offset are all
  // expressed in elements rather than bytes.
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t storage_offset = 0;
  ScalarType dtype = ScalarType::UInt8;
};

enum class BufferKind : uint32_t {
  Local,
  Remote,
};

// BufferRef is the planner/runtime address token. It references a registered
// buffer by id and a byte offset inside that buffer. Remote refs additionally
// name the peer rank that owns the referenced buffer instance.
struct BufferRef {
  BufferKind kind = BufferKind::Local;
  BufferId buffer_id = kInvalidBufferId;
  size_t offset_bytes = 0;
  int rank = -1;
};

inline BufferRef local_buffer_ref(BufferId buffer_id, size_t offset_bytes = 0) {
  return BufferRef{BufferKind::Local, buffer_id, offset_bytes, -1};
}

inline BufferRef remote_buffer_ref(BufferId buffer_id, int rank,
                                   size_t offset_bytes = 0) {
  return BufferRef{BufferKind::Remote, buffer_id, offset_bytes, rank};
}

struct PeerBufferView {
  uint32_t buffer_id = 0;
  bool same_node = false;
};

struct RegisteredBuffer {
  bool registered = false;
  void* local_ptr = nullptr;
  uint32_t local_buffer_id = 0;
  size_t bytes = 0;
  TensorLayout layout{};
  bool remotely_accessible = true;
  // Rank-indexed remote view table for the same logical buffer id.
  std::vector<PeerBufferView> peer_views;
};

struct BufferRegistry {
  int local_rank = 0;
  std::vector<RegisteredBuffer> buffers;

  RegisteredBuffer& ensure_buffer(BufferId id) {
    if (id == kInvalidBufferId) {
      throw std::invalid_argument("cannot materialize invalid buffer id");
    }
    size_t index = static_cast<size_t>(id);
    if (index >= buffers.size()) {
      buffers.resize(index + 1);
    }
    buffers[index].registered = true;
    return buffers[index];
  }

  bool has_buffer(BufferId id) const {
    return id != kInvalidBufferId && static_cast<size_t>(id) < buffers.size() &&
           buffers[static_cast<size_t>(id)].registered;
  }

  RegisteredBuffer& buffer(BufferId id) {
    if (!has_buffer(id)) {
      throw std::invalid_argument("buffer id is not registered");
    }
    return buffers[static_cast<size_t>(id)];
  }

  RegisteredBuffer const& buffer(BufferId id) const {
    if (!has_buffer(id)) {
      throw std::invalid_argument("buffer id is not registered");
    }
    return buffers[static_cast<size_t>(id)];
  }

  std::vector<BufferId> registered_buffer_ids() const {
    std::vector<BufferId> ids;
    ids.reserve(buffers.size());
    for (size_t index = 0; index < buffers.size(); ++index) {
      if (!buffers[index].registered) continue;
      ids.push_back(static_cast<BufferId>(index));
    }
    return ids;
  }
};

enum class CollectiveBufferRole : uint32_t {
  Input,
  Output,
  Scratch,
};

// Role mapping keeps planner/runtime independent from concrete buffer ids.
// Input is the logical source, Output is the logical destination
// (equal to Input for in-place collectives), and Scratch is temporary
// workspace used by algorithms (for example ring/pairwise staging).
struct CollectiveBufferRoles {
  BufferId input_buffer_id = kDefaultInputBufferId;
  BufferId output_buffer_id = kDefaultOutputBufferId;
  BufferId scratch_buffer_id = kDefaultScratchBufferId;

  BufferId buffer_id(CollectiveBufferRole role) const {
    switch (role) {
      case CollectiveBufferRole::Input:
        return input_buffer_id;
      case CollectiveBufferRole::Output:
        return output_buffer_id;
      case CollectiveBufferRole::Scratch:
        return scratch_buffer_id;
    }
    return kInvalidBufferId;
  }

  void validate() const {
    if (input_buffer_id == kInvalidBufferId) {
      throw std::invalid_argument(
          "collective binding requires a valid input buffer id");
    }
    if (output_buffer_id == kInvalidBufferId) {
      throw std::invalid_argument(
          "collective binding requires a valid output buffer id");
    }
    if (scratch_buffer_id == kInvalidBufferId) {
      throw std::invalid_argument(
          "collective binding requires a valid scratch buffer id");
    }
    if (input_buffer_id == scratch_buffer_id ||
        output_buffer_id == scratch_buffer_id) {
      throw std::invalid_argument(
          "collective binding scratch buffer must be distinct from "
          "input/output");
    }
  }
};

struct CollectiveBinding {
  std::shared_ptr<BufferRegistry> registry;
  CollectiveBufferRoles roles{};
  uint64_t transport_initialized_backend_key = 0;
  uint64_t transport_initialized_signature = 0;

  int local_rank() const {
    return registry != nullptr ? registry->local_rank : 0;
  }

  BufferId buffer_id(CollectiveBufferRole role) const {
    return roles.buffer_id(role);
  }

  RegisteredBuffer& ensure_buffer(BufferId id) {
    if (registry == nullptr) {
      registry = std::make_shared<BufferRegistry>();
    }
    return registry->ensure_buffer(id);
  }

  bool has_buffer(BufferId id) const {
    return registry != nullptr && registry->has_buffer(id);
  }

  RegisteredBuffer& buffer(BufferId id) {
    if (registry == nullptr) {
      throw std::invalid_argument("collective binding has no buffer registry");
    }
    return registry->buffer(id);
  }

  RegisteredBuffer const& buffer(BufferId id) const {
    if (registry == nullptr) {
      throw std::invalid_argument("collective binding has no buffer registry");
    }
    return registry->buffer(id);
  }

  RegisteredBuffer& role_buffer(CollectiveBufferRole role) {
    return buffer(buffer_id(role));
  }

  RegisteredBuffer const& role_buffer(CollectiveBufferRole role) const {
    return buffer(buffer_id(role));
  }

  void invalidate_transport_cache() {
    transport_initialized_backend_key = 0;
    transport_initialized_signature = 0;
  }

  // Signature used by transport backend cache invalidation.
  // Any bind-relevant mutation (roles, ptr/bytes/accessibility) changes this
  // value and forces MR/IPC metadata re-initialization for correctness.
  uint64_t transport_signature() const {
    auto hash_combine = [](uint64_t seed, uint64_t value) {
      return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    };

    uint64_t signature = 0;
    signature =
        hash_combine(signature, static_cast<uint64_t>(roles.input_buffer_id));
    signature =
        hash_combine(signature, static_cast<uint64_t>(roles.output_buffer_id));
    signature =
        hash_combine(signature, static_cast<uint64_t>(roles.scratch_buffer_id));
    if (registry == nullptr) return signature;
    for (BufferId id : registry->registered_buffer_ids()) {
      RegisteredBuffer const& entry = registry->buffer(id);
      signature = hash_combine(signature, static_cast<uint64_t>(id));
      signature = hash_combine(signature, entry.registered ? 1ULL : 0ULL);
      signature = hash_combine(
          signature,
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(entry.local_ptr)));
      signature = hash_combine(signature, static_cast<uint64_t>(entry.bytes));
      signature =
          hash_combine(signature, entry.remotely_accessible ? 1ULL : 0ULL);
    }
    return signature;
  }
};

}  // namespace CCL
}  // namespace UKernel
