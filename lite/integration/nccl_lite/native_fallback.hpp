#pragma once
#include "uccl_lite/runtime/collective_backend.hpp"
#include "uccl_lite/runtime/prepared.hpp"
#include <nccl.h>

namespace uccl_lite::integration {
// Read the participant from the actual native communicator, without creating
// transport resources. The ID is a separate UCCL bootstrap session.
runtime::RuntimeContext nativeNcclContext(ncclComm_t communicator,
                                          std::string const& nativeLibrary,
                                          NetworkId const& id, size_t capacity);
// Borrow an initialized blocking native NCCL communicator from the caller. The
// caller keeps it alive and supplies a common identity for its
// initialization/config. No communicator or router is created/destroyed here.
// The explicit native library path avoids resolving the strict UCCL ABI as the
// fallback.
runtime::CallerFallback makeNativeNcclFallback(
    runtime::BackendContext const& context, ncclComm_t communicator,
    std::string const& nativeLibrary, std::string const& configurationIdentity,
    ScalarType dtype = ScalarType::BFloat16);
runtime::CallerFallback makeNativeNcclFallback(
    runtime::RuntimeContext const& context, ncclComm_t communicator,
    std::string const& nativeLibrary, std::string const& configurationIdentity,
    ScalarType dtype = ScalarType::BFloat16);
}  // namespace uccl_lite::integration
