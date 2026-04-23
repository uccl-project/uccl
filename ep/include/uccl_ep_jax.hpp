// ============================================================================
// JAX FFI bridge for UCCL-EP -- public interface
// ----------------------------------------------------------------------------
// Every JAX-specific C++ component of the ``uccl.ep`` module (eight XLA
// custom-call handlers, per-device ``Buffer*`` registry, nanobind
// bindings for ``register_jax_ffi_buffer`` / ``unregister_jax_ffi_buffer``
// / ``get_jax_ffi_targets``) lives in ``ep/src/uccl_ep_jax.cc``.
//
// This header exists only to let ``uccl_ep.cc`` hand the nanobind
// module to the JAX TU at module-init time.
// ============================================================================

#pragma once

#include <nanobind/nanobind.h>

// NOTE (typed-FFI migration):
// Handlers no longer take ``XlaCustomCallStatus*``. They are now built
// with the XLA Typed FFI (``XLA_FFI_DEFINE_HANDLER_SYMBOL``) declared
// in ``xla/ffi/api/ffi.h`` (shipped with jaxlib). This header only
// needs the nanobind forward declaration used by the one hook below.

namespace uccl_jax_ffi {

namespace nb = nanobind;

// Attach ``register_jax_ffi_buffer`` / ``unregister_jax_ffi_buffer`` /
// ``get_jax_ffi_targets`` to the ``uccl.ep`` module. Called from
// ``NB_MODULE(ep, m)`` in ``uccl_ep.cc`` exactly once.
void register_jax_bindings(nb::module_& m);

}  // namespace uccl_jax_ffi
