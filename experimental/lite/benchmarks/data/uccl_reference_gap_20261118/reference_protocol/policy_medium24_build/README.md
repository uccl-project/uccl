# Explicit medium24 policy build verification

UCCL_LITE_NCCL_POLICY=medium24 reproduces the measured CUDA and link command
arguments. The default remains base; this is not zero-regression admission.
The backend, shared library, standalone component check, network size/queue
executables and234-call ABI executable build successfully. An unchanged policy
does not rebuild. Switching the policy stamp to base then back to medium24
invalidates and rebuilds the backend correctly. The rebuilt library passes234
mixed ABI calls/rank on the two four-L4 hosts, with full outputs and native absence.

Raw library hashes differ because NVCC records a temporary filename in the
non-allocated symbol table and the ELF build ID changes. The independent identity
includes every allocated section and its layout, all program headers, dynamic
symbols/relocations, code, constants and CUDA fatbins. Only the GNU build-ID
descriptor is masked; non-allocated symbols/debug metadata are excluded. Both
library and component identities match the measured artifacts. Self-checks detect
code and dynamic-string changes while ignoring only the specified metadata.
Raw hashes and complete identity records are retained. This comparison concerns
build provenance; the measured RS16MiB regression remains a failure.
