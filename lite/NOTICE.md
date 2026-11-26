# Third-party notices

UCCL Lite uses MSCCL++, CUDA, NCCL, and CUTLASS as external dependencies. Their
API names and licenses remain those of their respective upstream projects.

`include/uccl_lite/cutlass_gated_grouped.h` adapts the operator body from
CUTLASS 4.5.0's `include/cutlass/gemm/kernel/gemm_grouped.h`, adding arrival
gating, an optional dispatcher, and output-readiness hooks. The upstream
copyright notice and BSD-3-Clause terms are reproduced in that header.

`include/uccl_lite/cpu_router/cutlass_gated_grouped.cuh` is the CPU-router
branch's resident-GEMM adaptation of the same upstream grouped kernel. Its
upstream copyright and BSD-3-Clause notice are preserved in the file.
