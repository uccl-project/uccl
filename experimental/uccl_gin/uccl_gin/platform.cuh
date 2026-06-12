#pragma once
//
// Platform shims so the UCCL-GIN device headers compile on both NVIDIA (CUDA)
// and AMD (HIP/ROCm). Keep this header tiny and dependency-free; it is included
// by resources.cuh and uccl_gin_rail.cuh before any device code.
//
// NOTE: this header deliberately does NOT pull in <nccl_device.h>. resources.cuh
// is reached from host translation units (context.cpp via context.hpp), so the
// NCCL device header must stay out of this path. Team-tag types live in
// uccl_gin.cuh, the only header that actually needs them.

// ---- device trap ----------------------------------------------------------
// __trap() is a CUDA-only PTX intrinsic; HIP device code uses __builtin_trap().
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define UCCL_GIN_TRAP() __builtin_trap()
#else
#define UCCL_GIN_TRAP() __trap()
#endif

// ---- NCCL-GIN reference path toggle ---------------------------------------
// The build system defines this (1 on NVIDIA, 0 on AMD). Default to 1 so a
// stray NVIDIA include that forgets the flag still gets the real NCCL tags.
#ifndef UCCL_GIN_WITH_NCCL_GIN
#define UCCL_GIN_WITH_NCCL_GIN 1
#endif

// True when the NCCL device API (and its team-tag types) is available.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || !UCCL_GIN_WITH_NCCL_GIN
#define UCCL_GIN_HAVE_NCCL_DEVICE 0
#else
#define UCCL_GIN_HAVE_NCCL_DEVICE 1
#endif
