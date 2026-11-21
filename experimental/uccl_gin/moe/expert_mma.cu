// SPDX-License-Identifier: Apache-2.0
// CUTLASS 4.3.4 collective builders supply WGMMA (SM90) / TMEM MMA
// (SM100-family).
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "thirdparty/nccl-ep/device/ht_scheduler.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <cuda_bf16.h>

#ifndef UCCL_EXPERT_STAGES
#define UCCL_EXPERT_STAGES 0
#endif

#define CK(call)                                                           \
  do {                                                                     \
    auto status = (call);                                                  \
    if (status != cudaSuccess) {                                           \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
      std::abort();                                                        \
    }                                                                      \
  } while (0)
#define GEMM_CHECK(call)                                                       \
  do {                                                                         \
    auto status = (call);                                                      \
    if (status != cutlass::Status::kSuccess) {                                 \
      std::fprintf(stderr, "%s: %s\n", #call, cutlassGetStatusString(status)); \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

using namespace cute;
using Element = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Column = cutlass::layout::ColumnMajor;
#if defined(UCCL_EXPERT_BLACKWELL)
using Arch = cutlass::arch::Sm100;
using Schedule = cutlass::gemm::collective::KernelScheduleAuto;
#else
using Arch = cutlass::arch::Sm90;
using Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
#endif
using MmaTile = Shape<_128, _128, _64>;
using Cluster = Shape<_1, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, Element, Row,
    8, Element, Row, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using StagePolicy = std::conditional_t<
    UCCL_EXPERT_STAGES == 0,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
        typename Epilogue::SharedStorage)>,
    cutlass::gemm::collective::StageCount<UCCL_EXPERT_STAGES>>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassTensorOp, Element, Row, 8, Element, Column, 8,
    float, MmaTile, Cluster, StagePolicy, Schedule>::CollectiveOp;
// SM100's default CLC scheduler launches every logical tile and ignores
// sm_count as a grid cap. Static persistence gives both architectures the same
// CTA budget.
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue,
    cutlass::gemm::StaticPersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

struct ExpertMma {
  void* workspace;
  static constexpr size_t workspace_bytes = 1 << 20;
  cudaDeviceProp device;
  int id, residency;
  Gemm gemm;

  ExpertMma() {
    CK(cudaGetDevice(&id));
    CK(cudaGetDeviceProperties(&device, id));
    residency = Gemm::maximum_active_blocks();
    if (residency <= 0) std::abort();
    CK(cudaMalloc(&workspace, workspace_bytes));
  }
  ~ExpertMma() { CK(cudaFree(workspace)); }

  Gemm::Arguments arguments(void const* input, void const* weight, void* output,
                            int m, int n, int k, int ctas) {
    Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {static_cast<Element const*>(input),
         cutlass::make_cute_packed_stride(Kernel::StrideA{}, {m, k, 1}),
         static_cast<Element const*>(weight),
         cutlass::make_cute_packed_stride(Kernel::StrideB{}, {n, k, 1})},
        {{1.f, 0.f},
         static_cast<Element const*>(output),
         cutlass::make_cute_packed_stride(Kernel::StrideC{}, {m, n, 1}),
         static_cast<Element*>(output),
         cutlass::make_cute_packed_stride(Kernel::StrideD{}, {m, n, 1})}};
    args.hw_info.device_id = id;
    args.hw_info.sm_count = ctas;
    return args;
  }
  void run(void const* input, void const* weight, void* output, int m, int n,
           int k, int ctas, cudaStream_t stream) {
    auto args = arguments(input, weight, output, m, n, k, ctas);
    if (Gemm::get_workspace_size(args) > workspace_bytes) std::abort();
    GEMM_CHECK(gemm.can_implement(args));
    GEMM_CHECK(gemm.initialize(args, workspace, stream));
    GEMM_CHECK(gemm.run(stream));
  }
  void query(void const* input, void const* weight, void* output, int m, int n,
             int k, int ctas, nccl_ep::hybridep::KernelResources& resource) {
    auto args = arguments(input, weight, output, m, n, k, ctas);
    cudaFuncAttributes attributes;
    CK(cudaFuncGetAttributes(&attributes, cutlass::device_kernel<Kernel>));
    auto grid = Gemm::get_grid_shape(args, workspace);
    if (int(grid.x * grid.y * grid.z) > ctas) std::abort();
    resource = {id,
                device.multiProcessorCount,
                attributes.numRegs,
                Kernel::MaxThreadsPerBlock,
                int(grid.x * grid.y * grid.z),
                residency,
                attributes.sharedSizeBytes,
                Kernel::SharedStorageSize,
                device.sharedMemPerMultiprocessor,
                device.sharedMemPerBlockOptin};
  }
};

// Two separate GEMMs preserve the model's BF16 gate/up rounding. The activation
// has the same two BF16 rounding steps as torch.silu(gate) * up.
__global__ void swiglu(__nv_bfloat16 const* gate, __nv_bfloat16 const* up,
                       __nv_bfloat16* output, size_t count) {
  for (size_t at = size_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count;
       at += size_t(gridDim.x) * blockDim.x) {
    float value = __bfloat162float(gate[at]);
    auto activated = __float2bfloat16_rn(value / (1.f + expf(-value)));
    output[at] = __float2bfloat16_rn(__bfloat162float(activated) *
                                     __bfloat162float(up[at]));
  }
}

extern "C" {
void* expert_mma_create() { return new ExpertMma; }
void expert_mma_destroy(void* handle) {
  delete static_cast<ExpertMma*>(handle);
}
void expert_mma_run(void* handle, void const* input, void const* weight,
                    void* output, int m, int n, int k, int ctas, void* stream) {
  static_cast<ExpertMma*>(handle)->run(input, weight, output, m, n, k, ctas,
                                       static_cast<cudaStream_t>(stream));
}
void expert_mma_query(void* handle, void const* input, void const* weight,
                      void* output, int m, int n, int k, int ctas,
                      nccl_ep::hybridep::KernelResources* resource) {
  static_cast<ExpertMma*>(handle)->query(input, weight, output, m, n, k, ctas,
                                         *resource);
}
void expert_mma_swiglu(void const* gate, void const* up, void* output,
                       size_t count, int ctas, void* stream) {
  swiglu<<<ctas, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<__nv_bfloat16 const*>(gate),
      static_cast<__nv_bfloat16 const*>(up),
      static_cast<__nv_bfloat16*>(output), count);
  CK(cudaGetLastError());
}
}
