// Per-dtype reduce dispatch instantiations. Compiled as its own TU with
// relocatable device code so the persistent-kernel TU calls these across
// TUs instead of inlining them (which spilled the 255-register budget).
#include "ops/reduce_dispatch.h"

namespace UKernel {
namespace Device {

#define UK_DEFINE_REDUCE_DISPATCH(T, suffix)                                \
  __device__ void dispatch_reduce_##suffix(TaskArgs const& args,            \
                                           uint32_t block_id,               \
                                           uint32_t num_blocks,             \
                                           void* smem_buf) {                \
    run_reduce<T>(args, block_id, num_blocks, smem_buf);                    \
  }
UK_DEFINE_REDUCE_DISPATCH(float, fp32)
UK_DEFINE_REDUCE_DISPATCH(__half, fp16)
UK_DEFINE_REDUCE_DISPATCH(nv_bfloat16, bf16)
UK_DEFINE_REDUCE_DISPATCH(int32_t, int32)
UK_DEFINE_REDUCE_DISPATCH(int8_t, int8)
UK_DEFINE_REDUCE_DISPATCH(int64_t, int64)
UK_DEFINE_REDUCE_DISPATCH(double, fp64)
#undef UK_DEFINE_REDUCE_DISPATCH

}  // namespace Device
}  // namespace UKernel
