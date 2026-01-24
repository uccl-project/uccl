#include <cstdio>
#include <cstdint>
#include <cstring>
#include <type_traits>

/* ================= GPU compatibility layer ================= */

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIPCC__)

  #include <hip/hip_runtime.h>
  using GpuStream = hipStream_t;

  #define GPU_CHECK(x) do {                                  \
    hipError_t err = (x);                                    \
    if (err != hipSuccess) {                                 \
      printf("HIP error %s at %s:%d\n",                      \
             hipGetErrorString(err), __FILE__, __LINE__);    \
      std::abort();                                          \
    }                                                        \
  } while (0)

  #define GPU_MEMCPY_ASYNC     hipMemcpyAsync
  #define GPU_STREAM_SYNC      hipStreamSynchronize
  #define GPU_MEMCPY_H2D       hipMemcpyHostToDevice
  #define GPU_STREAM_CREATE    hipStreamCreate
  #define GPU_STREAM_DESTROY   hipStreamDestroy
  #define GPU_MALLOC           hipMalloc
  #define GPU_FREE             hipFree

#else

  #include <cuda_runtime.h>
  using GpuStream = cudaStream_t;

  #define GPU_CHECK(x) do {                                  \
    cudaError_t err = (x);                                   \
    if (err != cudaSuccess) {                                \
      printf("CUDA error %s at %s:%d\n",                     \
             cudaGetErrorString(err), __FILE__, __LINE__);   \
      std::abort();                                          \
    }                                                        \
  } while (0)

  #define GPU_MEMCPY_ASYNC     cudaMemcpyAsync
  #define GPU_STREAM_SYNC      cudaStreamSynchronize
  #define GPU_MEMCPY_H2D       cudaMemcpyHostToDevice
  #define GPU_STREAM_CREATE    cudaStreamCreate
  #define GPU_STREAM_DESTROY   cudaStreamDestroy
  #define GPU_MALLOC           cudaMalloc
  #define GPU_FREE             cudaFree

#endif

/* ================= 封装：numInBatch = 1 ================= */

inline uintptr_t* allocAndCopyParamsSync(
    const void** in,
    const uint32_t* inSize,
    void** out,
    GpuStream stream) {

  static_assert(sizeof(void*) == sizeof(uintptr_t), "");
  static_assert(sizeof(uint32_t) <= sizeof(uintptr_t), "");

  // host staging buffer（栈上即可）
  uintptr_t host_params[3];
  host_params[0] = reinterpret_cast<uintptr_t>(*in);
  host_params[1] = static_cast<uintptr_t>(*inSize);
  host_params[2] = reinterpret_cast<uintptr_t>(*out);

  // device allocation
  uintptr_t* dev_params = nullptr;
  GPU_CHECK(GPU_MALLOC(&dev_params, 3 * sizeof(uintptr_t)));

  // async copy H2D
  GPU_CHECK(GPU_MEMCPY_ASYNC(
      dev_params,
      host_params,
      3 * sizeof(uintptr_t),
      GPU_MEMCPY_H2D,
      stream));
  GPU_CHECK(GPU_STREAM_SYNC(stream));
  return dev_params;
}

/* ================= main：可直接运行 ================= */

int main() {
  // fake input / output pointers（仅用于演示）
  int dummy_in = 42;
  int dummy_out = 0;
  uint32_t dummy_size = sizeof(int);

  const void* in = &dummy_in;
  void* out = &dummy_out;

  // create stream
  GpuStream stream;
  GPU_CHECK(GPU_STREAM_CREATE(&stream));

  // call our unified async function
  uintptr_t* dev_params =
      allocAndCopyParamsSync(&in, &dummy_size, &out, stream);

  // wait for async memcpy to finish
  

  // cleanup
  GPU_CHECK(GPU_FREE(dev_params));
  GPU_CHECK(GPU_STREAM_DESTROY(stream));

  printf("Async param copy completed successfully.\n");
  return 0;
}
