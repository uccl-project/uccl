#include <iostream>
#include <thread>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    CUresult result = call;                                                  \
    if (result != CUDA_SUCCESS) {                                            \
      char const* errorString;                                               \
      cuGetErrorString(result, &errorString);                                \
      fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, errorString); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

#define N 1024  // Number of elements

__device__ uint __smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

// Dummy kernel to simulate computation
__global__ void kernel1(int* data) {
  // if (threadIdx.x == 0) printf("Kernel1 SM %d\n", __smid());
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (uint64_t i = 0; i < 10000000u; i++) {  // Simulate workload
    data[idx] *= 2 * i;
  }
}

// Another dummy kernel
__global__ void kernel2(int* data) {
  // if (threadIdx.x == 0) printf("Kernel2 SM %d\n", __smid());
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < 1000000; i++) {  // Simulate workload
    data[idx] *= 2 * i;
  }
}

unsigned int const kNbGroups = 2;        // Desired number of groups
CUdevResource resources[kNbGroups + 1];  // Array to store split resources
CUcontext context[kNbGroups + 1];

void PartitionGreenCtx() {
  CUdevice device;

  CHECK_CUDA(cuInit(0));
  CHECK_CUDA(cuDeviceGet(&device, 0));

  // now, opt into green context
  CUdevResource sm_resource;
  CHECK_CUDA(
      cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));
  printf("SM Resource: %d\n", sm_resource.sm.smCount);

  // Split the SM resource
  unsigned int minCount = 8;  // Minimum SMs per group
  uint32_t nbGroups = kNbGroups;
  CHECK_CUDA(cuDevSmResourceSplitByCount(resources, &nbGroups, &sm_resource,
                                         &resources[kNbGroups], 0, minCount));
  printf("Number of groups created: %d\n", nbGroups + 1);
  assert(nbGroups == kNbGroups);

  for (int i = 0; i < nbGroups + 1; i++) {
    printf("Group %d: %d SMs\n", i, resources[i].sm.smCount);

    // generate descriptor for the first group
    CUdevResourceDesc desc;
    CHECK_CUDA(cuDevResourceGenerateDesc(&desc, &resources[i], 1));

    CUgreenCtx green_ctx;
    CHECK_CUDA(cuGreenCtxCreate(&green_ctx, desc, device,
                                CU_GREEN_CTX_DEFAULT_STREAM));

    CUdevResource green_sm_resource;
    CHECK_CUDA(cuGreenCtxGetDevResource(green_ctx, &green_sm_resource,
                                        CU_DEV_RESOURCE_TYPE_SM));
    printf("Green SM Resource: %d\n", green_sm_resource.sm.smCount);

    CHECK_CUDA(cuCtxFromGreenCtx(&context[i], green_ctx));
  }
}

// nvcc -o cuda_concurrent cuda_concurrent.cu -lcuda -lcudart

int main() {
  PartitionGreenCtx();

  cudaStream_t stream1, stream2;
  int *d_data1, *d_data2;

  cuCtxSetCurrent(context[0]);
  cudaMalloc(&d_data1, N * sizeof(int));
  cudaStreamCreate(&stream1);

  cuCtxSetCurrent(context[1]);
  cudaMalloc(&d_data2, N * sizeof(int));
  cudaStreamCreate(&stream2);

  // Launching convention #1
  cuCtxSetCurrent(context[0]);
  for (int i = 0; i < 10; i++) {
    kernel1<<<4, 256, 0, stream1>>>(d_data1);
  }
  cuCtxSetCurrent(context[1]);
  for (int i = 0; i < 10; i++) {
    kernel2<<<1, 256, 0, stream2>>>(d_data2);
  }
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  // Launching convention #2
  for (int i = 0; i < 10; i++) {
    cuCtxSetCurrent(context[0]);
    kernel1<<<4, 256, 0, stream1>>>(d_data1);
    cuCtxSetCurrent(context[1]);
    kernel2<<<1, 256, 0, stream2>>>(d_data2);
  }
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  // Cleanup
  cudaFree(d_data1);
  cudaFree(d_data2);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  std::cout << "✅ Both kernels executed concurrently!" << std::endl;
  return 0;
}
