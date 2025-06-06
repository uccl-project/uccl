#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
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

__device__ uint __smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

__global__ void write_smid(int* d_sm_ids, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    d_sm_ids[idx] = __smid();
  }
}

void launch_smid(int* h_sm_ids, int num_sm) {
  int* d_sm_ids;
  size_t bytes = num_sm * sizeof(int);

  // Allocate device memory
  cudaMalloc((void**)&d_sm_ids, bytes);

  // Copy input data to device
  cudaMemcpy(d_sm_ids, h_sm_ids, bytes, cudaMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 1;
  int blocksPerGrid = num_sm;
  write_smid<<<blocksPerGrid, threadsPerBlock>>>(d_sm_ids, num_sm);

  // Copy result back to host
  cudaMemcpy(h_sm_ids, d_sm_ids, bytes, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_sm_ids);

  // Check for any errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

int main() {
  CUdevice device;
  CUcontext context;

  CHECK_CUDA(cuInit(0));
  CHECK_CUDA(cuDeviceGet(&device, 0));
  CHECK_CUDA(cuCtxCreate(&context, 0, device));

  // now, opt into green context
  CUdevResource sm_resource;
  CHECK_CUDA(
      cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));
  printf("SM Resource: %d\n", sm_resource.sm.smCount);

  // Split the SM resource
  unsigned int nbGroups = 4;                 // Desired number of groups
  CUdevResource result_resources[nbGroups];  // Array to store split resources
  CUdevResource remaining;
  unsigned int minCount = 8;  // Minimum SMs per group
  CHECK_CUDA(cuDevSmResourceSplitByCount(
      result_resources, &nbGroups, &sm_resource, &remaining, 0, minCount));
  printf("Number of groups created: %d\n", nbGroups);

  for (int i = 0; i < nbGroups; i++) {
    printf("Group %d: %d SMs\n", i, result_resources[i].sm.smCount);
  }

  printf("Remaining SMs: %d\n", remaining.sm.smCount);

  // generate descriptor for the first group
  CUdevResourceDesc desc;
  // CHECK_CUDA(cuDevResourceGenerateDesc(&desc, &result_resources[1], 1));
  CHECK_CUDA(cuDevResourceGenerateDesc(&desc, &remaining, 1));

  CUgreenCtx green_ctx;
  CHECK_CUDA(
      cuGreenCtxCreate(&green_ctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

  CUdevResource green_sm_resource;
  CHECK_CUDA(cuGreenCtxGetDevResource(green_ctx, &green_sm_resource,
                                      CU_DEV_RESOURCE_TYPE_SM));
  printf("Green SM Resource: %d\n", green_sm_resource.sm.smCount);

  CUcontext green_ctx_ctx;
  CHECK_CUDA(cuCtxFromGreenCtx(&green_ctx_ctx, green_ctx));

  CHECK_CUDA(cuCtxSetCurrent(green_ctx_ctx));

  int num_sm = 200;
  int h_sm_ids[num_sm];

  printf("Launch %d SMs\n", num_sm);

  launch_smid(h_sm_ids, num_sm);

  // sort sm ids
  std::sort(h_sm_ids, h_sm_ids + num_sm);

  for (int i = 0; i < num_sm; i++) {
    printf("%d ", h_sm_ids[i]);
  }
  printf("\n");

  CHECK_CUDA(cuCtxDestroy(context));

  return 0;
}