#include "scattered_memcpy.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define cudaCheckErrors(msg)                                  \
  do {                                                        \
    cudaError_t __err = cudaGetLastError();                   \
    if (__err != cudaSuccess) {                               \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
              cudaGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n");             \
      exit(1);                                                \
    }                                                         \
  } while (0)

__global__ void kernelScatteredMemcpy(uint32_t num_copies,
                                      __grid_constant__ const copy_param_t p) {
  // Total threads in the grid.
  int total_threads = gridDim.x * blockDim.x;
  // Compute our unique global thread id.
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  // Number of threads per copy.
  int threads_per_copy = total_threads / num_copies;

  // Map each thread to a copy.
  int copy_idx = global_id / threads_per_copy;
  if (copy_idx >= num_copies) return;  // In case of rounding

  // Compute local thread index within the group assigned to this copy.
  int local_thread_idx = global_id % threads_per_copy;

  // Retrieve parameters for this copy.
  uint64_t total_size = p.len[copy_idx];
  if (total_size == 0) return;

  char* src_ptr = (char*)p.src[copy_idx];
  char* dst_ptr = (char*)p.dst[copy_idx];

  // Copy 8-byte chunks first (if possible)
  uint64_t num_full = total_size / 8;
  uint64_t* src_u64 = (uint64_t*)src_ptr;
  uint64_t* dst_u64 = (uint64_t*)dst_ptr;

  // Each thread in the group copies its portion of 64-bit words.
  for (uint64_t i = local_thread_idx; i < num_full; i += threads_per_copy) {
    dst_u64[i] = src_u64[i];
  }

  // Handle the remaining tail bytes (if any)
  uint64_t tail_start = num_full * 8;
  // Let only one thread in the copy group (e.g. local_thread_idx == 0) copy
  // the tail.
  if (local_thread_idx == 0) {
    for (uint64_t i = tail_start; i < total_size; i++) {
      dst_ptr[i] = src_ptr[i];
    }
  }
}

void launchScatteredMemcpy(uint32_t num_copies, copy_param_t const* params) {
  // Launch the kernel
  kernelScatteredMemcpy<<<THREAD_BLOCKS, THREADS_PER_BLOCK>>>(num_copies,
                                                              *params);

  // Wait for kernel to complete.
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
            cudaGetErrorString(err));
  }
}

void launchScatteredMemcpyAsync(uint32_t num_copies, copy_param_t const* params,
                                cudaStream_t stream) {
  // Launch the kernel
  kernelScatteredMemcpy<<<THREAD_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
      num_copies, *params);

  // void* args[] = {(void*)&num_copies, (void*)params};
  // cudaLaunchCooperativeKernel((void*)kernelScatteredMemcpy,
  //                             dim3(THREAD_BLOCKS), dim3(THREADS_PER_BLOCK),
  //                             args, 0, stream);
}

int pollScatteredMemcpy(cudaStream_t stream) {
  cudaError_t err = cudaStreamQuery(stream);
  if (err != cudaSuccess && err != cudaErrorNotReady) {
    fprintf(stderr, "cudaStreamQuery failed: %s\n", cudaGetErrorString(err));
    exit(0);
  }
  return err == cudaSuccess;
}

__device__ uint __smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

__device__ int volatile blkcnt1 = 0;
__device__ int volatile blkcnt2 = 0;
__device__ int volatile itercnt = 0;

__device__ void my_compute_function(int* buf, int idx, int data) {
  buf[idx] = data;  // put your work code here
}

__global__ void persistentScatteredMemcpy(int* buffer1, int* buffer2,
                                          int volatile* buffer1_ready,
                                          int volatile* buffer2_ready,
                                          int const buffersize,
                                          int const iterations) {
  // assumption of persistent block-limited kernel launch
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int iter_count = 0;
  // persistent until iterations complete
  while (iter_count < iterations) {
    // while (true) {  // persistent
    int* buf =
        (iter_count & 1) ? buffer2 : buffer1;  // ping pong between buffers
    int volatile* bufrdy = (iter_count & 1) ? (buffer2_ready) : (buffer1_ready);
    int volatile* blkcnt = (iter_count & 1) ? (&blkcnt2) : (&blkcnt1);
    int my_idx = idx;
    while (iter_count - itercnt > 1)
      ;  // don't overrun buffers on device
    while (*bufrdy == 2)
      ;  // wait for buffer to be consumed
    printf("SM %d, block %d, thread %d\n", __smid(), blockIdx.x, threadIdx.x);
    while (my_idx < buffersize) {  // perform the "work"
      my_compute_function(buf, my_idx, iter_count);
      my_idx += gridDim.x * blockDim.x;  // grid-striding loop
    }
    __syncthreads();  // wait for my block to finish
    __threadfence();  // make sure global buffer writes are "visible"
    if (!threadIdx.x) atomicAdd((int*)blkcnt, 1);  // mark my block done
    if (!idx) {  // am I the master block/thread?
      while (*blkcnt < gridDim.x)
        ;  // wait for all blocks to finish
      *blkcnt = 0;
      *bufrdy = 2;             // indicate that buffer is ready
      __threadfence_system();  // push it out to mapped memory
      itercnt++;
    }
    iter_count++;
  }
}

#define DSIZE 65536
#define nTPB 256

void launchPersistentScatteredMemcpy(int iterations, cudaStream_t streamk) {
  int *h_buf1, *d_buf1, *h_buf2, *d_buf2;
  int volatile *m_bufrdy1, *m_bufrdy2;
  // buffer and "mailbox" setup
  cudaHostAlloc(&h_buf1, DSIZE * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc(&h_buf2, DSIZE * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc(&m_bufrdy1, sizeof(int), cudaHostAllocMapped);
  cudaHostAlloc(&m_bufrdy2, sizeof(int), cudaHostAllocMapped);
  cudaCheckErrors("cudaHostAlloc fail");
  cudaMalloc(&d_buf1, DSIZE * sizeof(int));
  cudaMalloc(&d_buf2, DSIZE * sizeof(int));
  *m_bufrdy1 = 0;
  *m_bufrdy2 = 0;
  cudaMemset(d_buf1, 0xFF, DSIZE * sizeof(int));
  cudaMemset(d_buf2, 0xFF, DSIZE * sizeof(int));
  cudaCheckErrors("cudaMemset fail");

  int nblock = 1;
  persistentScatteredMemcpy<<<nblock, nTPB, 0, streamk>>>(
      d_buf1, d_buf2, m_bufrdy1, m_bufrdy2, DSIZE, iterations);
  cudaCheckErrors("kernel launch fail");
}
