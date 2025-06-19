#include "peer_copy_worker.hpp"
#include "common.hpp"
#include "peer_copy.cuh"

CopyRing g_ring;
std::atomic<bool> g_run;
thread_local uint64_t async_memcpy_count = 0;
thread_local uint64_t async_memcpy_total_time = 0;
int src_device = 0;

void peer_copy_worker() {
  // one stream for this thread
  cudaStream_t stream;
  cudaSetDevice(src_device);  // source GPU in your example
  cudaStreamCreate(&stream);

  // peer-enable table (dst GPU -> enabled?)
  bool peer_ok[NUM_GPUS] = {false};

  while (g_run.load(std::memory_order_acquire)) {
    CopyTask* t = g_ring.pop();
    if (!t) {
      // nothing to do – yield the core briefly
      std::this_thread::yield();
      continue;
    }
    // printf("Peer copy worker popped task: wr_id=%llu, dst_dev=%d, src_ptr=%p,
    // "
    //        "dst_ptr=%p, bytes=%zu\n",
    //        static_cast<unsigned long long>(t->wr_id), t->dst_dev,
    //        t->src_ptr, t->dst_ptr, t->bytes);

    if (t->dst_dev == src_device) {
      // if dst_dev is 0, we are copying to the same GPU, so skip
      // this task
      async_memcpy_count++;
      continue;
    }

    // enable peer access to this destination once
    if (!peer_ok[t->dst_dev]) {
      cudaDeviceEnablePeerAccess(t->dst_dev, 0);
      cudaSetDevice(t->dst_dev);
      cudaDeviceEnablePeerAccess(0, 0);  // back-enable to src
      cudaSetDevice(0);
      peer_ok[t->dst_dev] = true;
    }

    auto st = std::chrono::high_resolution_clock::now();
    // printf("Before cudaMemcpyPeerAsync\n");
    // cudaError_t err = cudaMemcpyPeerAsync(t->dst_ptr, t->dst_dev, t->src_ptr,
    // src_device, t->bytes, stream);

    cudaError_t err = launch_peer_bulk_copy(t->dst_ptr, t->dst_dev, t->src_ptr,
                                            src_device, t->bytes, stream);

    // printf("cudaMemcpyPeerAsync finished!\n");
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaMemcpyPeerAsync failed (%s) wr_id=%llu\n",
              cudaGetErrorString(err),
              static_cast<unsigned long long>(t->wr_id));
      std::abort();
    }

    if (async_memcpy_count % kRemoteNVLinkBatchSize == 0) {
      err = cudaStreamSynchronize(stream);
      if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n",
                cudaGetErrorString(err));
        std::abort();
      }
    }

    async_memcpy_count++;
    async_memcpy_total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - st)
            .count();
    if (async_memcpy_count % 100000 == 0) {
      printf("Total async memcpy calls: %lu\n", async_memcpy_count);
      if (async_memcpy_count == 0) {
        printf("No async memcpy calls were made.\n");
      } else {
        printf("Average async memcpy time: %lu us\n",
               async_memcpy_total_time / async_memcpy_count);
        printf(
            "Ring size: %d, head: %u, tail: %u, emplace count: %u, pop count: "
            "%u\n",
            COPY_RING_CAP, g_ring.head.load(), g_ring.tail.load(),
            g_ring.emplace_count.load(), g_ring.pop_count.load());
      }
    }
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}