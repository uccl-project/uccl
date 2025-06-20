#include "peer_copy_worker.hpp"
#include "common.hpp"
#include "peer_copy.cuh"
#include <mutex>

std::atomic<bool> g_run;
thread_local uint64_t async_memcpy_count = 0;
thread_local uint64_t async_memcpy_total_time = 0;
int src_device = 0;
std::once_flag peer_ok_flag[NUM_GPUS];

void maybe_enable_peer_access(int src_dev, int dst_dev) {
  std::call_once(peer_ok_flag[dst_dev], [&]() {
    cudaSetDevice(dst_dev);
    cudaDeviceEnablePeerAccess(src_dev, 0);  // enable from dst to src
    cudaSetDevice(src_dev);
    cudaDeviceEnablePeerAccess(dst_dev, 0);  // enable from src to dst
  });
}

void peer_copy_worker(CopyRing& g_ring, int idx) {
  // one stream for this thread

  pin_thread_to_cpu(idx + 1 + MAIN_THREAD_CPU_IDX);
  printf("Peer copy worker %d started on CPU core %d\n", idx + 1,
         sched_getcpu());

  cudaStream_t stream;
  cudaSetDevice(src_device);  // source GPU in your example
  cudaStreamCreate(&stream);

  while (g_run.load(std::memory_order_acquire)) {
    CopyTask t;
    int copy_batch_size = 0;
    if (RECEIVER_BATCH_SIZE == 1) {
      CopyTask* t_ptr = g_ring.pop();
      if (!t_ptr) {
        continue;
      }
      t = *t_ptr;
      copy_batch_size = 1;
    } else {
      std::vector<CopyTask> tasks;
      size_t n = g_ring.popN(tasks, RECEIVER_BATCH_SIZE);
      if (n == 0) {
        continue;
      }
      t = tasks[0];
      copy_batch_size = n;
    }

    if (copy_batch_size == 0) {
      fprintf(stderr, "Error: copy_batch_size is zero\n");
      std::abort();
    }
    // printf("Peer copy worker popped task: wr_id=%llu, dst_dev=%d, src_ptr=%p,
    // "
    //        "dst_ptr=%p, bytes=%zu\n",
    //        static_cast<unsigned long long>(t->wr_id), t->dst_dev,
    //        t->src_ptr, t->dst_ptr, t->bytes);

    if (t.dst_dev == src_device) {
      // if dst_dev is 0, we are copying to the same GPU, so skip
      // this task
      async_memcpy_count += copy_batch_size;
      continue;
    }

    // enable peer access to this destination once
    maybe_enable_peer_access(src_device, t.dst_dev);

    auto st = std::chrono::high_resolution_clock::now();
    // printf("Before cudaMemcpyPeerAsync\n");
    cudaError_t err =
        cudaMemcpyPeerAsync(t.dst_ptr, t.dst_dev, t.src_ptr, src_device,
                            t.bytes * copy_batch_size, stream);

    // printf("After cudaMemcpyPeerAsync, wr_id=%llu, dst_dev=%d, src_ptr=%p, "
    //        "dst_ptr=%p, bytes=%zu\n",
    //        static_cast<unsigned long long>(t.wr_id), t.dst_dev, t.src_ptr,
    //        t.dst_ptr, t.bytes * copy_batch_size);

    // cudaError_t err = launch_peer_bulk_copy(t.dst_ptr, t.dst_dev, t.src_ptr,
    //                                         src_device, t.bytes *
    //                                         copy_batch_size, stream);

    // printf("cudaMemcpyPeerAsync finished!\n");
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaMemcpyPeerAsync failed (%s) wr_id=%llu\n",
              cudaGetErrorString(err),
              static_cast<unsigned long long>(t.wr_id));
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

    async_memcpy_count += copy_batch_size;
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
            "%u, ratio: %d\n",
            COPY_RING_CAP, g_ring.head.load(), g_ring.tail.load(),
            g_ring.emplace_count.load(), g_ring.pop_count.load(),
            g_ring.emplace_count.load() / g_ring.pop_count.load());
      }
    }
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}