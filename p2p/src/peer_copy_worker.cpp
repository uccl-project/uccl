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
    std::vector<CopyTask> tasks;
    int copy_batch_size = 0;
    if (RECEIVER_BATCH_SIZE == 1) {
      CopyTask* t_ptr = g_ring.pop();
      if (!t_ptr) {
        continue;
      }
      t = *t_ptr;
      copy_batch_size = 1;
      tasks.push_back(t);
    } else {
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
    if (t.dst_dev == src_device) {
      async_memcpy_count += copy_batch_size;
      continue;
    }

    for (auto task : tasks) {
      maybe_enable_peer_access(src_device, task.dst_dev);
    }

    auto st = std::chrono::high_resolution_clock::now();
    cudaError_t err;

    if (false) {
      err = cudaMemcpyPeerAsync(t.dst_ptr, t.dst_dev, t.src_ptr, src_device,
                                t.bytes * copy_batch_size, stream);
    } else if (false) {
      err = launch_peer_bulk_copy(t.dst_ptr, t.dst_dev, t.src_ptr, src_device,
                                  t.bytes * copy_batch_size, stream);
    } else {
      err = launch_peer_bulk_copy2(tasks.data(), tasks.size(), stream);
    }

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