#include "peer_copy_worker.hpp"
#include "common.hpp"
#include "peer_copy.cuh"
#include <mutex>

std::atomic<bool> g_run;
thread_local uint64_t async_memcpy_count = 0;
thread_local uint64_t async_memcpy_total_time = 0;
int src_device = 0;
std::once_flag peer_ok_flag[NUM_GPUS][NUM_GPUS];

void maybe_enable_peer_access(int src_dev, int dst_dev) {
  if (src_dev == dst_dev) return;
  std::call_once(peer_ok_flag[src_dev][dst_dev], [&]() {
    cudaSetDevice(dst_dev);
    cudaError_t err = cudaDeviceEnablePeerAccess(src_dev, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "Peer access from dst_dev=%d to src_dev=%d failed: %s\n",
              dst_dev, src_dev, cudaGetErrorString(err));
    }

    cudaSetDevice(src_dev);
    err = cudaDeviceEnablePeerAccess(dst_dev, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "Peer access from src_dev=%d to dst_dev=%d failed: %s\n",
              src_dev, dst_dev, cudaGetErrorString(err));
    }
  });
}

void peer_copy_worker(CopyRing& g_ring, int idx) {
  pin_thread_to_cpu(idx + 1 + MAIN_THREAD_CPU_IDX);
  printf("Peer copy worker %d started on CPU core %d\n", idx + 1,
         sched_getcpu());

  cudaStream_t stream;
  cudaSetDevice(src_device);
  cudaStreamCreate(&stream);
  CopyTask* d_tasks;
  cudaMallocAsync(&d_tasks, RECEIVER_BATCH_SIZE * sizeof(CopyTask), stream);
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
    std::string func_name;

    if (false) {
      err = cudaMemcpyPeerAsync(t.dst_ptr, t.dst_dev, t.src_ptr, src_device,
                                t.bytes * copy_batch_size, stream);
      func_name = "cudaMemcpyPeerAsync";
    } else if (false) {
      err = launch_peer_bulk_copy(t.dst_ptr, t.dst_dev, t.src_ptr, src_device,
                                  t.bytes * copy_batch_size, stream);
      func_name = "launch_peer_bulk_copy";
    } else {
      err = launch_peer_bulk_copy2(tasks.data(), tasks.size(), stream,
                                   src_device, d_tasks);
      func_name = "launch_peer_bulk_copy2";
    }

    if (err != cudaSuccess) {
      fprintf(stderr, "%s failed (%s) wr_id=%llu\n", func_name.c_str(),
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
    if (false && async_memcpy_count % 100000 == 0) {
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
  cudaFreeAsync(d_tasks, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}