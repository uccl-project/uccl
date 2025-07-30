#include "peer_copy_worker.hpp"
#include "common.hpp"
#include "peer_copy.cuh"
#include "proxy.hpp"
#include "rdma.hpp"
#include <mutex>

void maybe_enable_peer_access(PeerCopyShared& shared, int dst_dev) {
  if (shared.src_device == dst_dev) return;
  std::call_once(shared.peer_ok_flag[shared.src_device][dst_dev], [&]() {
    cudaSetDevice(dst_dev);
    cudaError_t err = cudaDeviceEnablePeerAccess(shared.src_device, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "Peer access from dst_dev=%d to src_dev=%d failed: %s\n",
              dst_dev, shared.src_device, cudaGetErrorString(err));
    }

    cudaSetDevice(shared.src_device);
    err = cudaDeviceEnablePeerAccess(dst_dev, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "Peer access from src_dev=%d to dst_dev=%d failed: %s\n",
              shared.src_device, dst_dev, cudaGetErrorString(err));
    }
  });
}

void sync_and_post(PeerWorkerCtx& ctx, CopyRingBuffer& ring,
                   cudaStream_t& stream, int idx) {
  if (ctx.async_memcpy_count > ctx.prev_completed_async_memcpy_count) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
      std::abort();
    }
    remote_send_ack(ring.ack_qp, ctx.highest_issued_wr_id, ring.ack_mr,
                    ring.ack_buf, idx);
    ctx.prev_completed_async_memcpy_count = ctx.async_memcpy_count;
  }
}

void peer_copy_worker(PeerCopyShared& shared, PeerWorkerCtx& ctx,
                      CopyRingBuffer& ring, int idx) {
  pin_thread_to_cpu(idx + 1 + MAIN_THREAD_CPU_IDX);
  printf("Peer copy worker %d started on CPU core %d\n", idx + 1,
         sched_getcpu());

  cudaStream_t stream;
  cudaSetDevice(shared.src_device);
  cudaStreamCreate(&stream);
  CopyTask* d_tasks;
  cudaMallocAsync(&d_tasks, RECEIVER_BATCH_SIZE * sizeof(CopyTask), stream);

#ifdef REMOTE_PERSISTENT_KERNEL
  cudaStream_t persistent_stream;
  cudaStreamCreate(&persistent_stream);
  HostToDeviceNVlinkBuffer* rb =
      initialize_ring_buffer_for_nvlink_forwarding(persistent_stream);
#endif

  while (shared.run.load(std::memory_order_acquire)) {
    CopyTask t;
    int copy_batch_size = 0;
    if (RECEIVER_BATCH_SIZE == 1) {
      if (!ring.pop(t)) {
        sync_and_post(ctx, ring, stream, idx);
        continue;
      }
      copy_batch_size = 1;
      ctx.tasks[0] = t;
    } else {
      size_t n = ring.popN(ctx.tasks, RECEIVER_BATCH_SIZE);
      if (n == 0) {
        sync_and_post(ctx, ring, stream, idx);
        continue;
      }
      t = ctx.tasks[0];
      copy_batch_size = n;
    }

    if (copy_batch_size == 0) {
      fprintf(stderr, "Error: copy_batch_size is zero\n");
      std::abort();
    }

    for (int i = 0; i < copy_batch_size; ++i) {
      maybe_enable_peer_access(shared, ctx.tasks[i].dst_dev);
      ctx.task_wrs[i] = ctx.tasks[i].wr_id;
    }

    ctx.highest_issued_wr_id =
        std::max(ctx.highest_issued_wr_id, ctx.task_wrs[copy_batch_size - 1]);

    auto st = std::chrono::high_resolution_clock::now();
    cudaError_t err;
    std::string func_name;

    if (false) {
      err = cudaMemcpyPeerAsync(t.dst_ptr, t.dst_dev, t.src_ptr,
                                shared.src_device, t.bytes * copy_batch_size,
                                stream);
      func_name = "cudaMemcpyPeerAsync";
    } else if (false) {
      err = launch_peer_bulk_copy(t.dst_ptr, t.dst_dev, t.src_ptr,
                                  shared.src_device, t.bytes * copy_batch_size,
                                  stream);
      func_name = "launch_peer_bulk_copy";
#ifdef REMOTE_PERSISTENT_KERNEL
    } else if (false) {
#else
    } else {
#endif
      /* The fastest among the three. */
      err = launch_peer_bulk_copy2(ctx.tasks, copy_batch_size, stream,
                                   shared.src_device, d_tasks);
      func_name = "launch_peer_bulk_copy2";
    }
#ifdef REMOTE_PERSISTENT_KERNEL
    else {
      bool post_success = true;
      while (!post_success)
        post_success = post_copy_task(rb, ctx.tasks, copy_batch_size, stream,
                                      shared.src_device, d_tasks);
    }
#endif
    if (err != cudaSuccess) {
      fprintf(stderr, "%s failed (%s) wr_id=%llu\n", func_name.c_str(),
              cudaGetErrorString(err),
              static_cast<unsigned long long>(t.wr_id));
      std::abort();
    }

    if (ctx.async_memcpy_count % kRemoteNVLinkBatchSize == 0 ||
        ctx.async_memcpy_count - ctx.prev_completed_async_memcpy_count >=
            kRemoteNVLinkBatchSize) {
      err = cudaStreamSynchronize(stream);
      if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n",
                cudaGetErrorString(err));
        std::abort();
      }

      if (copy_batch_size > 0) {
        // Post the last wr is enough.
        remote_send_ack(ring.ack_qp, ctx.highest_issued_wr_id, ring.ack_mr,
                        ring.ack_buf, idx);
      }
      ctx.prev_completed_async_memcpy_count = ctx.async_memcpy_count;
    }

    ctx.async_memcpy_count += copy_batch_size;
    ctx.async_memcpy_total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - st)
            .count();
  }
  cudaFreeAsync(d_tasks, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}