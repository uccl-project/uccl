#include <iostream>
#include <vector>
#include <cassert>
#include "persistent.h"
#include "operator.h"
#include "c2d_fifo.h"

#define N 1024

using namespace eccl;

uint64_t submit_copy_task(PersistentKernel<OpTask>& kernel,
                          void* dst, const void* src,
                          uint64_t bytes, uint64_t wpt) {
    OpTask task(reinterpret_cast<uint64_t>(src),
                reinterpret_cast<uint64_t>(dst),
                bytes,
                OpTaskCopy, OpDataFp32, OpRedSum,
                wpt);
    return kernel.submit(task);
}

uint64_t submit_reduce_task(PersistentKernel<OpTask>& kernel,
                            void* dst, const void* src,
                            uint64_t elem_count, OpDataType dtype,
                            OpRedType redop, uint64_t wpt) {
    uint64_t elem_size = 0;
    switch (dtype) {
      case OpDataFp32: elem_size = 4; break;
      case OpDataFp16: elem_size = 2; break;
      case OpDataFp8:  elem_size = 1; break;
      default:         elem_size = 4; break;
    }

    OpTask task(reinterpret_cast<uint64_t>(src),
                reinterpret_cast<uint64_t>(dst),
                elem_count * elem_size,
                OpTaskReduce, dtype, redop,
                wpt);
    return kernel.submit(task);
}

void run_persistent_kernel() {
    PersistentKernelConfig config;
    config.numBlocks = 1;
    config.threadsPerBlock = 64;
    config.fifoCapacity = 16;
    config.smemSize = 0;

    PersistentKernel<OpTask> kernel(config);

    kernel.launch();
    std::cout << "Persistent kernel launched." << std::endl;

    uint64_t taskId = 0;

    // --- Copy test ---
    float *dst_copy = nullptr, *src_copy = nullptr;
    cudaMalloc(&dst_copy, N * sizeof(float));
    cudaMalloc(&src_copy, N * sizeof(float));

    // 每线程处理 16 个 float => encoded wpt = 15
    taskId = submit_copy_task(kernel, dst_copy, src_copy, N * sizeof(float), 15);
    std::cout << "submit copy task finished." << std::endl;

    // 等待 copy 完成
    while (!kernel.is_done(taskId)) {}

    // --- Reduce test (inplace: dst = red(dst, src)) ---
    float *dst_reduce = nullptr, *src_reduce = nullptr;
    cudaMalloc(&dst_reduce, N * sizeof(float));
    cudaMalloc(&src_reduce, N * sizeof(float));

    // 线程处理 4 elements => encoded wpt = 3
    taskId = submit_reduce_task(kernel,
                                dst_reduce, src_reduce,
                                /*elem_count=*/N,
                                /*dtype=*/OpDataFp32,
                                /*redop=*/OpRedSum,
                                /*wpt=*/3);
    std::cout << "submit reduce task finished." << std::endl;

    while (!kernel.is_done(taskId)) {}

    std::cout << "Persistent kernel finished the task." << std::endl;

    kernel.stop();
    std::cout << "Stop signal sent." << std::endl;

    cudaFree(dst_copy);
    cudaFree(src_copy);
    cudaFree(dst_reduce);
    cudaFree(src_reduce);

    std::cout << "Persistent kernel started and completed." << std::endl;
}

int main() {
    run_persistent_kernel();
    return 0;
}
