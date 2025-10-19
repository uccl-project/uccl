#include "../device/persistent.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <unistd.h>

using namespace tcpx::device;

static constexpr int kNumThBlocks = 4;
static constexpr int kNumThPerBlock = 1024;
static constexpr int kPipelineDepth = 2;
using DType = float4;

static constexpr int kCopySize = 8888;
static constexpr int kTestIovs = 128;

void fill_data(void **srcs_gpu, int iov_n, int *lens, uint8_t value,
               cudaStream_t stream) {
    // make a CPU buffer, then copy to GPU
    uint8_t *cpu_buf = (uint8_t *)malloc(kCopySize);
    for (unsigned i = 0; i < kCopySize / sizeof(uint8_t); i++) {
        cpu_buf[i] = value;
    }
    for (int i = 0; i < iov_n; i++) {
        cudaMemcpyAsync(srcs_gpu[i], cpu_buf, lens[i], cudaMemcpyHostToDevice,
                        stream);
        cudaStreamSynchronize(stream);
        cudaCheckErrors("cudaMemcpy failed");
    }
    free(cpu_buf);
}

void check_data(void **dsts_gpu, int iov_n, int *lens, uint8_t value,
                cudaStream_t stream) {
    // check the data
    uint8_t *cpu_buf = (uint8_t *)malloc(kCopySize);
    for (int i = 0; i < iov_n; i++) {
        cudaMemcpyAsync(cpu_buf, dsts_gpu[i], lens[i], cudaMemcpyDeviceToHost,
                        stream);
        cudaStreamSynchronize(stream);
        cudaCheckErrors("cudaMemcpy failed");
        for (unsigned j = 0; j < lens[i] / sizeof(uint8_t); j++) {
            assert(cpu_buf[j] == value);
        }
    }
    free(cpu_buf);
}

// -------------------- test wrapper --------------------
class TestPersistentKernel : public UnpackerPersistentKernel {
public:
    using UnpackerPersistentKernel::UnpackerPersistentKernel;
    cudaStream_t copy_stream = nullptr;

    bool submitTestSingle(int fifo_index, uint8_t data) {
        std::cout << "submit " << fifo_index << data << std::endl;
        if (copy_stream == nullptr) {
            cudaStreamCreate(&copy_stream);
            cudaCheckErrors("cudaStreamCreate failed");
        }

        int copy_size_once = 0;
        struct Iov *cpu_iov = (struct Iov *)malloc(sizeof(struct Iov));

        // new cpu iovs
        for (int i = 0; i < kTestIovs; i++) {
            cudaMalloc(&cpu_iov->srcs[i], kCopySize);
            cudaMalloc(&cpu_iov->dsts[i], kCopySize);
            cpu_iov->lens[i] = kCopySize - rand() % 2048;
            copy_size_once += cpu_iov->lens[i];
        }
        fill_data((void **)cpu_iov->srcs, kTestIovs, cpu_iov->lens, data, copy_stream);
        cpu_iov->iov_n = kTestIovs;

        auto slot_idx = submit(fifo_index, *cpu_iov);

        auto start = std::chrono::high_resolution_clock::now();
        // Wait for the GPU to finish the work.
        while (!is_done(fifo_index, slot_idx)) std::this_thread::yield();

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start);
        auto bw_GBps =
            copy_size_once * 1.0 / elapsed_us.count() / 1000;
        printf("CPU wait time: %ld us, bw: %lf GBps\n", elapsed_us.count(),
                bw_GBps);
        free(cpu_iov); 
        return true;
    }
};

// -------------------- main --------------------
int main() {
    PersistentKernelConfig cfg;
    cfg.numThBlocks = kNumThBlocks;
    cfg.numThPerBlock = kNumThPerBlock;
    cfg.smem_size = cfg.numThPerBlock * sizeof(DType) * kPipelineDepth;

    TestPersistentKernel kernel(cfg);

    if (!kernel.launch()) {
        std::cerr << "Failed to launch persistent kernel!" << std::endl;
        return -1;
    }

    std::cout << "Kernel launched successfully.\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    for (int i = 0; i < 4; i++){
        kernel.submitTestSingle(i % cfg.numThBlocks, i);
        sleep(1);
    }

    std::cout << "Stopping persistent kernel...\n";
    kernel.stop();

    std::cout << "✅ All persistent tests completed successfully.\n";
    return 0;
}