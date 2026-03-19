#include "../worker.h"
#include "test_support.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#define N 1024

using UKernel::Device::Testing::ck;
using UKernel::Device::Testing::feq;
using UKernel::Device::Testing::fill;

static bool wait_for_done(UKernel::Device::WorkerPool& pool, uint64_t id, uint32_t fifo_id, int timeout_ms = 5000) {
    auto start = std::chrono::steady_clock::now();
    while (!pool.is_done(id, fifo_id)) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed > timeout_ms) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

static uint64_t submit_copy_task(
    UKernel::Device::WorkerPool& pool,
    void* dst, void const* src, uint64_t bytes,
    UKernel::Device::DataType dtype, uint32_t fifo_id) {
    UKernel::Device::TaskArgs h{};
    h.src = const_cast<void*>(src);
    h.src2 = nullptr;
    h.dst = dst;
    h.bytes = bytes;
    h.src_rank = 0;
    h.dst_rank = 0;
    h.src_device = 0;
    h.dst_device = 0;
    h.redType = UKernel::Device::ReduceType::None;
    h.flags = 0;

    UKernel::Device::Task t =
        UKernel::Device::TaskManager::instance().create_task(
            h, UKernel::Device::TaskType::CollCopy, dtype, fifo_id);

    return pool.enqueue(t, fifo_id);
}

bool test_create_and_enqueue() {
    printf("[TEST] create_and_enqueue\n");

    UKernel::Device::WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 64;
    config.fifoCapacity = 16;

    float *dst = nullptr, *src = nullptr;
    ck(gpuMalloc(&dst, N * sizeof(float)), "gpuMalloc");
    ck(gpuMalloc(&src, N * sizeof(float)), "gpuMalloc");

    std::vector<float> h_src(N), h_dst(N);
    fill(h_src, 1.5f, 0.25f);
    ck(gpuMemcpy(src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice), "H2D");
    ck(gpuMemset(dst, 0, N * sizeof(float)), "memset");
    ck(gpuStreamSynchronize(0), "sync");

    {
        UKernel::Device::WorkerPool pool(config);
        
        bool ok = pool.createWorker(0, 1);
        printf("[DEBUG] createWorker(0, 1) = %s\n", ok ? "true" : "false");
        if (!ok) {
            printf("[FAIL] createWorker failed\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }

        pool.waitWorker(0);
        printf("[DEBUG] worker 0 ready\n");

        uint64_t id = submit_copy_task(pool, dst, src, N * sizeof(float),
                                       UKernel::Device::DataType::Fp32, 0);

        if (!wait_for_done(pool, id, 0)) {
            printf("[FAIL] timeout\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }
        pool.shutdown_all();
    }

    ck(gpuStreamSynchronize(0), "sync");
    ck(gpuMemcpy(h_dst.data(), dst, N * sizeof(float), gpuMemcpyDeviceToHost), "D2H");

    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
        if (!feq(h_dst[i], h_src[i])) ++bad;
    }

    ck(gpuFree(dst), "gpuFree");
    ck(gpuFree(src), "gpuFree");

    if (bad == 0) {
        printf("[PASS] create_and_enqueue\n");
        return true;
    }
    printf("[FAIL] data mismatch\n");
    return false;
}

bool test_multiple_workers() {
    printf("[TEST] multiple_workers\n");

    UKernel::Device::WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 64;
    config.fifoCapacity = 16;

    float *dst = nullptr, *src = nullptr;
    ck(gpuMalloc(&dst, N * sizeof(float)), "gpuMalloc");
    ck(gpuMalloc(&src, N * sizeof(float)), "gpuMalloc");

    std::vector<float> h_src(N), h_dst(N);
    fill(h_src, 2.5f, 0.5f);
    ck(gpuMemcpy(src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice), "H2D");
    ck(gpuMemset(dst, 0, N * sizeof(float)), "memset");
    ck(gpuStreamSynchronize(0), "sync");

    {
        UKernel::Device::WorkerPool pool(config);
        
        pool.createWorker(0, 1);
        pool.createWorker(1, 2);
        pool.waitWorker(0);
        pool.waitWorker(1);

        uint64_t id0 = submit_copy_task(pool, dst, src, N * sizeof(float),
                                       UKernel::Device::DataType::Fp32, 0);
        uint64_t id1 = submit_copy_task(pool, dst, src, N * sizeof(float),
                                       UKernel::Device::DataType::Fp32, 1);

        if (!wait_for_done(pool, id0, 0)) {
            printf("[FAIL] timeout fifo 0\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }
        if (!wait_for_done(pool, id1, 1)) {
            printf("[FAIL] timeout fifo 1\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }
        pool.shutdown_all();
    }

    ck(gpuStreamSynchronize(0), "sync");
    ck(gpuMemcpy(h_dst.data(), dst, N * sizeof(float), gpuMemcpyDeviceToHost), "D2H");

    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
        if (!feq(h_dst[i], h_src[i])) ++bad;
    }

    ck(gpuFree(dst), "gpuFree");
    ck(gpuFree(src), "gpuFree");

    if (bad == 0) {
        printf("[PASS] multiple_workers\n");
        return true;
    }
    printf("[FAIL] data mismatch\n");
    return false;
}

bool test_create_duplicate_fails() {
    printf("[TEST] create_duplicate_fails\n");

    UKernel::Device::WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 64;
    config.fifoCapacity = 16;

    float *dst = nullptr, *src = nullptr;
    ck(gpuMalloc(&dst, N * sizeof(float)), "gpuMalloc");
    ck(gpuMalloc(&src, N * sizeof(float)), "gpuMalloc");

    std::vector<float> h_src(N), h_dst(N);
    fill(h_src, 3.5f, 0.35f);
    ck(gpuMemcpy(src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice), "H2D");
    ck(gpuMemset(dst, 0, N * sizeof(float)), "memset");
    ck(gpuStreamSynchronize(0), "sync");

    {
        UKernel::Device::WorkerPool pool(config);
        
        bool ok1 = pool.createWorker(0, 1);
        printf("[DEBUG] createWorker(0, 1) = %s\n", ok1 ? "true" : "false");
        
        bool ok2 = pool.createWorker(0, 1);
        printf("[DEBUG] createWorker(0, 1) again = %s (should be false)\n", ok2 ? "true" : "false");
        
        if (!ok1 || ok2) {
            printf("[FAIL] expected ok1=true, ok2=false\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }

        pool.waitWorker(0);

        uint64_t id = submit_copy_task(pool, dst, src, N * sizeof(float),
                                       UKernel::Device::DataType::Fp32, 0);

        if (!wait_for_done(pool, id, 0)) {
            printf("[FAIL] timeout\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }
        pool.shutdown_all();
    }

    ck(gpuStreamSynchronize(0), "sync");
    ck(gpuMemcpy(h_dst.data(), dst, N * sizeof(float), gpuMemcpyDeviceToHost), "D2H");

    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
        if (!feq(h_dst[i], h_src[i])) ++bad;
    }

    ck(gpuFree(dst), "gpuFree");
    ck(gpuFree(src), "gpuFree");

    if (bad == 0) {
        printf("[PASS] create_duplicate_fails\n");
        return true;
    }
    printf("[FAIL] data mismatch\n");
    return false;
}

bool test_poll_worker() {
    printf("[TEST] poll_worker\n");

    UKernel::Device::WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 64;
    config.fifoCapacity = 16;

    float *dst = nullptr, *src = nullptr;
    ck(gpuMalloc(&dst, N * sizeof(float)), "gpuMalloc");
    ck(gpuMalloc(&src, N * sizeof(float)), "gpuMalloc");

    std::vector<float> h_src(N), h_dst(N);
    fill(h_src, 4.5f, 0.45f);
    ck(gpuMemcpy(src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice), "H2D");
    ck(gpuMemset(dst, 0, N * sizeof(float)), "memset");
    ck(gpuStreamSynchronize(0), "sync");

    {
        UKernel::Device::WorkerPool pool(config);
        
        printf("[DEBUG] before createWorker, pollWorker(0) = %s\n", 
               pool.pollWorker(0) ? "true" : "false");
        
        pool.createWorker(0, 1);
        
        printf("[DEBUG] after createWorker, pollWorker(0) = %s\n", 
               pool.pollWorker(0) ? "true" : "false");
        
        int attempts = 0;
        while (!pool.pollWorker(0) && attempts < 100) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            attempts++;
        }
        
        printf("[DEBUG] after wait, pollWorker(0) = %s (attempts=%d)\n", 
               pool.pollWorker(0) ? "true" : "false", attempts);

        if (!pool.pollWorker(0)) {
            printf("[FAIL] worker not ready\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }

        uint64_t id = submit_copy_task(pool, dst, src, N * sizeof(float),
                                       UKernel::Device::DataType::Fp32, 0);

        if (!wait_for_done(pool, id, 0)) {
            printf("[FAIL] timeout\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }
        pool.shutdown_all();
    }

    ck(gpuStreamSynchronize(0), "sync");
    ck(gpuMemcpy(h_dst.data(), dst, N * sizeof(float), gpuMemcpyDeviceToHost), "D2H");

    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
        if (!feq(h_dst[i], h_src[i])) ++bad;
    }

    ck(gpuFree(dst), "gpuFree");
    ck(gpuFree(src), "gpuFree");

    if (bad == 0) {
        printf("[PASS] poll_worker\n");
        return true;
    }
    printf("[FAIL] data mismatch\n");
    return false;
}

bool test_destroy_worker() {
    printf("[TEST] destroy_worker\n");

    UKernel::Device::WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 64;
    config.fifoCapacity = 16;

    float *dst = nullptr, *src = nullptr;
    ck(gpuMalloc(&dst, N * sizeof(float)), "gpuMalloc");
    ck(gpuMalloc(&src, N * sizeof(float)), "gpuMalloc");

    std::vector<float> h_src(N), h_dst(N);
    fill(h_src, 5.5f, 0.55f);
    ck(gpuMemcpy(src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice), "H2D");
    ck(gpuMemset(dst, 0, N * sizeof(float)), "memset");
    ck(gpuStreamSynchronize(0), "sync");

    {
        UKernel::Device::WorkerPool pool(config);
        
        pool.createWorker(0, 1);
        pool.waitWorker(0);

        uint64_t id = submit_copy_task(pool, dst, src, N * sizeof(float),
                                       UKernel::Device::DataType::Fp32, 0);

        if (!wait_for_done(pool, id, 0)) {
            printf("[FAIL] timeout before destroy\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }

        pool.destroyWorker(0);
        printf("[DEBUG] after destroyWorker(0)\n");

        bool ok = pool.createWorker(0, 1);
        printf("[DEBUG] createWorker(0, 1) after destroy = %s\n", ok ? "true" : "false");
        
        if (!ok) {
            printf("[FAIL] should be able to recreate worker\n");
            ck(gpuFree(dst), "gpuFree"); ck(gpuFree(src), "gpuFree");
            return false;
        }

        pool.shutdown_all();
    }

    ck(gpuFree(dst), "gpuFree");
    ck(gpuFree(src), "gpuFree");

    printf("[PASS] destroy_worker\n");
    return true;
}

int main() {
    printf("=== WorkerPool API Tests ===\n\n");
    UKernel::Device::TaskManager::instance().init(1024);

    int passed = 0, failed = 0;

    if (test_create_and_enqueue()) passed++; else failed++;
    if (test_multiple_workers()) passed++; else failed++;
    if (test_create_duplicate_fails()) passed++; else failed++;
    if (test_poll_worker()) passed++; else failed++;
    if (test_destroy_worker()) passed++; else failed++;

    printf("\n=== Summary: %d/%d passed ===\n", passed, passed + failed);
    return failed > 0 ? 1 : 0;
}