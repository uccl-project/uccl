#include <cstring>
#include <iostream>
#include <vector>
#include <cuda_bf16.h>
#include "worker.h"
#include "gpu_rt.h"

using namespace UKernel::Device;

static bool g_pass = true;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        g_pass = false; \
    } \
} while(0)

void test_worker_poll() {
    std::cout << "1. init" << std::endl;
    TaskManager::instance().init(256);
    
    std::cout << "2. config" << std::endl;
    WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 64;
    config.fifoCapacity = 16;
    
    std::cout << "3. WorkerPool constructor" << std::endl;
    WorkerPool wp(config);
    std::cout << "4. wp created" << std::endl;
    
    std::cout << "5. pollWorker" << std::endl;
    TEST_ASSERT(wp.pollWorker(0) == false, "pollWorker before createWorker");
    std::cout << "6. pollWorker done" << std::endl;
    
    std::cout << "7. createWorker" << std::endl;
    bool created = wp.createWorker(0, 1);
    TEST_ASSERT(created == true, "createWorker should succeed");
    std::cout << "8. createWorker done" << std::endl;
    
    std::cout << "9. waitWorker" << std::endl;
    wp.waitWorker(0);
    std::cout << "10. waitWorker done" << std::endl;
    
    std::cout << "11. pollWorker" << std::endl;
    TEST_ASSERT(wp.pollWorker(0) == true, "pollWorker after worker ready");
    std::cout << "12. pollWorker done" << std::endl;
    
    std::cout << "13. shutdown_all" << std::endl;
    wp.shutdown_all();
    std::cout << "14. shutdown_all done" << std::endl;
    
    std::cout << "15. release" << std::endl;
    TaskManager::instance().release();
    std::cout << "16. release done" << std::endl;
    
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << "17. sync done" << std::endl;
    std::cout << "[PASS] Worker Poll test" << std::endl;
}

void test_single_sm_copy() {
    std::cout << "\n=== Test: Single SM Copy ===" << std::endl;
    TaskManager::instance().init(256);
    
    WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 256;
    config.fifoCapacity = 16;
    
    WorkerPool wp(config);
    
    const size_t bytes = 1024 * 1024;
    void* d_src, *d_dst;
    GPU_RT_CHECK(gpuMalloc(&d_src, bytes));
    GPU_RT_CHECK(gpuMalloc(&d_dst, bytes));
    
    std::vector<char> h_src(bytes, 0);
    std::vector<char> h_dst(bytes, 0);
    for (size_t i = 0; i < bytes; i++) {
        h_src[i] = static_cast<char>(i & 0xFF);
    }
    GPU_RT_CHECK(gpuMemcpy(d_src, h_src.data(), bytes, gpuMemcpyHostToDevice));
    
    TaskArgs args;
    memset(&args, 0, sizeof(args));
    args.src = d_src;
    args.dst = d_dst;
    args.bytes = bytes;
    
    uint32_t argsIdx = TaskManager::instance().alloc_task_args();
    GPU_RT_CHECK(gpuMemcpy(TaskManager::instance().d_task_args() + argsIdx, &args, sizeof(TaskArgs), gpuMemcpyHostToDevice));
    
    wp.createWorker(0, 1);
    wp.waitWorker(0);
    
    Task task(TaskType::CollCopy, DataType::Int8, 0, argsIdx);
    uint64_t taskId = wp.enqueue(task, 0);
    std::cout << "Enqueued task " << taskId << std::endl;
    
    // Wait for task completion using WorkerPool's is_done method
    while (!wp.is_done(taskId, 0)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    // Check dst buffer before full copy
    std::vector<char> h_dst_check(bytes, 0);
    GPU_RT_CHECK(gpuMemcpy(h_dst_check.data(), d_dst, 64, gpuMemcpyDeviceToHost));
    std::cout << "After kernel, first 64 bytes of dst:" << std::endl;
    for (int i = 0; i < 64; i++) {
        if (i % 16 == 0) std::cout << "\n";
        std::cout << (int)h_dst_check[i] << " ";
    }
    std::cout << std::endl;
    
    GPU_RT_CHECK(gpuMemcpy(h_dst.data(), d_dst, bytes, gpuMemcpyDeviceToHost));
    
    bool match = true;
    for (size_t i = 0; i < bytes; i++) {
        if (h_dst[i] != h_src[i]) {
            match = false;
            std::cerr << "Mismatch at " << i << ": " << (int)h_dst[i] << " vs " << (int)h_src[i] << std::endl;
            break;
        }
    }
    TEST_ASSERT(match, "Copy content matches");
    
    wp.shutdown_all();
    TaskManager::instance().free_task_args(argsIdx);
    gpuFree(d_src);
    gpuFree(d_dst);
    TaskManager::instance().release();
    
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << "[PASS] Single SM Copy test" << std::endl;
}

void test_multi_block_copy() {
    std::cout << "\n=== Test: Multi-Block Copy ===" << std::endl;
    TaskManager::instance().init(256);
    
    WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 256;
    config.fifoCapacity = 16;
    
    WorkerPool wp(config);
    
    const size_t bytes = 4 * 1024 * 1024;
    void* d_src, *d_dst;
    GPU_RT_CHECK(gpuMalloc(&d_src, bytes));
    GPU_RT_CHECK(gpuMalloc(&d_dst, bytes));
    
    std::vector<char> h_src(bytes, 0);
    std::vector<char> h_dst(bytes, 0);
    for (size_t i = 0; i < bytes; i++) {
        h_src[i] = static_cast<char>((i * 3) & 0xFF);
    }
    GPU_RT_CHECK(gpuMemcpy(d_src, h_src.data(), bytes, gpuMemcpyHostToDevice));
    
    TaskArgs args;
    memset(&args, 0, sizeof(args));
    args.src = d_src;
    args.dst = d_dst;
    args.bytes = bytes;
    
    uint32_t argsIdx = TaskManager::instance().alloc_task_args();
    GPU_RT_CHECK(gpuMemcpy(TaskManager::instance().d_task_args() + argsIdx, &args, sizeof(TaskArgs), gpuMemcpyHostToDevice));
    
    wp.createWorker(0, 4);
    wp.waitWorker(0);
    
    Task task(TaskType::CollCopy, DataType::Int8, 0, argsIdx);
    uint64_t taskId = wp.enqueue(task, 0);
    std::cout << "Enqueued task " << taskId << " with 4 blocks" << std::endl;
    
    // Wait for task completion using WorkerPool's is_done method
    while (!wp.is_done(taskId, 0)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    GPU_RT_CHECK(gpuMemcpy(h_dst.data(), d_dst, bytes, gpuMemcpyDeviceToHost));
    
    bool match = true;
    for (size_t i = 0; i < bytes; i++) {
        if (h_dst[i] != h_src[i]) {
            match = false;
            std::cerr << "Mismatch at " << i << std::endl;
            break;
        }
    }
    TEST_ASSERT(match, "Multi-block copy content matches");
    
    wp.shutdown_all();
    TaskManager::instance().free_task_args(argsIdx);
    gpuFree(d_src);
    gpuFree(d_dst);
    TaskManager::instance().release();
    
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << "[PASS] Multi-Block Copy test" << std::endl;
}

void test_data_types() {
    std::cout << "\n=== Test: Multiple Data Types ===" << std::endl;
    TaskManager::instance().init(256);
    
    WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 256;
    config.fifoCapacity = 16;
    
    WorkerPool wp(config);
    
    const size_t bytes_f32 = 256 * sizeof(float);
    const size_t bytes_f16 = 256 * sizeof(__half);
    
    void* d_src_f32, *d_dst_f32;
    void* d_src_f16, *d_dst_f16;
    GPU_RT_CHECK(gpuMalloc(&d_src_f32, bytes_f32));
    GPU_RT_CHECK(gpuMalloc(&d_dst_f32, bytes_f32));
    GPU_RT_CHECK(gpuMalloc(&d_src_f16, bytes_f16));
    GPU_RT_CHECK(gpuMalloc(&d_dst_f16, bytes_f16));
    
    std::vector<float> h_src_f32(256);
    std::vector<float> h_dst_f32(256, 0);
    std::vector<__half> h_src_f16(256);
    std::vector<__half> h_dst_f16(256, __float2half(0.0f));
    
    for (int i = 0; i < 256; i++) {
        h_src_f32[i] = static_cast<float>(i * 2.5f);
        h_src_f16[i] = __float2half(static_cast<float>(i * 1.5f));
    }
    
    GPU_RT_CHECK(gpuMemcpy(d_src_f32, h_src_f32.data(), bytes_f32, gpuMemcpyHostToDevice));
    GPU_RT_CHECK(gpuMemcpy(d_src_f16, h_src_f16.data(), bytes_f16, gpuMemcpyHostToDevice));
    
    auto testDataType = [&](void* d_src, void* d_dst, size_t bytes, DataType dtype, const char* name) {
        TaskArgs args;
        memset(&args, 0, sizeof(args));
        args.src = d_src;
        args.dst = d_dst;
        args.bytes = bytes;
        
        uint32_t argsIdx = TaskManager::instance().alloc_task_args();
        GPU_RT_CHECK(gpuMemcpy(TaskManager::instance().d_task_args() + argsIdx, &args, sizeof(TaskArgs), gpuMemcpyHostToDevice));
        
        wp.createWorker(0, 1);
        wp.waitWorker(0);
        
        Task task(TaskType::CollCopy, dtype, 0, argsIdx);
        uint64_t taskId = wp.enqueue(task, 0);
        
        // Wait for task completion using WorkerPool's is_done method
        while (!wp.is_done(taskId, 0)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        std::vector<char> h_dst(bytes);
        GPU_RT_CHECK(gpuMemcpy(h_dst.data(), d_dst, bytes, gpuMemcpyDeviceToHost));
        
        bool match = true;
        if (dtype == DataType::Fp32) {
            for (int i = 0; i < 256; i++) {
                float val = *reinterpret_cast<float*>(&h_dst[i * sizeof(float)]);
                if (abs(val - h_src_f32[i]) > 0.01f) {
                    match = false;
                    break;
                }
            }
        } else if (dtype == DataType::Fp16) {
            for (int i = 0; i < 256; i++) {
                __half val = *reinterpret_cast<__half*>(&h_dst[i * sizeof(__half)]);
                if (abs(__half2float(val) - __half2float(h_src_f16[i])) > 0.01f) {
                    match = false;
                    break;
                }
            }
        }
        TEST_ASSERT(match, std::string("Copy ") + name + " matches");
        
        wp.shutdown_all();
        TaskManager::instance().free_task_args(argsIdx);
    };
    
    testDataType(d_src_f32, d_dst_f32, bytes_f32, DataType::Fp32, "Fp32");
    testDataType(d_src_f16, d_dst_f16, bytes_f16, DataType::Fp16, "Fp16");
    
    gpuFree(d_src_f32);
    gpuFree(d_dst_f32);
    gpuFree(d_src_f16);
    gpuFree(d_dst_f16);
    TaskManager::instance().release();
    
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << "[PASS] Multiple Data Types test" << std::endl;
}

void test_repeated_launch() {
    std::cout << "\n=== Test: Repeated Worker Launch ===" << std::endl;
    TaskManager::instance().init(256);
    
    WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 256;
    config.fifoCapacity = 16;
    
    WorkerPool wp(config);
    
    const size_t bytes = 1024;
    void* d_src, *d_dst;
    GPU_RT_CHECK(gpuMalloc(&d_src, bytes));
    GPU_RT_CHECK(gpuMalloc(&d_dst, bytes));
    
    std::vector<char> h_src(bytes, 0);
    std::vector<char> h_dst(bytes, 0);
    for (size_t i = 0; i < bytes; i++) {
        h_src[i] = static_cast<char>(i);
    }
    GPU_RT_CHECK(gpuMemcpy(d_src, h_src.data(), bytes, gpuMemcpyHostToDevice));
    
    for (int iter = 0; iter < 5; iter++) {
        std::cout << "Iteration " << iter << std::endl;
        
        TaskArgs args;
        memset(&args, 0, sizeof(args));
        args.src = d_src;
        args.dst = d_dst;
        args.bytes = bytes;
        
        uint32_t argsIdx = TaskManager::instance().alloc_task_args();
        GPU_RT_CHECK(gpuMemcpy(TaskManager::instance().d_task_args() + argsIdx, &args, sizeof(TaskArgs), gpuMemcpyHostToDevice));
        
        wp.createWorker(0, 1);
        wp.waitWorker(0);
        
        Task task(TaskType::CollCopy, DataType::Int8, 0, argsIdx);
        uint64_t taskId = wp.enqueue(task, 0);
        
        // Wait for task completion using WorkerPool's is_done method
        while (!wp.is_done(taskId, 0)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        GPU_RT_CHECK(gpuMemcpy(h_dst.data(), d_dst, bytes, gpuMemcpyDeviceToHost));
        
        bool match = true;
        for (size_t i = 0; i < bytes; i++) {
            if (h_dst[i] != h_src[i]) {
                match = false;
                break;
            }
        }
        TEST_ASSERT(match, std::string("Iteration ") + std::to_string(iter) + " matches");
        
        wp.shutdown_all();
        TaskManager::instance().free_task_args(argsIdx);
    }
    
    gpuFree(d_src);
    gpuFree(d_dst);
    TaskManager::instance().release();
    
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << "[PASS] Repeated Launch test" << std::endl;
}

void test_repeated_enqueue() {
    std::cout << "\n=== Test: Repeated Task Submission ===" << std::endl;
    TaskManager::instance().init(256);
    
    WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 256;
    config.fifoCapacity = 64;
    
    WorkerPool wp(config);
    
    const size_t bytes = 512;
    void* d_src, *d_dst;
    GPU_RT_CHECK(gpuMalloc(&d_src, bytes));
    GPU_RT_CHECK(gpuMalloc(&d_dst, bytes));
    
    std::vector<char> h_src(bytes, 0);
    std::vector<char> h_dst(bytes, 0);
    for (size_t i = 0; i < bytes; i++) {
        h_src[i] = static_cast<char>(i);
    }
    GPU_RT_CHECK(gpuMemcpy(d_src, h_src.data(), bytes, gpuMemcpyHostToDevice));
    
    wp.createWorker(0, 1);
    wp.waitWorker(0);
    
    // Track all task IDs for later synchronization
    std::vector<uint64_t> taskIds;
    for (int iter = 0; iter < 10; iter++) {
        TaskArgs args;
        memset(&args, 0, sizeof(args));
        args.src = d_src;
        args.dst = d_dst;
        args.bytes = bytes;
        
        uint32_t argsIdx = TaskManager::instance().alloc_task_args();
        GPU_RT_CHECK(gpuMemcpy(TaskManager::instance().d_task_args() + argsIdx, &args, sizeof(TaskArgs), gpuMemcpyHostToDevice));
        
        Task task(TaskType::CollCopy, DataType::Int8, 0, argsIdx);
        uint64_t taskId = wp.enqueue(task, 0);
        taskIds.push_back(taskId);
    }
    
    // Wait for all tasks to complete
    for (uint64_t taskId : taskIds) {
        while (!wp.is_done(taskId, 0)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
    
    GPU_RT_CHECK(gpuMemcpy(h_dst.data(), d_dst, bytes, gpuMemcpyDeviceToHost));
    
    bool match = true;
    for (size_t i = 0; i < bytes; i++) {
        if (h_dst[i] != h_src[i]) {
            match = false;
            std::cerr << "Mismatch at " << i << std::endl;
            break;
        }
    }
    TEST_ASSERT(match, "All enqueued tasks completed correctly");
    
    wp.shutdown_all();
    TaskManager::instance().release();
    
    gpuFree(d_src);
    gpuFree(d_dst);
    TaskManager::instance().release();
    
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << "[PASS] Repeated Enqueue test" << std::endl;
}

void test_multiple_fifos() {
    std::cout << "\n=== Test: Multiple FIFOs ===" << std::endl;
    TaskManager::instance().init(256);
    
    WorkerPool::Config config;
    config.numMaxWorkers = 8;
    config.threadsPerBlock = 256;
    config.fifoCapacity = 16;
    
    WorkerPool wp(config);
    
    const size_t bytes = 256;
    std::vector<void*> d_src(3), d_dst(3);
    std::vector<std::vector<char>> h_src(3, std::vector<char>(bytes));
    std::vector<std::vector<char>> h_dst(3, std::vector<char>(bytes));
    
    for (int fifo = 0; fifo < 3; fifo++) {
        GPU_RT_CHECK(gpuMalloc(&d_src[fifo], bytes));
        GPU_RT_CHECK(gpuMalloc(&d_dst[fifo], bytes));
        for (size_t i = 0; i < bytes; i++) {
            h_src[fifo][i] = static_cast<char>(fifo * 100 + i);
        }
        GPU_RT_CHECK(gpuMemcpy(d_src[fifo], h_src[fifo].data(), bytes, gpuMemcpyHostToDevice));
    }
    
    for (int fifo = 0; fifo < 3; fifo++) {
        wp.createWorker(fifo, 1);
        wp.waitWorker(fifo);
    }
    
    // Track task IDs for each FIFO
    std::vector<uint64_t> taskIds(3);
    for (int fifo = 0; fifo < 3; fifo++) {
        TaskArgs args;
        memset(&args, 0, sizeof(args));
        args.src = d_src[fifo];
        args.dst = d_dst[fifo];
        args.bytes = bytes;
        
        uint32_t argsIdx = TaskManager::instance().alloc_task_args();
        GPU_RT_CHECK(gpuMemcpy(TaskManager::instance().d_task_args() + argsIdx, &args, sizeof(TaskArgs), gpuMemcpyHostToDevice));
        
        Task task(TaskType::CollCopy, DataType::Int8, 0, argsIdx);
        taskIds[fifo] = wp.enqueue(task, fifo);
    }
    
    for (int fifo = 0; fifo < 3; fifo++) {
        // Wait for each task to complete
        while (!wp.is_done(taskIds[fifo], fifo)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
    
    bool all_match = true;
    for (int fifo = 0; fifo < 3; fifo++) {
        GPU_RT_CHECK(gpuMemcpy(h_dst[fifo].data(), d_dst[fifo], bytes, gpuMemcpyDeviceToHost));
        for (size_t i = 0; i < bytes; i++) {
            if (h_dst[fifo][i] != h_src[fifo][i]) {
                all_match = false;
                std::cerr << "FIFO " << fifo << " mismatch at " << i << std::endl;
            }
        }
    }
    TEST_ASSERT(all_match, "All FIFOs data correct");
    
    for (int fifo = 0; fifo < 3; fifo++) {
        wp.shutdown_all();
    }
    
    TaskManager::instance().release();
    
    for (int fifo = 0; fifo < 3; fifo++) {
        gpuFree(d_src[fifo]);
        gpuFree(d_dst[fifo]);
    }
    TaskManager::instance().release();
    
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << "[PASS] Multiple FIFOs test" << std::endl;
}

void test_boundary() {
    std::cout << "\n=== Test: Boundary Conditions ===" << std::endl;
    TaskManager::instance().init(256);
    
    WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 256;
    config.fifoCapacity = 16;
    
    WorkerPool wp(config);
    
    auto testSize = [&](size_t bytes, const char* name) {
        void* d_src, *d_dst;
        GPU_RT_CHECK(gpuMalloc(&d_src, bytes));
        GPU_RT_CHECK(gpuMalloc(&d_dst, bytes));
        
        std::vector<char> h_src(bytes, 0);
        std::vector<char> h_dst(bytes, 0);
        for (size_t i = 0; i < bytes; i++) {
            h_src[i] = static_cast<char>(i);
        }
        GPU_RT_CHECK(gpuMemcpy(d_src, h_src.data(), bytes, gpuMemcpyHostToDevice));
        
        TaskArgs args;
        memset(&args, 0, sizeof(args));
        args.src = d_src;
        args.dst = d_dst;
        args.bytes = bytes;
        
        uint32_t argsIdx = TaskManager::instance().alloc_task_args();
        GPU_RT_CHECK(gpuMemcpy(TaskManager::instance().d_task_args() + argsIdx, &args, sizeof(TaskArgs), gpuMemcpyHostToDevice));
        
        wp.createWorker(0, 1);
        wp.waitWorker(0);
        
        Task task(TaskType::CollCopy, DataType::Int8, 0, argsIdx);
        uint64_t taskId = wp.enqueue(task, 0);
        
        // Wait for task completion using WorkerPool's is_done method
        while (!wp.is_done(taskId, 0)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        GPU_RT_CHECK(gpuMemcpy(h_dst.data(), d_dst, bytes, gpuMemcpyDeviceToHost));
        
        bool match = true;
        for (size_t i = 0; i < bytes; i++) {
            if (h_dst[i] != h_src[i]) {
                match = false;
                break;
            }
        }
        TEST_ASSERT(match, std::string(name) + " boundary test");
        
        wp.shutdown_all();
        TaskManager::instance().free_task_args(argsIdx);
        gpuFree(d_src);
        gpuFree(d_dst);
    };
    
    testSize(1, "1 byte");
    testSize(16, "16 bytes");
    testSize(256, "256 bytes");
    testSize(1024, "1 KB");
    testSize(65536, "64 KB");
    
    TaskManager::instance().release();
    GPU_RT_CHECK(gpuDeviceSynchronize());
    std::cout << "[PASS] Boundary Conditions test" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "Starting tests..." << std::endl;
    int deviceCount;
    GPU_RT_CHECK(gpuGetDeviceCount(&deviceCount));
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    
    if (deviceCount > 0) {
        GPU_RT_CHECK(gpuSetDevice(7));
        printf("Using device 7 (L40S)\n");
    }
    
    test_worker_poll();
    test_single_sm_copy();
    test_multi_block_copy();
    test_data_types();
    test_repeated_launch();
    test_repeated_enqueue();
    test_multiple_fifos();
    test_boundary();
    
    if (g_pass) {
        std::cout << "\n=== ALL TESTS PASSED! ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== SOME TESTS FAILED! ===" << std::endl;
        return 1;
    }
}
