/**
 * Test program for DietGPU float compression on AMD HIP
 *
 * Build:
 *   make -f Makefile.hip test
 *
 * Run:
 *   ./float_compress_test
 */

#include <hip/hip_runtime.h>
#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>
#include <cassert>

#include "dietgpu/float/GpuFloatCodec_hip.h"
#include "dietgpu/utils/StackDeviceMemory_hip.h"

using namespace dietgpu;

#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t error = cmd;                                                \
        if (error != hipSuccess) {                                             \
            std::cerr << "HIP error: " << hipGetErrorString(error)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Test float16 compression/decompression
void testFloat16Compression() {
    std::cout << "=== Testing Float16 Compression ===" << std::endl;

    const uint32_t numFloats = 1024 * 16;  // 16K float16 values
    const uint32_t numBytes = numFloats * sizeof(uint16_t);

    // Create StackDeviceMemory for temporary allocations
    auto res = makeStackMemory();

    // Create HIP stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Allocate host memory for input data
    std::vector<uint16_t> hostInput(numFloats);

    // Fill with random data (simulating float16 bit patterns)
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint16_t> dist(0, 65535);
    for (uint32_t i = 0; i < numFloats; ++i) {
        hostInput[i] = dist(gen);
    }

    // Allocate device memory for input
    uint16_t* devInput = nullptr;
    HIP_CHECK(hipMalloc(&devInput, numBytes));
    HIP_CHECK(hipMemcpyAsync(devInput, hostInput.data(), numBytes,
                              hipMemcpyHostToDevice, stream));

    // Allocate device memory for compressed output
    uint32_t maxCompressedSize = getMaxFloatCompressedSize(FloatType::kFloat16, numFloats);
    std::cout << "Max compressed size: " << maxCompressedSize << " bytes" << std::endl;

    void* devCompressed = nullptr;
    HIP_CHECK(hipMalloc(&devCompressed, maxCompressedSize));

    // Allocate device memory for compressed size output
    uint32_t* devCompressedSize = nullptr;
    HIP_CHECK(hipMalloc(&devCompressedSize, sizeof(uint32_t)));

    // Setup compression config
    FloatCompressConfig compressConfig;
    compressConfig.floatType = FloatType::kFloat16;
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    // Setup batch (single element batch)
    const void* inPtrs[1] = { devInput };
    uint32_t inSizes[1] = { numFloats };
    void* outPtrs[1] = { devCompressed };

    // Compress
    std::cout << "Compressing..." << std::endl;
    floatCompress(
        res,
        compressConfig,
        1,              // numInBatch
        inPtrs,
        inSizes,
        outPtrs,
        devCompressedSize,
        stream
    );

    // Get compressed size
    uint32_t compressedSize = 0;
    HIP_CHECK(hipMemcpyAsync(&compressedSize, devCompressedSize, sizeof(uint32_t),
                              hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::cout << "Original size: " << numBytes << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressedSize << " bytes" << std::endl;
    std::cout << "Compression ratio: " << (float)numBytes / compressedSize << "x" << std::endl;

    // Allocate device memory for decompressed output
    uint16_t* devDecompressed = nullptr;
    HIP_CHECK(hipMalloc(&devDecompressed, numBytes));
    HIP_CHECK(hipMemset(devDecompressed, 0, numBytes));

    // Setup decompression config
    FloatDecompressConfig decompressConfig;
    decompressConfig.floatType = FloatType::kFloat16;
    decompressConfig.useChecksum = false;
    decompressConfig.is16ByteAligned = true;

    // Setup batch for decompression
    const void* compInPtrs[1] = { devCompressed };
    void* decompOutPtrs[1] = { devDecompressed };
    uint32_t outCapacities[1] = { numFloats };

    // Decompress
    std::cout << "Decompressing..." << std::endl;
    FloatDecompressStatus status = floatDecompress(
        res,
        decompressConfig,
        1,              // numInBatch
        compInPtrs,
        decompOutPtrs,
        outCapacities,
        nullptr,        // outSuccess_dev (optional)
        nullptr,        // outSize_dev (optional)
        stream
    );

    HIP_CHECK(hipStreamSynchronize(stream));

    if (status.error != FloatDecompressError::None) {
        std::cerr << "Decompression failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy decompressed data back to host
    std::vector<uint16_t> hostDecompressed(numFloats);
    HIP_CHECK(hipMemcpy(hostDecompressed.data(), devDecompressed, numBytes,
                         hipMemcpyDeviceToHost));

    // Verify data
    bool success = true;
    for (uint32_t i = 0; i < numFloats; ++i) {
        if (hostInput[i] != hostDecompressed[i]) {
            std::cerr << "Mismatch at index " << i << ": expected "
                      << hostInput[i] << ", got " << hostDecompressed[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Float16 compression/decompression test PASSED!" << std::endl;
    } else {
        std::cout << "Float16 compression/decompression test FAILED!" << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(devInput));
    HIP_CHECK(hipFree(devCompressed));
    HIP_CHECK(hipFree(devCompressedSize));
    HIP_CHECK(hipFree(devDecompressed));
    HIP_CHECK(hipStreamDestroy(stream));

    std::cout << std::endl;
}

// Test float32 compression/decompression
void testFloat32Compression() {
    std::cout << "=== Testing Float32 Compression ===" << std::endl;

    const uint32_t numFloats = 1024 * 8;  // 8K float32 values
    const uint32_t numBytes = numFloats * sizeof(float);

    // Create StackDeviceMemory for temporary allocations
    auto res = makeStackMemory();

    // Create HIP stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Allocate host memory for input data
    std::vector<float> hostInput(numFloats);

    // Fill with random float data
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (uint32_t i = 0; i < numFloats; ++i) {
        hostInput[i] = dist(gen);
    }

    // Allocate device memory for input
    float* devInput = nullptr;
    HIP_CHECK(hipMalloc(&devInput, numBytes));
    HIP_CHECK(hipMemcpyAsync(devInput, hostInput.data(), numBytes,
                              hipMemcpyHostToDevice, stream));

    // Allocate device memory for compressed output
    uint32_t maxCompressedSize = getMaxFloatCompressedSize(FloatType::kFloat32, numFloats);
    std::cout << "Max compressed size: " << maxCompressedSize << " bytes" << std::endl;

    void* devCompressed = nullptr;
    HIP_CHECK(hipMalloc(&devCompressed, maxCompressedSize));

    // Allocate device memory for compressed size output
    uint32_t* devCompressedSize = nullptr;
    HIP_CHECK(hipMalloc(&devCompressedSize, sizeof(uint32_t)));

    // Setup compression config
    FloatCompressConfig compressConfig;
    compressConfig.floatType = FloatType::kFloat32;
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    // Setup batch (single element batch)
    const void* inPtrs[1] = { devInput };
    uint32_t inSizes[1] = { numFloats };
    void* outPtrs[1] = { devCompressed };

    // Compress
    std::cout << "Compressing..." << std::endl;
    floatCompress(
        res,
        compressConfig,
        1,              // numInBatch
        inPtrs,
        inSizes,
        outPtrs,
        devCompressedSize,
        stream
    );

    // Get compressed size
    uint32_t compressedSize = 0;
    HIP_CHECK(hipMemcpyAsync(&compressedSize, devCompressedSize, sizeof(uint32_t),
                              hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::cout << "Original size: " << numBytes << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressedSize << " bytes" << std::endl;
    std::cout << "Compression ratio: " << (float)numBytes / compressedSize << "x" << std::endl;

    // Allocate device memory for decompressed output
    float* devDecompressed = nullptr;
    HIP_CHECK(hipMalloc(&devDecompressed, numBytes));
    HIP_CHECK(hipMemset(devDecompressed, 0, numBytes));

    // Setup decompression config
    FloatDecompressConfig decompressConfig;
    decompressConfig.floatType = FloatType::kFloat32;
    decompressConfig.useChecksum = false;
    decompressConfig.is16ByteAligned = true;

    // Setup batch for decompression
    const void* compInPtrs[1] = { devCompressed };
    void* decompOutPtrs[1] = { devDecompressed };
    uint32_t outCapacities[1] = { numFloats };

    // Decompress
    std::cout << "Decompressing..." << std::endl;
    FloatDecompressStatus status = floatDecompress(
        res,
        decompressConfig,
        1,              // numInBatch
        compInPtrs,
        decompOutPtrs,
        outCapacities,
        nullptr,        // outSuccess_dev (optional)
        nullptr,        // outSize_dev (optional)
        stream
    );

    HIP_CHECK(hipStreamSynchronize(stream));

    if (status.error != FloatDecompressError::None) {
        std::cerr << "Decompression failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy decompressed data back to host
    std::vector<float> hostDecompressed(numFloats);
    HIP_CHECK(hipMemcpy(hostDecompressed.data(), devDecompressed, numBytes,
                         hipMemcpyDeviceToHost));

    // Verify data (compare as uint32_t to avoid floating point comparison issues)
    bool success = true;
    for (uint32_t i = 0; i < numFloats; ++i) {
        uint32_t origBits, decompBits;
        std::memcpy(&origBits, &hostInput[i], sizeof(uint32_t));
        std::memcpy(&decompBits, &hostDecompressed[i], sizeof(uint32_t));

        if (origBits != decompBits) {
            std::cerr << "Mismatch at index " << i << ": expected bits "
                      << std::hex << origBits << ", got " << decompBits << std::dec << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Float32 compression/decompression test PASSED!" << std::endl;
    } else {
        std::cout << "Float32 compression/decompression test FAILED!" << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(devInput));
    HIP_CHECK(hipFree(devCompressed));
    HIP_CHECK(hipFree(devCompressedSize));
    HIP_CHECK(hipFree(devDecompressed));
    HIP_CHECK(hipStreamDestroy(stream));

    std::cout << std::endl;
}

// Test getMaxFloatCompressedSize function
void testGetMaxCompressedSize() {
    std::cout << "=== Testing getMaxFloatCompressedSize ===" << std::endl;

    uint32_t size16 = getMaxFloatCompressedSize(FloatType::kFloat16, 1000);
    uint32_t sizeBF16 = getMaxFloatCompressedSize(FloatType::kBFloat16, 1000);
    uint32_t size32 = getMaxFloatCompressedSize(FloatType::kFloat32, 1000);

    std::cout << "Float16 max compressed size for 1000 elements: " << size16 << " bytes" << std::endl;
    std::cout << "BFloat16 max compressed size for 1000 elements: " << sizeBF16 << " bytes" << std::endl;
    std::cout << "Float32 max compressed size for 1000 elements: " << size32 << " bytes" << std::endl;

    std::cout << "getMaxFloatCompressedSize test PASSED!" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize glog
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    // Check for HIP devices
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Found " << deviceCount << " HIP device(s)" << std::endl;

    // Get device properties
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Using device: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
    std::cout << std::endl;

    // Run tests
    testGetMaxCompressedSize();
    testFloat16Compression();
    testFloat32Compression();

    std::cout << "All tests completed!" << std::endl;

    return EXIT_SUCCESS;
}
