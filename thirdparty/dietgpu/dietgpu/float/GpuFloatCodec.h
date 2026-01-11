/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include "dietgpu/ans/GpuANSCodec.h"

namespace dietgpu {

class StackDeviceMemory;

// The various floating point types we support for compression
enum class FloatType : uint32_t {
  kUndefined = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kFloat32 = 3,
};

// Returns the maximum possible compressed size in bytes of an array of `size`
// float words of type `floatType`. Note that this will in fact be larger than
// size * sizeof(the float word type), as if something is uncompressible it will
// be expanded during compression.
// This can be used to bound memory consumption for the destination compressed
// buffer
uint32_t getMaxFloatCompressedSize(FloatType floatType, uint32_t size);

struct FloatCodecConfig {
  inline FloatCodecConfig()
      : floatType(FloatType::kFloat16),
        useChecksum(false),
        is16ByteAligned(false) {}

  inline FloatCodecConfig(
      FloatType ft,
      const ANSCodecConfig& ansConf,
      bool align,
      bool checksum = false)
      : floatType(ft),
        useChecksum(checksum),
        ansConfig(ansConf),
        is16ByteAligned(align) {
    // ANS-level checksumming is not allowed in float mode, only float level
    // checksumming
    assert(!ansConf.useChecksum);
  }

  // What kind of floats are we compressing/decompressing?
  FloatType floatType;

  // If true, we calculate a checksum on the uncompressed float input data to
  // compression and store it in the archive, and on the decompression side
  // post-decompression, we calculate a checksum on the decompressed data which
  // is compared with the original stored in the archive.
  // This is an optional feature useful if DietGPU data will be stored
  // persistently on disk.
  bool useChecksum;

  // ANS entropy coder parameters
  // Checksumming will happen at the float level, not the ANS level, as
  // decompression from ANS is immediately consumed by the float layer,
  // so ansConfig.useChecksum being true is an error.
  ANSCodecConfig ansConfig;

  // Are all all float input pointers/offsets (compress) or output
  // pointers/offsets (decompress) are aligned to 16 bytes?
  //
  // If so, we can accelerate the decompression. If not, the float addresses
  // should be aligned to the floating point word size (e.g.,
  // FloatType::kFloat16, all are assumed sizeof(float16) == 2 byte aligned)
  bool is16ByteAligned;
};

// Same config options for compression and decompression for now
using FloatCompressConfig = FloatCodecConfig;
using FloatDecompressConfig = FloatCodecConfig;

enum class FloatDecompressError : uint32_t {
  None = 0,
  ChecksumMismatch = 1,
};

// Error status for decompression
struct FloatDecompressStatus {
  inline FloatDecompressStatus() : error(FloatDecompressError::None) {}

  // Overall error status
  FloatDecompressError error;

  // Error-specific information for the batch
  std::vector<std::pair<int, std::string>> errorInfo;
};

//
// Encode
//

void floatCompress(
    StackDeviceMemory& res,
    // How should we compress our data?
    const FloatCompressConfig& config,

    // Optional region of device temporary memory provided for our use
    // Usage of this region of memory is ordered with respect to `stream`,
    // so can be reused after execution of the kernels that we launch on
    // that stream.
    // If either nullptr is passed, or if the size is not sufficient for our
    // needs, we will internally call cudaMalloc and cudaFree and will
    // print warnings to stderr in this case. Providing a sufficient sized chunk
    // of temp memory avoids the h2d synchronization overhead of
    // cudaMalloc/cudaFree.
    // The base address should be aligned to 16 bytes
    // void* tempMem_dev,
    // // The size in bytes of tempMem
    // size_t tempMemBytes,

    // Number of separate, independent compression problems
    uint32_t numInBatch,

    // Host array with addresses of device pointers comprising the batch
    const void** in,
    // Host array with sizes of batch members (in float words, NOT bytes)
    const uint32_t* inSize,

    // Host array with addresses of device pointers of outputs, each pointing
    // to a valid region of memory of at least size
    // getMaxFloatCompressedSize(ft, inSize[i])
    void** out,
    // Device memory array of size numInBatch (optional)
    // Provides the size of actual used memory in bytes for each batch element
    uint32_t* outSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

void floatCompressSplitSize(
    StackDeviceMemory& res,
    // How should we compress our data?
    const FloatCompressConfig& config,

    // Number of separate, independent compression problems
    uint32_t numInBatch,

    // Device pointer into a valid region of memory of size at least
    // sum_i(inSplitSizes[i]) float words.
    const void* in_dev,

    // Host array with the size (in floating point words) of the input
    // floating point arrays in the batch.
    // Each array in the batch is read starting at offset splitSize[i].
    const uint32_t* inSplitSizes,

    // Device pointer to a matrix of at least size
    // numInBatch x getMaxFloatCompressedSize(ft, max(inSplitSizes[i]))
    void* out_dev,

    // Stride between rows in bytes
    uint32_t outStride,

    // Device memory array of size numInBatch (optional)
    // Provides the size of actual used memory in bytes for each batch element
    uint32_t* outSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

//
// Decode
//

FloatDecompressStatus floatDecompress(
    StackDeviceMemory& res,
    // How should we decompress our data?
    const FloatDecompressConfig& config,
    // Number of separate, independent compression problems
    uint32_t numInBatch,

    // Host array with addresses of device pointers comprising the batch
    const void** in,

    // Host array with addresses of device pointers of outputs, each pointing
    // to a valid region of memory of at least size outCapacity[i]
    void** out,
    // Host memory array of size numInBatch (optional)
    // Provides the maximum amount of space present for decopressing each batch
    // problem
    const uint32_t* outCapacity,

    // Decode success/fail status (optional, can be nullptr)
    // If present, this is a device pointer to an array of length numInBatch,
    // with true/false for whether or not decompression status was successful
    // FIXME: not bool due to issues with __nv_bool
    uint8_t* outSuccess_dev,

    // Decode size status (optional, can be nullptr)
    // If present, this is a device pointer to an array of length numInBatch,
    // with either the size decompressed reported if successful, or the required
    // size reported if our outPerBatchCapacity was insufficient. Size reported
    // is in float words
    uint32_t* outSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

FloatDecompressStatus floatDecompressSplitSize(
    StackDeviceMemory& res,
    // How should we decompress our data?
    const FloatDecompressConfig& config,
    // Number of separate, independent compression problems
    uint32_t numInBatch,

    // Host array with addresses of device pointers comprising the batch
    const void** in,

    // Device pointer into a valid region of memory of size at least
    // sum_i(outSplitSizes[i]) float words
    void* out_dev,

    // Host array with the size (in floating point words) of the output
    // decompressed floating point arrays in the batch.
    // Each decompressed array in the batch is written at offset
    // outSplitSizes[i].
    // The decompressed size must match exactly these sizes, otherwise there's a
    // decompression error
    const uint32_t* outSplitSizes,

    // Decode success/fail status (optional, can be nullptr)
    // If present, this is a device pointer to an array of length numInBatch,
    // with true/false for whether or not decompression status was successful
    // FIXME: not bool due to issues with __nv_bool
    uint8_t* outSuccess_dev,

    // Decode size status (optional, can be nullptr)
    // If present, this is a device pointer to an array of length numInBatch,
    // with either the size decompressed reported if successful, or the required
    // size reported if our outPerBatchCapacity was insufficient. Size reported
    // is in float words
    uint32_t* outSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

//
// Information
//

void floatGetCompressedInfo(
    StackDeviceMemory& res,
    // Host array with addresses of device pointers comprising the batch of
    // compressed float data
    const void** in,
    // Number of compressed arrays in the batch
    uint32_t numInBatch,
    // Optional device array to receive the resulting sizes. 0 is reported if
    // the compresed data is not as expected, otherwise the size is reported in
    // floating point words
    uint32_t* outSizes_dev,
    // Optional device array to receive the resulting FloatTypes.
    // FloatType::kUndefined is reported if the compresed data is not as
    // expected, otherwise the size is reported in floating point words
    uint32_t* outTypes_dev,
    // Optional device array to receive pre-compression checksums stored in the
    // archive, if the checksum feature was enabled.
    uint32_t* outChecksum_dev,
    // stream on the current device on which this runs
    cudaStream_t stream);

void floatGetCompressedInfoDevice(
    StackDeviceMemory& res,
    // Device array with addresses of device pointers comprising the batch of
    // compressed float data
    const void** in_dev,
    // Number of compressed arrays in the batch
    uint32_t numInBatch,
    // Optional device array to receive the resulting sizes. 0 is reported if
    // the compresed data is not as expected, otherwise the size is reported in
    // floating point words
    uint32_t* outSizes_dev,
    // Optional device array to receive the resulting FloatTypes.
    // FloatType::kUndefined is reported if the compresed data is not as
    // expected, otherwise the size is reported in floating point words
    uint32_t* outTypes_dev,
    // Optional device array to receive pre-compression checksums stored in the
    // archive, if the checksum feature was enabled.
    uint32_t* outChecksum_dev,
    // stream on the current device on which this runs
    cudaStream_t stream);

} // namespace dietgpu
