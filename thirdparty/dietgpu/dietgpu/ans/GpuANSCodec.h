/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include "dietgpu/utils/StackDeviceMemory.h"

namespace dietgpu {

// Required minimum alignment in bytes of all data to be compressed in the batch
constexpr int kANSRequiredAlignment = 4;

// Default number of probability quantization bits to use, if an alternative is
// not specified
constexpr int kANSDefaultProbBits = 10;

uint32_t getMaxCompressedSize(uint32_t uncompressedBytes);

struct ANSCodecConfig {
  inline ANSCodecConfig() : probBits(kANSDefaultProbBits), useChecksum(false) {}

  explicit inline ANSCodecConfig(int pb, bool checksum = false)
      : probBits(pb), useChecksum(checksum) {}

  // What the ANS probability accuracy is; all symbols have quantized
  // probabilities of 1/2^probBits.
  // 9, 10, 11 are only valid values. When in doubt, use 10 (e.g., all symbol
  // probabilities are one of {1/1024, 2/1024, ..., 1023/1024, 1024/1024})
  int probBits;

  // If true, we calculate a checksum on the uncompressed input data to
  // compression and store it in the archive, and on the decompression side
  // post-decompression, we calculate a checksum on the decompressed data which
  // is compared with the original stored in the archive.
  // This is an optional feature useful if DietGPU data will be stored
  // persistently on disk.
  bool useChecksum;
};

enum class ANSDecodeError : uint32_t {
  None = 0,
  ChecksumMismatch = 1,
};

// Error status for decompression
struct ANSDecodeStatus {
  inline ANSDecodeStatus() : error(ANSDecodeError::None) {}

  // Overall error status
  ANSDecodeError error;

  // Error-specific information for the batch
  std::vector<std::pair<int, std::string>> errorInfo;
};

//
// Encode
//

void ansEncodeBatchStride(
    StackDeviceMemory& res,
    // Compression configuration
    const ANSCodecConfig& config,

    // Number of separate, independent compression problems
    uint32_t numInBatch,

    // Region in device memory of size at least
    // numInBatch * inPerBatchSize + max(numInBatch - 1, 0) * inPerBatchStride
    const void* in_dev,
    // Bytes per batch member for compression
    uint32_t inPerBatchSize,
    // Stride per separate input compression problem (must be >= inPerBatchSize)
    uint32_t inPerBatchStride,

    // Optional (can be null): region in device memory of size 256 words
    // containing pre-calculated symbol counts (histogram) of the data to be
    // compressed
    const uint32_t* histogram_dev,

    // Region in device memory of size at least
    // numInBatch * getMaxCompressedSize(inPerBatchSize) +
    //   max(numInBatch - 1, 0) * outPerBatchStride
    void* out_dev,
    // Stride per separate output compression problem, which must be
    // >= getMaxCompressedSize(inPerBatchSize)
    uint32_t outPerBatchStride,
    // Device memory array of size numInBatch (optional)
    // Provides the size of actual used memory in each output compressed batch
    uint32_t* outBatchSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

void ansEncodeBatchPointer(
    StackDeviceMemory& res,
    // Compression configuration
    const ANSCodecConfig& config,

    // Number of separate, independent compression problems
    uint32_t numInBatch,

    // Host array with addresses of device pointers comprising the input batch
    // to compress
    const void** in,
    // Host array with sizes of batch members
    const uint32_t* inSize,

    // Optional (can be null): region in device memory of size 256 words
    // containing pre-calculated symbol counts (histogram) of the data to be
    // compressed
    const uint32_t* histogram_dev,

    // Host array with addresses of device pointers for the compressed output
    // arrays. Each out[i] must be a region of memory of size at least
    // getMaxCompressedSize(inSize[i])
    void** out,
    // Device memory array of size numInBatch (optional)
    // Provides the size of actual used memory in each output compressed batch
    uint32_t* outSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

void ansEncodeBatchSplitSize(
    StackDeviceMemory& res,

    // Compression configuration
    const ANSCodecConfig& config,

    // Number of separate, independent compression problems
    uint32_t numInBatch,

    // Device pointer into a valid region of memory of size at least
    // sum_i(inSplitSizes[i]) bytes
    const void* in_dev,

    // Host array with the size (in bytes) of the input arrays in the batch.
    // Each array in the batch is read starting at offset inSplitSizes[i].
    const uint32_t* inSplitSizes,

    // Optional (can be null): region in device memory of size 256 words
    // containing pre-calculated symbol counts (histogram) of the data to be
    // compressed
    const uint32_t* histogram_dev,

    // Device pointer to a matrix of at least size
    // numInBatch x getMaxCompressedSize(max(inSplitSizes[i]))
    void* out_dev,

    // Stride between rows in bytes
    uint32_t outStride,

    // Device memory array of size numInBatch (optional)
    // Provides the size of actual used memory in each output compressed batch
    uint32_t* outSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

//
// Decode
//

ANSDecodeStatus ansDecodeBatchStride(
    StackDeviceMemory& res,

    // Expected compression configuration (we verify this upon decompression)
    const ANSCodecConfig& config,

    // Number of separate, independent decompression problems
    uint32_t numInBatch,

    // start of compressed input data (device pointer)
    const void* in_dev,
    // stride in in_dev between separate compressed inputs; e.g., the regions of
    // memory located at the following byte offset ranges from in_dev contain
    // the input per-batch element compressed data:
    //
    // [b * inPerBatchStride, b * inPerBatchStride + compressed_size[b] - 1]
    // where compressed_size[b] is the compressed size indicated in the header
    // metadata for each per-batch compressed data.
    //
    // The kernel will not access memory beyond the per-batch member compressed
    // size in each of these regions, thus the stride should be at least the
    // maximum of all of the individual per-batch compressed sizes.
    // If the stride is not sufficient, then the kernel may segfault.
    uint32_t inPerBatchStride,

    // start of decompressed output data (device pointer)
    void* out_dev,
    // Stride between each decompressed output, which must be greater than the
    // uncompressed size for each decompressed output and outPerBatchCapacity.
    uint32_t outPerBatchStride,
    // Overall space available for each decompression batch member; e.g., the
    // regions of memory located at the following byte offset ranges from
    // out_dev:
    //
    // [b * outPerBatchStride, b * outPerBatchStride + outPerBatchCapacity - 1]
    //
    // for all b \in [0, numInBatch - 1] are valid.
    // If the seen decompressed size for any individual batch member is less
    // than outBatchCapacity, that particular batch member will fail to
    // decompress, and the reported size for that batch member will be in
    // status_dev.
    uint32_t outPerBatchCapacity,

    // Decode success/fail status (optional, can be nullptr)
    // If present, this is a device pointer to an array of length numInBatch,
    // with true/false for whether or not decompression status was successful
    // FIXME: not bool due to issues with __nv_bool
    uint8_t* outSuccess_dev,

    // Decode size status (optional, can be nullptr)
    // If present, this is a device pointer to an array of length numInBatch,
    // with either the size decompressed reported if successful, or the required
    // size reported if our outPerBatchCapacity was insufficient
    uint32_t* outSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

ANSDecodeStatus ansDecodeBatchPointer(
    StackDeviceMemory& res,

    // Expected compression configuration (we verify this upon decompression)
    const ANSCodecConfig& config,

    // Number of separate, independent decompression problems
    uint32_t numInBatch,

    // Host array with addresses of device pointers corresponding to compressed
    // inputs
    const void** in,

    // Host array with addresses of device pointers corresponding to
    // uncompressed outputs
    void** out,

    // Host array with size of memory regions provided in out; if the seen
    // decompressed size is greater than this, then there will be an error in
    // decompression
    const uint32_t* outCapacity,

    // Decode success/fail status (optional, can be nullptr)
    // If present, this is a device pointer to an array of length numInBatch,
    // with true/false for whether or not decompression status was successful
    // FIXME: not bool due to issues with __nv_bool
    uint8_t* outSuccess_dev,

    // Decode size status (optional, can be nullptr)
    // If present, this is a device pointer to an array of length numInBatch,
    // with either the size decompressed reported if successful, or the required
    // size reported if our outPerBatchCapacity was insufficient
    uint32_t* outSize_dev,

    // stream on the current device on which this runs
    cudaStream_t stream);

ANSDecodeStatus ansDecodeBatchSplitSize(
    StackDeviceMemory& res,

    // Expected compression configuration (we verify this upon decompression)
    const ANSCodecConfig& config,

    // Number of separate, independent compression problems
    uint32_t numInBatch,

    // Host array with addresses of device pointers comprising the batch
    const void** in,

    // Device pointer into a valid region of memory of size at least
    // sum_i(outSplitSizes[i]) bytes
    void* out_dev,

    // Host array with the size (in bytes) of the output
    // decompressed arrays in the batch.
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

void ansGetCompressedInfo(
    StackDeviceMemory& res,
    // Host array with addresses of device pointers comprising the batch of
    // compressed ANS data
    const void** in,
    // Number of compressed arrays in the batch
    uint32_t numInBatch,
    // Optional device array to receive the resulting sizes. 0 is reported if
    // the compresed data is not as expected, otherwise the size is reported in
    // bytes
    uint32_t* outSizes_dev,
    // Optional device array to receive pre-compression checksums stored in the
    // archive, if the checksum feature was enabled.
    uint32_t* outChecksum_dev,
    // stream on the current device on which this runs
    cudaStream_t stream);

void ansGetCompressedInfoDevice(
    StackDeviceMemory& res,
    // Device array with addresses of device pointers comprising the batch of
    // compressed ANS data
    const void** in_dev,
    // Number of compressed arrays in the batch
    uint32_t numInBatch,
    // Optional device array to receive the resulting sizes. 0 is reported if
    // the compresed data is not as expected, otherwise the size is reported in
    // bytes
    uint32_t* outSizes_dev,
    // Optional device array to receive pre-compression checksums stored in the
    // archive, if the checksum feature was enabled.
    uint32_t* outChecksum_dev,
    // stream on the current device on which this runs
    cudaStream_t stream);

} // namespace dietgpu
