#pragma once
#include "define.h"
#include "memory_allocator.h"
/**
 * @brief Compressor class for handling GPU float compression and decompression.
 *
 * This class encapsulates dietgpu compression/decompression operations and manages
 * the associated GPU resources (CUDA streams, device memory, compression buffer).
 */
class Compressor {
 public:
  Compressor() : stream_(nullptr), res_(nullptr) {
    // Initialize compression buffer
    auto allocator = std::make_shared<MemoryAllocator>();
    buffer_ = allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

    // Initialize decompression buffer
    decompressBuffer_ = allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

    // Initialize GPU stream directly (not via CudaStream wrapper to avoid move issues)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    GPU_CHECK(hipStreamCreate(&stream_));
#else
    GPU_CHECK(cudaStreamCreate(&stream_));
#endif

    // Initialize StackDeviceMemory using placement new to avoid move/copy issues
    // Allocate on heap and manage manually
    res_ = new dietgpu::StackDeviceMemory(dietgpu::getCurrentDevice(),
                                          2*kCompressBufferSize); 
  }

  ~Compressor() {
    // Clean up stream
    if (stream_) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
      hipStreamDestroy(stream_);
#else
      cudaStreamDestroy(stream_);
#endif
      stream_ = nullptr;
    }

    // Clean up StackDeviceMemory
    if (res_) {
      delete res_;
      res_ = nullptr;
    }
  }

  // Non-copyable
  Compressor(const Compressor&) = delete;
  Compressor& operator=(const Compressor&) = delete;

  // Non-movable (due to raw pointer management)
  Compressor(Compressor&&) = delete;
  Compressor& operator=(Compressor&&) = delete;

  /**
   * @brief Get the compression buffer.
   * @return Shared pointer to the compression buffer RegMemBlock.
   */
  std::shared_ptr<RegMemBlock> getBuffer() const { return buffer_; }

  /**
   * @brief Get the decompression buffer.
   * @return Shared pointer to the decompression buffer RegMemBlock.
   */
  std::shared_ptr<RegMemBlock> getDecompressBuffer() const { return decompressBuffer_; }

  /**
   * @brief Register the compression buffer with a channel's memory region.
   * @param channel_id The channel ID for registration.
   * @param mr The memory region pointer to associate with this channel.
   */
  void registerBuffer(uint32_t channel_id, struct ibv_mr* mr) {
    if (buffer_) {
      buffer_->setMRByChannelID(channel_id, mr);
    }
  }

  /**
   * @brief Register the decompression buffer with a channel's memory region.
   * @param channel_id The channel ID for registration.
   * @param mr The memory region pointer to associate with this channel.
   */
  void registerDecompressBuffer(uint32_t channel_id, struct ibv_mr* mr) {
    if (decompressBuffer_) {
      decompressBuffer_->setMRByChannelID(channel_id, mr);
    }
  }

  /**
   * @brief Compress a send request's data.
   *
   * Compresses the data from req->local_mem into the internal compression buffer,
   * then updates req->local_mem to point to the compressed data.
   *
   * @param req The send request to compress. The request's local_mem will be
   *            updated to point to the compressed buffer on success.
   * @return true on success, false on failure.
   */
  bool compress(std::shared_ptr<RDMASendRequest> req) {
    if (!req || !req->local_mem || !stream_ || !res_ || !buffer_) {
      LOG(WARNING) << "Compressor::compress - Invalid parameters";
      return false;
    }

    // Setup compression config
    dietgpu::FloatCompressConfig compressConfig;
    compressConfig.floatType = req->float_type;
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    // Calculate element count from bytes
    uint32_t numFloats =
        getElementCountFromBytes(req->float_type, req->local_mem->size);

    // Setup batch (single element batch)
    const void* inPtrs[1] = {req->local_mem->addr};
    uint32_t inSizes[1] = {numFloats};
    void* outPtrs[1] = {buffer_->addr};

    // Allocate device memory for compressed size output
    uint32_t* devCompressedSize = nullptr;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    GPU_CHECK(hipMalloc(&devCompressedSize, sizeof(uint32_t)));
#else
    GPU_CHECK(cudaMalloc(&devCompressedSize, sizeof(uint32_t)));
#endif

    // Compress
    dietgpu::floatCompress(
        *res_,
        compressConfig,
        1,  // numInBatch
        inPtrs,
        inSizes,
        outPtrs,
        devCompressedSize,
        stream_);

    // Get compressed size
    uint32_t compressedSize = 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    GPU_CHECK(hipMemcpyAsync(&compressedSize, devCompressedSize,
                              sizeof(uint32_t), hipMemcpyDeviceToHost,
                              stream_));
    GPU_CHECK(hipStreamSynchronize(stream_));
#else
    GPU_CHECK(cudaMemcpyAsync(&compressedSize, devCompressedSize,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               stream_));
    GPU_CHECK(cudaStreamSynchronize(stream_));
#endif

    LOG(INFO) << "Compressor: Compressed " << req->local_mem->size
              << " bytes to " << compressedSize << " bytes, ratio: "
              <<  static_cast<float>(compressedSize)/static_cast<float>(req->local_mem->size) << "x";

    // Update request to use compressed buffer
    req->local_mem = std::make_shared<RegMemBlock>(
        buffer_->addr, compressedSize, buffer_->mr_array, buffer_->type);

    // Cleanup
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    GPU_CHECK(hipFree(devCompressedSize));
#else
    GPU_CHECK(cudaFree(devCompressedSize));
#endif

    return true;
  }

  /**
   * @brief Prepare a receive request for decompression.
   *
   * Backs up the original local_mem (which contains compressed data) to
   * local_compression_mem, so that decompress() can read from local_mem
   * and write to local_compression_mem.
   *
   * @param req The receive request to prepare. After this call:
   *            - req->local_compression_mem will hold the original local_mem
   *            - req->local_mem still points to the compressed data buffer
   * @return true on success, false on failure.
   */
  bool prepare_for_decompress(std::shared_ptr<RDMARecvRequest> req) {
    if (!req || !req->local_mem) {
      LOG(WARNING) << "Compressor::prepare_for_decompress - Invalid parameters";
      return false;
    }

    // Backup local_mem to local_compression_mem
    req->local_compression_mem = req->local_mem;
    req->local_mem = std::make_shared<RegMemBlock>(
        decompressBuffer_->addr, decompressBuffer_->size,
        decompressBuffer_->mr_array, decompressBuffer_->type);
    LOG(INFO) << "Compressor: Prepared for decompression, backed up local_mem ("
              << req->local_compression_mem->size << " bytes) to local_compression_mem";

    return true;
  }

  /**
   * @brief Decompress received data into the request's local_compression_mem.
   *
   * Decompresses data from req->local_mem (compressed data) into
   * req->local_compression_mem (decompressed output).
   * Call prepare_for_decompress() before this function to set up the buffers.
   *
   * @param req The receive request.
   *            - Input: req->local_mem (compressed data)
   *            - Output: req->local_compression_mem (decompressed data)
   * @return true on success, false on failure.
   */
  bool decompress(std::shared_ptr<RDMARecvRequest> req) {
    if (!req || !req->local_mem || !req->local_compression_mem || !stream_ || !res_) {
      LOG(WARNING) << "Compressor::decompress - Invalid parameters";
      return false;
    }

    // Setup decompression config
    dietgpu::FloatDecompressConfig decompressConfig;
    decompressConfig.floatType = req->float_type;
    decompressConfig.useChecksum = false;
    decompressConfig.is16ByteAligned = true;

    // Calculate element count from bytes for output capacity
    // Use local_compression_mem size as it represents the original uncompressed size
    uint32_t numFloats =
        getElementCountFromBytes(req->float_type, req->local_compression_mem->size);

    // Setup batch for decompression
    // Input is the compressed data in local_mem
    // Output is the local_compression_mem buffer
    const void* compInPtrs[1] = {req->local_mem->addr};
    void* decompOutPtrs[1] = {req->local_compression_mem->addr};
    uint32_t outCapacities[1] = {numFloats};

    // Decompress
    dietgpu::FloatDecompressStatus status = dietgpu::floatDecompress(
        *res_,
        decompressConfig,
        1,  // numInBatch
        compInPtrs,
        decompOutPtrs,
        outCapacities,
        nullptr,  // outSuccess_dev (optional)
        nullptr,  // outSize_dev (optional)
        stream_);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    GPU_CHECK(hipStreamSynchronize(stream_));
#else
    GPU_CHECK(cudaStreamSynchronize(stream_));
#endif

    if (status.error != dietgpu::FloatDecompressError::None) {
      LOG(ERROR) << "Compressor: Decompression failed!";
      return false;
    }

    LOG(INFO) << "Compressor: Decompressed data from " << req->local_mem->size
              << " bytes to " << req->local_compression_mem->size << " bytes";

    return true;
  }

  /**
   * @brief Decompress data from RemoteMemInfo to RegMemBlock.
   *
   * Decompresses data from the input buffer (referenced by RemoteMemInfo) into
   * the output buffer (referenced by RegMemBlock).
   *
   * @param input The RemoteMemInfo containing compressed data address and length.
   * @param output The RegMemBlock to write decompressed data to.
   * @param float_type The float type used for decompression configuration.
   * @return true on success, false on failure.
   */
  bool decompress(const RemoteMemInfo& input, RegMemBlock& output,
                  dietgpu::FloatType float_type) {
    if (!stream_ || !res_) {
      LOG(WARNING) << "Compressor::decompress - Invalid internal state";
      return false;
    }

    if (input.addr == 0 || input.length == 0) {
      LOG(WARNING) << "Compressor::decompress - Invalid input parameters";
      return false;
    }

    if (output.addr == nullptr || output.size == 0) {
      LOG(WARNING) << "Compressor::decompress - Invalid output parameters";
      return false;
    }

    // Setup decompression config
    dietgpu::FloatDecompressConfig decompressConfig;
    decompressConfig.floatType = float_type;
    decompressConfig.useChecksum = false;
    decompressConfig.is16ByteAligned = true;

    // Calculate element count from bytes for output capacity
    uint32_t numFloats = getElementCountFromBytes(float_type, output.size);

    // Setup batch for decompression
    // Input is the compressed data from RemoteMemInfo
    // Output is the RegMemBlock buffer
    const void* compInPtrs[1] = {reinterpret_cast<const void*>(input.addr)};
    void* decompOutPtrs[1] = {output.addr};
    uint32_t outCapacities[1] = {numFloats};

    // Decompress
    dietgpu::FloatDecompressStatus status = dietgpu::floatDecompress(
        *res_,
        decompressConfig,
        1,  // numInBatch
        compInPtrs,
        decompOutPtrs,
        outCapacities,
        nullptr,  // outSuccess_dev (optional)
        nullptr,  // outSize_dev (optional)
        stream_);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    GPU_CHECK(hipStreamSynchronize(stream_));
#else
    GPU_CHECK(cudaStreamSynchronize(stream_));
#endif

    if (status.error != dietgpu::FloatDecompressError::None) {
      LOG(ERROR) << "Compressor: Decompression failed!";
      return false;
    }

    LOG(INFO) << "Compressor: Decompressed data from " << input.length
              << " bytes to " << output.size << " bytes";

    return true;
  }

  /**
   * @brief Check if a request should be compressed based on size threshold.
   * @param size The size in bytes to check.
   * @return true if the size meets the minimum compression threshold.
   */
  static bool shouldCompress(size_t size) { return size >= kMinCompressBytes; }

 private:
  std::shared_ptr<RegMemBlock> buffer_;                   // Compression buffer
  std::shared_ptr<RegMemBlock> decompressBuffer_;         // Decompression buffer
  GpuStream_t stream_;                                    // GPU stream (hipStream_t or cudaStream_t)
  dietgpu::StackDeviceMemory* res_;                       // Device memory manager (raw pointer to avoid move issues)
};
