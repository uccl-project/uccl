#pragma once
#include "define.h"
#include "memory_allocator.h"

/**
 * @brief Compressor class for handling GPU float compression and decompression.
 *
 * This class encapsulates dietgpu compression/decompression operations and
 * manages the associated GPU resources (CUDA streams, device memory,
 * compression buffer).
 */

class Compressor {
 public:
  static Compressor& getInstance() {
    static Compressor instance;
    return instance;
  }

  // Non-copyable, non-movable
  Compressor(Compressor const&) = delete;
  Compressor& operator=(Compressor const&) = delete;
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
  std::shared_ptr<RegMemBlock> getDecompressBuffer() const {
    return decompressBuffer_;
  }

  /**
   * @brief Compress a send request's data.
   *
   * Compresses the data from req->local_mem into the internal compression
   * buffer, then updates req->local_mem to point to the compressed data.
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
    compressConfig.floatType = req->compress_ctx->float_type;
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    // Calculate element count from bytes
    uint32_t numFloats = getElementCountFromBytes(req->compress_ctx->float_type,
                                                  req->local_mem->size);

    // Setup batch (single element batch)
    void const* inPtrs[1] = {req->local_mem->addr};
    uint32_t inSizes[1] = {numFloats};
    void* outPtrs[1] = {buffer_->addr};

    // Compress
    dietgpu::floatCompress(*res_, compressConfig,
                           1,  // numInBatch
                           inPtrs, inSizes, outPtrs, devCompressedSize_,
                           stream_);

    // Get compressed size
    uint32_t compressedSize = 0;
    copyDeviceToHostSync(&compressedSize, devCompressedSize_, stream_);

    LOG(INFO) << "Compressor: Compressed " << req->local_mem->size
              << " bytes to " << compressedSize << " bytes, ratio: "
              << static_cast<float>(compressedSize) /
                     static_cast<float>(req->local_mem->size)
              << "x";

    // Update request to use compressed buffer
    req->local_mem = std::make_shared<RegMemBlock>(
        buffer_->addr, compressedSize, buffer_->mr_array, buffer_->type);
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
              << req->local_compression_mem->size
              << " bytes) to local_compression_mem";

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
    if (!req || !req->local_mem || !req->local_compression_mem || !stream_ ||
        !res_) {
      LOG(WARNING) << "Compressor::decompress - Invalid parameters";
      return false;
    }

    // Setup decompression config
    dietgpu::FloatDecompressConfig decompressConfig;
    decompressConfig.floatType = req->compress_ctx->float_type;
    decompressConfig.useChecksum = false;
    decompressConfig.is16ByteAligned = true;

    // Calculate element count from bytes for output capacity
    // Use local_compression_mem size as it represents the original uncompressed
    // size
    uint32_t numFloats = getElementCountFromBytes(
        req->compress_ctx->float_type, req->local_compression_mem->size);

    // Setup batch for decompression
    // Input is the compressed data in local_mem
    // Output is the local_compression_mem buffer
    void const* compInPtrs[1] = {req->local_mem->addr};
    void* decompOutPtrs[1] = {req->local_compression_mem->addr};
    uint32_t outCapacities[1] = {numFloats};

    // Decompress
    dietgpu::FloatDecompressStatus status =
        dietgpu::floatDecompress(*res_, decompressConfig,
                                 1,  // numInBatch
                                 compInPtrs, decompOutPtrs, outCapacities,
                                 nullptr,  // outSuccess_dev (optional)
                                 nullptr,  // outSize_dev (optional)
                                 stream_);

    GPU_CHECK(GPU_STREAM_SYNC(stream_));

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
   * @param input The RemoteMemInfo containing compressed data address and
   * length.
   * @param output The RegMemBlock to write decompressed data to.
   * @param float_type The float type used for decompression configuration.
   * @return true on success, false on failure.
   */
  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
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
    void const* compInPtrs[1] = {reinterpret_cast<void const*>(input.addr)};
    void* decompOutPtrs[1] = {output.addr};
    uint32_t outCapacities[1] = {numFloats};

    // Decompress
    dietgpu::FloatDecompressStatus status =
        dietgpu::floatDecompress(*res_, decompressConfig,
                                 1,  // numInBatch
                                 compInPtrs, decompOutPtrs, outCapacities,
                                 nullptr,  // outSuccess_dev (optional)
                                 nullptr,  // outSize_dev (optional)
                                 stream_);

    GPU_CHECK(GPU_STREAM_SYNC(stream_));

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
  bool shouldCompress(size_t size) {
    return size >= kMinCompressBytes &&
           compress_strategy_ != CompressStrategy::kNone;
  }

  bool shouldCompressAndSplitFirst(size_t size){
    return compress_strategy_==CompressStrategy::kSplitOnly && size >= kMinCompressBytes;
  }

  /**
   * @brief Prepare a FloatCompressSplitContext for split+encode two-phase
   * compression.
   *
   * Builds the params_dev layout [in_ptr, inSize, out_ptr] on device using
   * res_meta_, and returns a shared_ptr to a FloatCompressSplitContext ready
   * for floatCompressSplitOneBatch / floatCompressEncodeOneBatch.
   *
   * @param addr  Device pointer to the input float data.
   * @param size  Size of the input data in bytes.
   * @param key   Float type as uint32_t (cast of dietgpu::FloatType).
   * @return shared_ptr to the prepared context.
   */
  void prepareSplitContext(
      void* addr, size_t size,
      std::shared_ptr<dietgpu::FloatCompressSplitContext> ctx) {
    if (unlikely(ctx == nullptr)) {
      std::cout<<"unlikely(ctx == nullptr)"<<std::endl;
      return;
    }
    if (!shouldCompress(size)){
      return;
    }
    auto float_type = ctx->float_type;
    uint32_t numFloats =
        static_cast<uint32_t>(getElementCountFromBytes(float_type, size));

    // Build params_dev on device: layout is [in_ptr, inSize, out_ptr]
    uintptr_t hostParams[3];
    hostParams[0] = reinterpret_cast<uintptr_t>(addr);
    hostParams[1] = static_cast<uintptr_t>(numFloats);
    hostParams[2] = reinterpret_cast<uintptr_t>(buffer_->addr);

    auto devParams = res_meta_->alloc<uintptr_t>(stream_, 3);
    GPU_CHECK(GPU_MEMCPY_ASYNC(devParams.data(), hostParams, sizeof(hostParams),
                               GPU_MEMCPY_H2D, stream_));
    GPU_CHECK(GPU_STREAM_SYNC(stream_));

    ctx->params_dev = std::move(devParams);
    ctx->maxSize = size;
  }

  /**
   * @brief Phase 1 of two-phase compression: split float data.
   *
   * Assumes req->compress_ctx has been prepared via prepareSplitContext().
   * Runs the split kernel that extracts compressible symbols and builds
   * the histogram. After this call, ctx is ready for compressEncodeOneBatch().
   *
   * @param req The send request with compress_ctx already prepared.
   * @return true on success, false on failure.
   */
  bool compressSplitOneBatch(std::shared_ptr<RDMASendRequest> req) {
    // std::cout<<"res_:: "<<res_->getSizeAvailable()<<std::endl;
    // std::cout<<"res_meta_:: "<<res_meta_->getSizeAvailable()<<std::endl;
    if (!req || !req->compress_ctx || !stream_ || !res_) {
      LOG(WARNING) << "Compressor::compressSplitOneBatch - Invalid parameters";
      return false;
    }

    dietgpu::FloatCompressConfig compressConfig;
    compressConfig.floatType = req->compress_ctx->float_type;
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    dietgpu::floatCompressSplitOneBatch(
        *res_, compressConfig, stream_, *req->compress_ctx);
    
    int32_t data_size = dietgpu::roundDown(getUncompDataSizeFromByteSize(compressConfig.floatType,req->compress_ctx->maxSize),kMessageChunkSizeKB);
    req->local_mem = std::make_shared<RegMemBlock>(
        buffer_->addr, data_size, buffer_->mr_array, buffer_->type);
    return true;
  }

  /**
   * @brief Phase 2 of two-phase compression: ANS encode.
   *
   * Must be called after compressSplitOneBatch(). Runs the ANS entropy
   * encoding on the split data, produces the compressed output, and
   * updates req->local_mem to point to the compressed buffer.
   *
   * @param req The send request with compress_ctx already split.
   * @return true on success, false on failure.
   */
  uint32_t compressEncodeOneBatch(std::shared_ptr<RDMASendRequest> req) {
    if (!req || !req->compress_ctx || !stream_ || !res_ || !buffer_) {
      LOG(WARNING) << "Compressor::compressEncodeOneBatch - Invalid parameters";
      return false;
    }

    dietgpu::FloatCompressConfig compressConfig;
    compressConfig.floatType = req->compress_ctx->float_type;
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    dietgpu::floatCompressEncodeOneBatch(
        *res_, compressConfig, *req->compress_ctx, devCompressedSize_, stream_);

    // Get compressed size
    uint32_t compressedSize = 0;
    copyDeviceToHostSync(&compressedSize, devCompressedSize_, stream_);

    LOG(INFO) << "Compressor: Encode done, " << req->local_mem->size
              << " bytes -> " << compressedSize << " bytes, ratio: "
              << static_cast<float>(compressedSize) /
                     static_cast<float>(req->local_mem->size)
              << "x";

    // Update request to use compressed buffer
    uint32_t uncompressed_size = req->local_mem->size;
    // std::cout<<"req->compress_ctx->maxSize::"<<req->compress_ctx->maxSize<<std::endl;
    // std::cout<<"uncompressed_size::"<<uncompressed_size<<std::endl;
    req->local_mem = std::make_shared<RegMemBlock>(
        static_cast<char*>(buffer_->addr) + uncompressed_size, compressedSize - uncompressed_size, buffer_->mr_array, buffer_->type);
    req->remote_mem->addr = req->remote_mem->addr + uncompressed_size;
    req->compress_ctx->histogram_dev.release();
    req->compress_ctx->toComp_dev.release();
    return compressedSize;
  }
  CompressStrategy getCompressStrategy(){
    return compress_strategy_;
  }

 private:
  Compressor() : stream_(nullptr), res_(nullptr), res_meta_(nullptr) {
    compress_strategy_ = getCompressStrategyFromEnv();
    if (compress_strategy_ == CompressStrategy::kNone) {
      return;
    }
    GPU_CHECK(GPU_MALLOC(&devCompressedSize_, sizeof(uint32_t)));
    // Initialize compression buffer
    auto allocator = std::make_shared<MemoryAllocator>();
    buffer_ =
        allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

    // Initialize decompression buffer
    decompressBuffer_ =
        allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

    // Initialize GPU stream
    GPU_CHECK(GPU_STREAM_CREATE(&stream_));

    // Initialize StackDeviceMemory for compress/decompress operations
    res_ = new dietgpu::StackDeviceMemory(dietgpu::getCurrentDevice(),
                                          3 * kCompressBufferSize);

    // Initialize StackDeviceMemory for split context metadata allocations
    res_meta_ = new dietgpu::StackDeviceMemory(dietgpu::getCurrentDevice(),
                                               kCompressBufferSize);
  }

  ~Compressor() {
    if (stream_) {
      GPU_STREAM_DESTROY(stream_);
      stream_ = nullptr;
    }
    if (res_) {
      delete res_;
      res_ = nullptr;
    }
    if (res_meta_) {
      delete res_meta_;
      res_meta_ = nullptr;
    }
    if (devCompressedSize_) {
      GPU_CHECK(GPU_FREE(devCompressedSize_));
    }
  }

  uint32_t* devCompressedSize_ = nullptr;
  std::shared_ptr<RegMemBlock> buffer_;            // Compression buffer
  std::shared_ptr<RegMemBlock> decompressBuffer_;  // Decompression buffer
  GpuStream_t stream_;                             // GPU stream
  dietgpu::StackDeviceMemory* res_;  // Device memory for compress/decompress
  dietgpu::StackDeviceMemory*
      res_meta_;  // Device memory for split context allocations
  CompressStrategy compress_strategy_;
};
