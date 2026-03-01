#pragma once
#include "define.h"
#include "memory_allocator.h"

/**
 * @brief Abstract interface for compressor backends.
 *
 * This pure virtual interface defines the compression/decompression operations
 * without any DietGPU-specific types. Implementations can use DietGPU or
 * provide a null/stub implementation.
 */
class ICompressorBackend {
 public:
  virtual ~ICompressorBackend() = default;

  /**
   * @brief Get the compression buffer.
   * @return Shared pointer to the compression buffer RegMemBlock.
   */
  virtual std::shared_ptr<RegMemBlock> getCompressBuffer() const = 0;

  /**
   * @brief Get the decompression buffer.
   * @return Shared pointer to the decompression buffer RegMemBlock.
   */
  virtual std::shared_ptr<RegMemBlock> getDecompressBuffer() const = 0;

  /**
   * @brief Compress a send request's data.
   * @param req The send request to compress.
   * @return true on success, false on failure.
   */
  virtual bool compress(std::shared_ptr<RDMASendRequest> req) = 0;

  /**
   * @brief Prepare a receive request for decompression.
   * @param req The receive request to prepare.
   * @return true on success, false on failure.
   */
  virtual bool prepareDecompress(std::shared_ptr<RDMARecvRequest> req) = 0;

  /**
   * @brief Decompress data from RemoteMemInfo to RegMemBlock.
   * @param input The RemoteMemInfo containing compressed data.
   * @param output The RegMemBlock to write decompressed data to.
   * @param float_type The float type.
   * @return true on success, false on failure.
   */
  virtual bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                          uccl::FloatType float_type) = 0;

  /**
   * @brief Check if a request should be compressed based on size threshold.
   * @param size The size in bytes to check.
   * @return true if the size meets the minimum compression threshold.
   */
  virtual bool shouldCompress(size_t size) = 0;

  /**
   * @brief Check if data should be compressed and split first.
   * @param size The size in bytes to check.
   * @return true if should compress and split first.
   */
  virtual bool shouldCompressAndSplitFirst(size_t size) = 0;

  /**
   * @brief Prepare a context for split+encode two-phase compression.
   * @param addr Device pointer to the input float data.
   * @param size Size of the input data in bytes.
   * @param ctx Compression context.
   */
  virtual void prepareSplitContext(void* addr, size_t size,
                                   CompressCtx ctx) = 0;

  /**
   * @brief Phase 1 of two-phase compression: split float data.
   * @param req The send request with compress_ctx already prepared.
   * @return true on success, false on failure.
   */
  virtual bool compressSplitOneBatch(std::shared_ptr<RDMASendRequest> req) = 0;

  /**
   * @brief Phase 2 of two-phase compression: ANS encode.
   * @param req The send request with compress_ctx already split.
   * @return compressed size on success, 0 on failure.
   */
  virtual uint32_t compressEncodeOneBatch(
      std::shared_ptr<RDMASendRequest> req) = 0;

  /**
   * @brief Get the compression strategy.
   * @return The current compression strategy.
   */
  virtual CompressStrategy getCompressStrategy() = 0;
};

/**
 * @brief Null compressor backend that provides no-op implementations.
 *
 * This backend is used when DietGPU is not available. All compression
 * operations return false or no-op results.
 */
class NullCompressorBackend : public ICompressorBackend {
 public:
  NullCompressorBackend() : compress_strategy_(CompressStrategy::kNone) {}
  ~NullCompressorBackend() override = default;

  std::shared_ptr<RegMemBlock> getCompressBuffer() const override { return nullptr; }

  std::shared_ptr<RegMemBlock> getDecompressBuffer() const override {
    return nullptr;
  }

  bool compress(std::shared_ptr<RDMASendRequest> /*req*/) override {
    return false;
  }

  bool prepareDecompress(std::shared_ptr<RDMARecvRequest> /*req*/) override {
    return false;
  }

  bool decompress(RemoteMemInfo const& /*input*/, RegMemBlock& /*output*/,
                  uccl::FloatType /*float_type*/) override {
    return false;
  }

  bool shouldCompress(size_t /*size*/) override { return false; }

  bool shouldCompressAndSplitFirst(size_t /*size*/) override { return false; }

  void prepareSplitContext(void* /*addr*/, size_t /*size*/,
                           CompressCtx /*ctx*/) override {}

  bool compressSplitOneBatch(
      std::shared_ptr<RDMASendRequest> /*req*/) override {
    return false;
  }

  uint32_t compressEncodeOneBatch(
      std::shared_ptr<RDMASendRequest> /*req*/) override {
    return 0;
  }

  CompressStrategy getCompressStrategy() override { return compress_strategy_; }

 private:
  CompressStrategy compress_strategy_;
};

#ifdef USE_DIETGPU

/**
 * @brief DietGPU-based compressor backend.
 *
 * This backend uses DietGPU library for GPU float compression and
 * decompression.
 */
class DietGPUCompressorBackend : public ICompressorBackend {
 public:
  DietGPUCompressorBackend()
      : stream_(nullptr),
        res_(nullptr),
        res_meta_(nullptr),
        devCompressedSize_(nullptr) {
    compress_strategy_ = getCompressStrategyFromEnv();
    if (compress_strategy_ == CompressStrategy::kNone) {
      return;
    }
    GPU_RT_CHECK(gpuMalloc(&devCompressedSize_, sizeof(uint32_t)));
    // Initialize compression buffer
    auto allocator = std::make_shared<MemoryAllocator>();
    buffer_ =
        allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

    // Initialize decompression buffer
    decompressBuffer_ =
        allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

    // Initialize GPU stream
    GPU_RT_CHECK(gpuStreamCreate(&stream_));

    // Initialize StackDeviceMemory for compress/decompress operations
    res_ = new dietgpu::StackDeviceMemory(dietgpu::getCurrentDevice(),
                                          3 * kCompressBufferSize);

    // Initialize StackDeviceMemory for split context metadata allocations
    res_meta_ = new dietgpu::StackDeviceMemory(dietgpu::getCurrentDevice(),
                                               kCompressBufferSize);
  }

  ~DietGPUCompressorBackend() override {
    if (stream_) {
      gpuStreamDestroy(stream_);
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
      GPU_RT_CHECK(gpuFree(devCompressedSize_));
    }
  }

  std::shared_ptr<RegMemBlock> getCompressBuffer() const override { return buffer_; }

  std::shared_ptr<RegMemBlock> getDecompressBuffer() const override {
    return decompressBuffer_;
  }

  bool compress(std::shared_ptr<RDMASendRequest> req) override {
    if (unlikely(!req || !req->local_mem || !stream_ || !res_ || !buffer_)) {
      LOG(WARNING) << "DietGPUCompressorBackend::compress - Invalid parameters";
      return false;
    }
    // Setup compression config
    dietgpu::FloatCompressConfig compressConfig;
    compressConfig.floatType = to_dietgpu(req->compress_ctx->getFloatType());
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    // Calculate element count from bytes
    uint32_t numFloats = dietgpu::getElementCountFromBytes(
        to_dietgpu(req->compress_ctx->getFloatType()), req->local_mem->size);

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
    GPU_RT_CHECK(gpuMemcpy(&compressedSize, devCompressedSize_,
                           sizeof(compressedSize), gpuMemcpyDeviceToHost));

    LOG(INFO) << "DietGPUCompressorBackend: Compressed " << req->local_mem->size
              << " bytes to " << compressedSize << " bytes, ratio: "
              << static_cast<float>(compressedSize) /
                     static_cast<float>(req->local_mem->size)
              << "x";

    // Update request to use compressed buffer
    req->local_mem = std::make_shared<RegMemBlock>(
        buffer_->addr, compressedSize, buffer_->mr_array, buffer_->type);
    return true;
  }

  bool prepareDecompress(std::shared_ptr<RDMARecvRequest> req) override {
    if (unlikely(!req || !req->local_mem)) {
      LOG(WARNING)
          << "DietGPUCompressorBackend::prepareDecompress - Invalid parameters";
      return false;
    }

    // Backup local_mem to local_compression_mem
    req->local_compression_mem = req->local_mem;
    req->local_mem = std::make_shared<RegMemBlock>(
        decompressBuffer_->addr, decompressBuffer_->size,
        decompressBuffer_->mr_array, decompressBuffer_->type);
    LOG(INFO) << "DietGPUCompressorBackend: Prepared for decompression, backed "
                 "up local_mem ("
              << req->local_compression_mem->size
              << " bytes) to local_compression_mem";

    return true;
  }

  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                  uccl::FloatType float_type) override {
    if (unlikely(!stream_ || !res_)) {
      LOG(WARNING)
          << "DietGPUCompressorBackend::decompress - Invalid internal state";
      return false;
    }

    if (unlikely(input.addr == 0 || input.length == 0)) {
      LOG(WARNING)
          << "DietGPUCompressorBackend::decompress - Invalid input parameters";
      return false;
    }

    if (unlikely(output.addr == nullptr || output.size == 0)) {
      LOG(WARNING)
          << "DietGPUCompressorBackend::decompress - Invalid output parameters";
      return false;
    }

    dietgpu::FloatType ft = to_dietgpu(float_type);

    // Setup decompression config
    dietgpu::FloatDecompressConfig decompressConfig;
    decompressConfig.floatType = ft;
    decompressConfig.useChecksum = false;
    decompressConfig.is16ByteAligned = true;

    // Calculate element count from bytes for output capacity
    uint32_t numFloats = dietgpu::getElementCountFromBytes(ft, output.size);

    // Setup batch for decompression
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

    GPU_RT_CHECK(gpuStreamSynchronize(stream_));

    if (unlikely(status.error != dietgpu::FloatDecompressError::None)) {
      LOG(ERROR) << "DietGPUCompressorBackend: Decompression failed!";
      return false;
    }

    LOG(INFO) << "DietGPUCompressorBackend: Decompressed data from "
              << input.length << " bytes to " << output.size << " bytes";

    return true;
  }

  bool shouldCompress(size_t size) override {
    if (compress_strategy_ == CompressStrategy::kNone) {
      return false;
    }
    if (unlikely(size > buffer_->size)) {
      LOG(ERROR) << "DietGPUCompressorBackend::shouldCompress: data size ("
                 << size << " bytes) exceeds compress buffer capacity ("
                 << buffer_->size << " bytes), skipping compression";
      return false;
    }
    return size >= kMinCompressBytes;
  }

  bool shouldCompressAndSplitFirst(size_t size) override {
    return compress_strategy_ == CompressStrategy::kSplitOnly && shouldCompress(size);
  }

  void prepareSplitContext(void* addr, size_t size, CompressCtx ctx) override {
    if (unlikely(ctx == nullptr)) {
      std::cout << "unlikely(ctx == nullptr)" << std::endl;
      return;
    }
    if (unlikely(!shouldCompress(size))) {
      return;
    }
    dietgpu::FloatType float_type = to_dietgpu(ctx->getFloatType());
    uint32_t numFloats =
        static_cast<uint32_t>(dietgpu::getElementCountFromBytes(float_type, size));

    // Build params_dev on device: layout is [in_ptr, inSize, out_ptr]
    uintptr_t hostParams[3];
    hostParams[0] = reinterpret_cast<uintptr_t>(addr);
    hostParams[1] = static_cast<uintptr_t>(numFloats);
    hostParams[2] = reinterpret_cast<uintptr_t>(buffer_->addr);

    auto devParams = res_meta_->alloc<uintptr_t>(stream_, 3);
    GPU_RT_CHECK(gpuMemcpyAsync(devParams.data(), hostParams,
                                sizeof(hostParams), gpuMemcpyHostToDevice,
                                stream_));
    GPU_RT_CHECK(gpuStreamSynchronize(stream_));

    ctx->params_dev = std::move(devParams);
    ctx->maxSize = size;
  }

  bool compressSplitOneBatch(std::shared_ptr<RDMASendRequest> req) override {
    if (unlikely(!req || !req->compress_ctx || !stream_ || !res_)) {
      LOG(WARNING) << "DietGPUCompressorBackend::compressSplitOneBatch - "
                      "Invalid parameters";
      return false;
    }

    dietgpu::FloatCompressConfig compressConfig;
    compressConfig.floatType = to_dietgpu(req->compress_ctx->getFloatType());
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    dietgpu::floatCompressSplitOneBatch(*res_, compressConfig, stream_,
                                        *req->compress_ctx);

    int32_t data_size = dietgpu::roundDown(
        getUncompDataSizeFromByteSize(compressConfig.floatType,
                                      req->compress_ctx->maxSize),
        ChunkSplitStrategy::kMessageChunkSizeKB);
    req->local_mem = std::make_shared<RegMemBlock>(
        buffer_->addr, data_size, buffer_->mr_array, buffer_->type);
    return true;
  }

  uint32_t compressEncodeOneBatch(
      std::shared_ptr<RDMASendRequest> req) override {
    if (unlikely(!req || !req->compress_ctx || !stream_ || !res_ || !buffer_)) {
      LOG(WARNING) << "DietGPUCompressorBackend::compressEncodeOneBatch - "
                      "Invalid parameters";
      return 0;
    }

    dietgpu::FloatCompressConfig compressConfig;
    compressConfig.floatType = to_dietgpu(req->compress_ctx->getFloatType());
    compressConfig.useChecksum = false;
    compressConfig.is16ByteAligned = true;

    dietgpu::floatCompressEncodeOneBatch(
        *res_, compressConfig, *req->compress_ctx, devCompressedSize_, stream_);

    // Get compressed size
    uint32_t compressedSize = 0;
    GPU_RT_CHECK(gpuMemcpy(&compressedSize, devCompressedSize_,
                           sizeof(compressedSize), gpuMemcpyDeviceToHost));

    LOG(INFO) << "DietGPUCompressorBackend: Encode done, "
              << req->local_mem->size << " bytes -> " << compressedSize
              << " bytes, ratio: "
              << static_cast<float>(compressedSize) /
                     static_cast<float>(req->local_mem->size)
              << "x";

    // Update request to use compressed buffer
    uint32_t uncompressed_size = req->local_mem->size;
    req->local_mem = std::make_shared<RegMemBlock>(
        static_cast<char*>(buffer_->addr) + uncompressed_size,
        compressedSize - uncompressed_size, buffer_->mr_array, buffer_->type);
    req->remote_mem->addr = req->remote_mem->addr + uncompressed_size;
    req->compress_ctx->histogram_dev.release();
    req->compress_ctx->toComp_dev.release();
    return compressedSize;
  }

  CompressStrategy getCompressStrategy() override { return compress_strategy_; }

 private:
  uint32_t* devCompressedSize_;
  std::shared_ptr<RegMemBlock> buffer_;            // Compression buffer
  std::shared_ptr<RegMemBlock> decompressBuffer_;  // Decompression buffer
  gpuStream_t stream_;                             // GPU stream
  dietgpu::StackDeviceMemory* res_;  // Device memory for compress/decompress
  dietgpu::StackDeviceMemory* res_meta_;  // Device memory for split context
  CompressStrategy compress_strategy_;
};

#endif  // USE_DIETGPU

/**
 * @brief Compressor class for handling GPU float compression and decompression.
 *
 * This class is a facade that delegates to the appropriate backend based on
 * compile-time configuration (USE_DIETGPU). It manages a singleton instance
 * and forwards all operations to the selected backend.
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
  std::shared_ptr<RegMemBlock> getCompressBuffer() const {
    return backend_->getCompressBuffer();
  }

  /**
   * @brief Get the decompression buffer.
   * @return Shared pointer to the decompression buffer RegMemBlock.
   */
  std::shared_ptr<RegMemBlock> getDecompressBuffer() const {
    return backend_->getDecompressBuffer();
  }

  /**
   * @brief Compress a send request's data.
   * @param req The send request to compress.
   * @return true on success, false on failure.
   */
  bool compress(std::shared_ptr<RDMASendRequest> req) {
    return backend_->compress(req);
  }

  /**
   * @brief Prepare a receive request for decompression.
   * @param req The receive request to prepare.
   * @return true on success, false on failure.
   */
  bool prepareDecompress(std::shared_ptr<RDMARecvRequest> req) {
    return backend_->prepareDecompress(req);
  }

  /**
   * @brief Decompress data from RemoteMemInfo to RegMemBlock.
   * @param input The RemoteMemInfo containing compressed data.
   * @param output The RegMemBlock to write decompressed data to.
   * @param float_type The float type as uint32_t.
   * @return true on success, false on failure.
   */
  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                  uccl::FloatType float_type) {
    return backend_->decompress(input, output, float_type);
  }

  /**
   * @brief Check if a request should be compressed based on size threshold.
   * @param size The size in bytes to check.
   * @return true if the size meets the minimum compression threshold.
   */
  bool shouldCompress(size_t size) { return backend_->shouldCompress(size); }

  /**
   * @brief Check if data should be compressed and split first.
   * @param size The size in bytes to check.
   * @return true if should compress and split first.
   */
  bool shouldCompressAndSplitFirst(size_t size) {
    return backend_->shouldCompressAndSplitFirst(size);
  }

  /**
   * @brief Prepare a context for split+encode two-phase compression.
   * @param addr Device pointer to the input float data.
   * @param size Size of the input data in bytes.
   * @param ctx Compression context.
   */
  void prepareSplitContext(void* addr, size_t size, CompressCtx ctx) {
    backend_->prepareSplitContext(addr, size, ctx);
  }

  /**
   * @brief Phase 1 of two-phase compression: split float data.
   * @param req The send request with compress_ctx already prepared.
   * @return true on success, false on failure.
   */
  bool compressSplitOneBatch(std::shared_ptr<RDMASendRequest> req) {
    return backend_->compressSplitOneBatch(req);
  }

  /**
   * @brief Phase 2 of two-phase compression: ANS encode.
   * @param req The send request with compress_ctx already split.
   * @return compressed size on success, 0 on failure.
   */
  uint32_t compressEncodeOneBatch(std::shared_ptr<RDMASendRequest> req) {
    return backend_->compressEncodeOneBatch(req);
  }

  /**
   * @brief Get the compression strategy.
   * @return The current compression strategy.
   */
  CompressStrategy getCompressStrategy() {
    return backend_->getCompressStrategy();
  }

 private:
  Compressor() {
#ifdef USE_DIETGPU
    backend_ = std::make_unique<DietGPUCompressorBackend>();
#else
    backend_ = std::make_unique<NullCompressorBackend>();
#endif
  }

  ~Compressor() = default;

  std::unique_ptr<ICompressorBackend> backend_;
};
