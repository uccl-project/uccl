#pragma once
#include "common.h"
#include "memory_allocator.h"
#include "util/gpu_rt.h"
#include <cstddef>
#include <cstdint>

size_t getMinCompressBytesFromEnv();
extern size_t const& kMinCompressBytes;

size_t getCompressBufferBytesFromEnv();
extern size_t const& kCompressBufferSize;

enum class CompressStrategy {
  kNone,         // no compression
  kSplitOnly,    // only split, no encode
  kSplitEncode,  // split + encode (default)
};

CompressStrategy getCompressStrategyFromEnv();

#if defined USE_DIETGPU
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || \
    defined(__HIPCC__)
#include "dietgpu/float/GpuFloatCodec_hip.h"
#include "dietgpu/utils/DeviceUtils_hip.h"
#include "dietgpu/utils/GreenContext_hip.h"
#include "dietgpu/utils/StackDeviceMemory_hip.h"
#include <hip/hip_fp16.h>
#else
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/GreenContext.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include <cuda_fp16.h>
#endif

dietgpu::FloatType to_dietgpu(uccl::FloatType t);
uccl::FloatType from_dietgpu(dietgpu::FloatType t);

/**
 * @brief Wrapper around dietgpu::FloatCompressSplitContext that exposes a
 * uccl::FloatType-based constructor and getFloatType() accessor, hiding the
 * internal dietgpu::FloatType from external callers.
 */
struct FloatCompressCtx : public dietgpu::FloatCompressSplitContext,
                          public CompressionContext {
  FloatCompressCtx();
  explicit FloatCompressCtx(uccl::FloatType ft);

  // Raw device buffer for the [in_ptr, inSize, out_ptr] tuple in params_dev.
  // Plain cudaMalloc so each ctx frees independently (no LIFO constraint).
  void* raw_params_dev = nullptr;

  FloatCompressCtx(FloatCompressCtx const&) = delete;
  FloatCompressCtx& operator=(FloatCompressCtx const&) = delete;

  ~FloatCompressCtx() override;

  uccl::FloatType getFloatType() const override;
  size_t getMaxSize() const override;
};

#else

/**
 * @brief Dummy device allocation that mimics dietgpu's
 * StackDeviceMemory::Reservation. Provides a no-op release() method for
 * compatibility.
 */
struct DummyDevAlloc {
  void release() {}
  void* data() { return nullptr; }
};

/**
 * @brief Dummy compression context that mirrors the interface of
 * dietgpu::FloatCompressSplitContext. This allows code to compile
 * without #ifdef guards scattered throughout.
 */
struct DummyCompressCtx : public CompressionContext {
  uccl::FloatType float_type = uccl::FloatType::kUndefined;
  size_t maxSize = 0;
  DummyDevAlloc params_dev;
  DummyDevAlloc histogram_dev;
  DummyDevAlloc toComp_dev;

  DummyCompressCtx();
  explicit DummyCompressCtx(uccl::FloatType ft);

  uccl::FloatType getFloatType() const override;
  size_t getMaxSize() const override;
};

#endif

CompressCtx makeCompressCtx(uccl::FloatType ft);

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

  // Queue decompress asynchronously. on_done fires via gpuLaunchHostFunc after
  // the kernel completes; it must not call any GPU APIs.
  virtual void decompressAsync(RemoteMemInfo const& input, RegMemBlock& output,
                               uccl::FloatType float_type, gpuHostFn_t on_done,
                               void* user_data) = 0;

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
  NullCompressorBackend();
  ~NullCompressorBackend() override;

  std::shared_ptr<RegMemBlock> getCompressBuffer() const override;

  std::shared_ptr<RegMemBlock> getDecompressBuffer() const override;

  bool compress(std::shared_ptr<RDMASendRequest> req) override;

  bool prepareDecompress(std::shared_ptr<RDMARecvRequest> req) override;

  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                  uccl::FloatType float_type) override;

  void decompressAsync(RemoteMemInfo const& input, RegMemBlock& output,
                       uccl::FloatType float_type, gpuHostFn_t on_done,
                       void* user_data) override;

  bool shouldCompress(size_t size) override;

  bool shouldCompressAndSplitFirst(size_t size) override;

  void prepareSplitContext(void* addr, size_t size, CompressCtx ctx) override;

  bool compressSplitOneBatch(std::shared_ptr<RDMASendRequest> req) override;

  uint32_t compressEncodeOneBatch(
      std::shared_ptr<RDMASendRequest> req) override;

  CompressStrategy getCompressStrategy() override;

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
  DietGPUCompressorBackend();
  ~DietGPUCompressorBackend() override;

  std::shared_ptr<RegMemBlock> getCompressBuffer() const override;

  std::shared_ptr<RegMemBlock> getDecompressBuffer() const override;

  bool compress(std::shared_ptr<RDMASendRequest> req) override;

  bool prepareDecompress(std::shared_ptr<RDMARecvRequest> req) override;

  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                  uccl::FloatType float_type) override;

  void decompressAsync(RemoteMemInfo const& input, RegMemBlock& output,
                       uccl::FloatType float_type, gpuHostFn_t on_done,
                       void* user_data) override;

  bool shouldCompress(size_t size) override;

  bool shouldCompressAndSplitFirst(size_t size) override;

  void prepareSplitContext(void* addr, size_t size, CompressCtx ctx) override;

  bool compressSplitOneBatch(std::shared_ptr<RDMASendRequest> req) override;

  uint32_t compressEncodeOneBatch(
      std::shared_ptr<RDMASendRequest> req) override;

  CompressStrategy getCompressStrategy() override;

 private:
  uint32_t* devCompressedSize_;
  std::shared_ptr<RegMemBlock> buffer_;            // Compression buffer
  std::shared_ptr<RegMemBlock> decompressBuffer_;  // Decompression buffer
  gpuStream_t stream_;                             // GPU stream
  dietgpu::StackDeviceMemory* res_;  // Device memory for compress/decompress
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
  static Compressor& getInstance();

  // Non-copyable, non-movable
  Compressor(Compressor const&) = delete;
  Compressor& operator=(Compressor const&) = delete;
  Compressor(Compressor&&) = delete;
  Compressor& operator=(Compressor&&) = delete;

  /**
   * @brief Get the compression buffer.
   * @return Shared pointer to the compression buffer RegMemBlock.
   */
  std::shared_ptr<RegMemBlock> getCompressBuffer() const;

  /**
   * @brief Get the decompression buffer.
   * @return Shared pointer to the decompression buffer RegMemBlock.
   */
  std::shared_ptr<RegMemBlock> getDecompressBuffer() const;

  /**
   * @brief Compress a send request's data.
   * @param req The send request to compress.
   * @return true on success, false on failure.
   */
  bool compress(std::shared_ptr<RDMASendRequest> req);

  /**
   * @brief Prepare a receive request for decompression.
   * @param req The receive request to prepare.
   * @return true on success, false on failure.
   */
  bool prepareDecompress(std::shared_ptr<RDMARecvRequest> req);

  /**
   * @brief Decompress data from RemoteMemInfo to RegMemBlock.
   * @param input The RemoteMemInfo containing compressed data.
   * @param output The RegMemBlock to write decompressed data to.
   * @param float_type The float type as uint32_t.
   * @return true on success, false on failure.
   */
  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                  uccl::FloatType float_type);

  void decompressAsync(RemoteMemInfo const& input, RegMemBlock& output,
                       uccl::FloatType float_type, gpuHostFn_t on_done,
                       void* user_data);

  /**
   * @brief Check if a request should be compressed based on size threshold.
   * @param size The size in bytes to check.
   * @return true if the size meets the minimum compression threshold.
   */
  bool shouldCompress(size_t size);

  /**
   * @brief Check if data should be compressed and split first.
   * @param size The size in bytes to check.
   * @return true if should compress and split first.
   */
  bool shouldCompressAndSplitFirst(size_t size);

  /**
   * @brief Prepare a context for split+encode two-phase compression.
   * @param addr Device pointer to the input float data.
   * @param size Size of the input data in bytes.
   * @param ctx Compression context.
   */
  void prepareSplitContext(void* addr, size_t size, CompressCtx ctx);

  /**
   * @brief Phase 1 of two-phase compression: split float data.
   * @param req The send request with compress_ctx already prepared.
   * @return true on success, false on failure.
   */
  bool compressSplitOneBatch(std::shared_ptr<RDMASendRequest> req);

  /**
   * @brief Phase 2 of two-phase compression: ANS encode.
   * @param req The send request with compress_ctx already split.
   * @return compressed size on success, 0 on failure.
   */
  uint32_t compressEncodeOneBatch(std::shared_ptr<RDMASendRequest> req);

  /**
   * @brief Get the compression strategy.
   * @return The current compression strategy.
   */
  CompressStrategy getCompressStrategy();

 private:
  Compressor();

  ~Compressor();

  std::unique_ptr<ICompressorBackend> backend_;
};
