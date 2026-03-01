#pragma once
#include <arpa/inet.h>
#include <glog/logging.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>
#include <errno.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include "common.h"
#include "util/gpu_rt.h"
#include "util/util.h"

namespace uccl {
enum class FloatType : uint32_t {
  kUndefined = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kFloat32 = 3,
};
}

#if defined USE_DIETGPU
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || \
    defined(__HIPCC__)
#include "dietgpu/float/GpuFloatCodec_hip.h"
#include "dietgpu/utils/DeviceUtils_hip.h"
#include "dietgpu/utils/StackDeviceMemory_hip.h"
#include <hip/hip_fp16.h>
#else
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include <cuda_fp16.h>
#endif


inline dietgpu::FloatType to_dietgpu(uccl::FloatType t) {
  switch (t) {
    case uccl::FloatType::kFloat16:
      return dietgpu::FloatType::kFloat16;
    case uccl::FloatType::kBFloat16:
      return dietgpu::FloatType::kBFloat16;
    case uccl::FloatType::kFloat32:
      return dietgpu::FloatType::kFloat32;
    case uccl::FloatType::kUndefined:
    default:
      return dietgpu::FloatType::kUndefined;
  }
}

inline uccl::FloatType from_dietgpu(dietgpu::FloatType t) {
  switch (t) {
    case dietgpu::FloatType::kFloat16:
      return uccl::FloatType::kFloat16;
    case dietgpu::FloatType::kBFloat16:
      return uccl::FloatType::kBFloat16;
    case dietgpu::FloatType::kFloat32:
      return uccl::FloatType::kFloat32;
    default:
      return uccl::FloatType::kUndefined;
  }
}

/**
 * @brief Wrapper around dietgpu::FloatCompressSplitContext that exposes a
 * uccl::FloatType-based constructor and getFloatType() accessor, hiding the
 * internal dietgpu::FloatType from external callers.
 */
struct FloatCompressCtx : public dietgpu::FloatCompressSplitContext {
  FloatCompressCtx() = default;
  explicit FloatCompressCtx(uccl::FloatType ft)
      : dietgpu::FloatCompressSplitContext(to_dietgpu(ft)) {}

  uccl::FloatType getFloatType() const { return from_dietgpu(float_type); }
};

using CompressCtx = std::shared_ptr<FloatCompressCtx>;

inline CompressCtx makeCompressCtx(uccl::FloatType ft) {
  return std::make_shared<FloatCompressCtx>(ft);
}

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
struct DummyCompressCtx {
  uccl::FloatType float_type = uccl::FloatType::kUndefined;
  size_t maxSize = 0;
  DummyDevAlloc params_dev;
  DummyDevAlloc histogram_dev;
  DummyDevAlloc toComp_dev;

  DummyCompressCtx() = default;
  explicit DummyCompressCtx(uccl::FloatType ft) : float_type(ft), maxSize(0) {}

  uccl::FloatType getFloatType() const { return float_type; }
};

using CompressCtx = std::shared_ptr<DummyCompressCtx>;

inline CompressCtx makeCompressCtx(uccl::FloatType ft) {
  return std::make_shared<DummyCompressCtx>(ft);
}

#endif

static constexpr int kNumEngines = 4;
static constexpr int kNumGpuRtStreams = 4;

static constexpr int kRankIDPlaceHolder = 9999;
static constexpr uint8_t kPortNum = 1;

// Use max 4 to accommodate AWS p5 instance
static constexpr int kQpNumPerChannel = 4;
static constexpr int kNICContextNumber = 4;
static constexpr int kControlChannelID = 0;

static constexpr int kMaxSendWr = 1024;
static constexpr int kMaxRecvWr = 1024;
static constexpr int kMaxSendSeg = 2;
static constexpr int kMaxRecvSeg = 2;
static constexpr uint32_t kBatchPostRecvWr = 32;
static constexpr uint32_t kBatchPollCqe = 32;

static constexpr size_t kTaskRingSize = 1024;
static constexpr size_t kRingCapacity = 16384;  // Must be power of 2

static constexpr size_t kInFlightMaxSizeKB =
    10240000;  // Max in-flight packets per channel

constexpr size_t kMinCompressBytes = 2 * 1024 * 1024;      // 1MB
constexpr size_t kCompressBufferSize = 400 * 1024 * 1024;  // 400MB

static constexpr uint32_t INVALID_RANK_ID =
    std::numeric_limits<uint32_t>::max();
static constexpr uint32_t INVALID_GPU = std::numeric_limits<uint32_t>::max();

inline size_t channelIdToContextId(uint32_t channel_id) {
  return (channel_id == 0) ? 0 : (channel_id - 1) % kNICContextNumber;
}

template <typename T = uint32_t>
struct ContextArrayT {
  T data[kNICContextNumber];

  ContextArrayT() { std::memset(data, 0, sizeof(data)); }

  inline void copyFrom(ContextArrayT<T> const& other) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ContextArrayT::copyFrom requires trivially copyable T");
    std::memcpy(data, other.data, sizeof(data));
  }

  inline void copyFrom(char const* other) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ContextArrayT::copyFrom requires trivially copyable T");
    std::memcpy(data, other, sizeof(data));
  }

  inline T getKeyByChannelID(uint32_t channel_id) const {
    return getKeyByContextID(channelIdToContextId(channel_id));
  }

  inline T getKeyByContextID(size_t context_id) const {
    return data[context_id];
  }

  inline void setKeyByContextID(uint32_t context_id, T key) {
    data[context_id] = key;
  }

  inline void setKeyByChannelID(uint32_t channel_id, T key) {
    setKeyByContextID(channelIdToContextId(channel_id), key);
  }

  T& operator[](int index) { return data[index]; }
  T const& operator[](int index) const { return data[index]; }

  friend std::ostream& operator<<(std::ostream& os,
                                  ContextArrayT<T> const& arr) {
    os << "ContextArrayT{";
    for (int i = 0; i < kNICContextNumber; ++i) {
      if (i > 0) os << ", ";
      if constexpr (std::is_integral_v<T>) {
        os << "0x" << std::hex << arr.data[i] << std::dec;
      } else {
        os << arr.data[i];
      }
    }
    os << "}";
    return os;
  }
};

using RKeyArray = ContextArrayT<uint32_t>;
using MRArray = ContextArrayT<struct ibv_mr*>;

// ImmData class: encapsulates immediate data with chunk_count (high 16 bits)
// and index (low 16 bits)
class ImmData {
 public:
  // Default constructor
  constexpr ImmData() : data_(0) {}

  // Constructor from raw uint32_t value
  constexpr ImmData(uint32_t raw) : data_(raw) {}
  constexpr ImmData(int raw) : data_(raw) {}
  // Constructor from chunk_count and index
  constexpr ImmData(uint16_t chunk_count, uint16_t index)
      : data_((static_cast<uint32_t>(chunk_count) << 16) | index) {}

  // Get chunk_count (high 16 bits)
  constexpr uint16_t chunk_count() const {
    return static_cast<uint16_t>(data_ >> 16);
  }

  // Get index (low 16 bits)
  constexpr uint16_t index() const {
    return static_cast<uint16_t>(data_ & 0xFFFF);
  }

  // Set chunk_count (preserves index)
  void set_chunk_count(uint16_t chunk_count) {
    data_ = (static_cast<uint32_t>(chunk_count) << 16) | (data_ & 0xFFFF);
  }

  // Set index (preserves chunk_count)
  void set_index(uint16_t index) { data_ = (data_ & 0xFFFF0000) | index; }

  // Implicit conversion to uint32_t for compatibility
  constexpr operator uint32_t() const { return data_; }

  // Assignment from uint32_t
  ImmData& operator=(uint32_t raw) {
    data_ = raw;
    return *this;
  }

  // Get raw value
  constexpr uint32_t raw() const { return data_; }

  // Comparison operators
  constexpr bool operator==(ImmData const& other) const {
    return data_ == other.data_;
  }
  constexpr bool operator!=(ImmData const& other) const {
    return data_ != other.data_;
  }
  constexpr bool operator==(uint32_t other) const { return data_ == other; }
  constexpr bool operator!=(uint32_t other) const { return data_ != other; }

  friend std::ostream& operator<<(std::ostream& os, ImmData const& imm) {
    os << "ImmData{raw: " << imm.data_ << ", chunk_count: " << imm.chunk_count()
       << ", index: " << imm.index() << "}";
    return os;
  }

 private:
  uint32_t data_;
};

struct MessageChunk {
  uint64_t offset;  // Offset from the start of the message
  size_t size;      // Size of this chunk in bytes

  MessageChunk(uint64_t off, size_t sz) : offset(off), size(sz) {}

  friend std::ostream& operator<<(std::ostream& os, MessageChunk const& chunk) {
    os << "MessageChunk{offset: " << chunk.offset << ", size: " << chunk.size
       << "}";
    return os;
  }
};

struct ChunkSplitStrategy {
  static constexpr uint64_t kMessageChunkSizeKB = 512;
  static constexpr uint64_t kMaxSplitNum = 16;

  static size_t getMessageChunkCount(size_t message_size) {
    constexpr size_t chunk_size_bytes = kMessageChunkSizeKB * 1024;
    if (message_size == 0) return 0;
    size_t chunk_count =
        (message_size + chunk_size_bytes - 1) / chunk_size_bytes;
    return std::min(chunk_count, static_cast<size_t>(kMaxSplitNum));
  }

  static std::vector<MessageChunk> splitMessageToChunks(size_t message_size) {
    size_t chunk_count = getMessageChunkCount(message_size);
    std::vector<MessageChunk> chunks;
    chunks.reserve(chunk_count);
    size_t actual_chunk_size = getRegularChunkSize(message_size, chunk_count);
    for (size_t i = 0; i < chunk_count; ++i) {
      uint64_t offset = i * actual_chunk_size;
      size_t size = std::min(actual_chunk_size, message_size - offset);
      chunks.emplace_back(offset, size);
    }
    return chunks;
  }

  // Given message_size and chunk_count, return the uniform chunk size used for
  // all but the (potentially smaller) last chunk.
  static size_t getRegularChunkSize(size_t message_size, size_t chunk_count) {
    if (chunk_count == 0 || message_size == 0) return 0;
    return (message_size + chunk_count - 1) / chunk_count;
  }
};

static_assert(
    kMinCompressBytes > 4 * ChunkSplitStrategy::kMessageChunkSizeKB,
    "kMinCompressBytes must be > 4 * ChunkSplitStrategy::kMessageChunkSizeKB");

#define LOG_EVERY_N_ENDPOINT(severity, freq)             \
  static std::atomic<int> LOG_OCCURRENCES_##__LINE__(0); \
  if (++LOG_OCCURRENCES_##__LINE__ % (freq) == 0)        \
  LOG(severity) << "[count=" << LOG_OCCURRENCES_##__LINE__ << "] "

struct ChannelMetaData {
  uint32_t qpn;
  union ibv_gid gid;  // RoCE
  uint16_t lid;       // Infiniband

  friend std::ostream& operator<<(std::ostream& os,
                                  ChannelMetaData const& meta) {
    os << "ChannelMetaData{qpn: " << meta.qpn << ", gid: ";
    std::ios_base::fmtflags flags = os.flags();
    for (int i = 0; i < 16; ++i) {
      os << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<int>(meta.gid.raw[i]);
      if (i == 7) os << ":";
    }
    os.flags(flags);
    os << "}";
    return os;
  }
};

enum class ChannelType : int16_t { Control, Normal };

inline void copyRKeyArrayFromMRArray(MRArray const& mr_array,
                                     RKeyArray& rkey_array) {
  for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
    ibv_mr* mr = mr_array.getKeyByContextID(ctx);
    uint32_t rkey = mr ? mr->rkey : 0;
    rkey_array.setKeyByContextID(ctx, rkey);
  }
}

inline void copyRKeysFromMRArrayToBytes(MRArray const& mr_array, char* dst,
                                        size_t dst_size) {
  constexpr size_t needed = sizeof(uint32_t) * kNICContextNumber;
  assert(dst_size >= needed);

  uint32_t* out = reinterpret_cast<uint32_t*>(dst);

  for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
    ibv_mr* mr = mr_array.getKeyByContextID(ctx);
    out[ctx] = mr ? mr->rkey : 0;
  }
}

struct OOBMetaData {
  std::string server_ip;
  int64_t server_port;
  int gpu_id;
  OOBMetaData() : server_ip(""), server_port(0), gpu_id(0) {}
  OOBMetaData(std::string conn_key_, uint16_t remote_port_)
      : server_ip(std::move(conn_key_)), server_port(remote_port_), gpu_id(0) {}
  friend std::ostream& operator<<(std::ostream& os, OOBMetaData const& meta) {
    os << "OOBMetaData{server_ip: " << meta.server_ip
       << ", server_port: " << meta.server_port << ", gpu_id: " << meta.gpu_id
       << "}";
    return os;
  }
};

struct CQMeta {
  uint64_t wr_id;
  ibv_wc_opcode op_code;
  uint32_t len;
  ImmData imm;
  inline bool hasIMM() const { return op_code == IBV_WC_RECV_RDMA_WITH_IMM; };

  friend std::ostream& operator<<(std::ostream& os, CQMeta const& meta) {
    os << "CQMeta{wr_id: " << meta.wr_id << ", op_code: " << meta.op_code
       << ", len: " << meta.len << ", imm: " << meta.imm
       << ", hasIMM: " << (meta.op_code == IBV_WC_RECV_RDMA_WITH_IMM) << "}";
    return os;
  }
};

// Registered memory region info
typedef struct RegMemBlock {
  void* addr;
  size_t size;
  MemoryType type;
  MRArray mr_array;  // Array of MR pointers for multiple contexts

  RegMemBlock() : addr(nullptr), size(0), type(MemoryType::HOST) {}
  RegMemBlock(void* a, size_t s, MemoryType t) : addr(a), size(s), type(t) {}

  RegMemBlock(void* a, size_t s, MRArray const& mr_array_in, MemoryType t)
      : addr(a), size(s), type(t) {
    mr_array.copyFrom(mr_array_in);
  }

  // Equality operator for hash support (based on size and type only)
  bool operator==(RegMemBlock const& other) const {
    return addr == other.addr && size == other.size && type == other.type;
  }

  // Set MR by context ID
  inline void setMRByContextID(uint32_t context_id, struct ibv_mr* mr) {
    mr_array.setKeyByContextID(context_id, mr);
  }

  // Set MR by channel ID
  inline void setMRByChannelID(uint32_t channel_id, struct ibv_mr* mr) {
    mr_array.setKeyByChannelID(channel_id, mr);
  }

  // Get MR by channel ID
  inline struct ibv_mr* getMRByChannelID(uint32_t channel_id) const {
    return mr_array.getKeyByChannelID(channel_id);
  }

  // Get MR by context ID
  inline struct ibv_mr* getMRByContextID(uint32_t context_id) const {
    return mr_array.getKeyByContextID(context_id);
  }

  // Get rkey by channel ID (for backward compatibility)
  inline uint32_t getKeyByChannelID(uint32_t channel_id) const {
    struct ibv_mr* mr = getMRByChannelID(channel_id);
    return mr ? mr->rkey : 0;
  }

  // Get rkey by context ID (for backward compatibility)
  inline uint32_t getKeyByContextID(uint32_t context_id) const {
    struct ibv_mr* mr = getMRByContextID(context_id);
    return mr ? mr->rkey : 0;
  }

  friend std::ostream& operator<<(std::ostream& os, RegMemBlock const& block) {
    os << "RegMemBlock{addr: " << block.addr << ", size: " << block.size
       << ", type: " << (block.type == MemoryType::HOST ? "HOST" : "GPU")
       << block.mr_array << std::endl;

    os << "}";
    return os;
  }
} RegMemBlock;

typedef struct RemoteMemInfo {
  uint64_t addr;
  RKeyArray rkey_array;  // For multiple memory regions (e.g., GPU memory)
  size_t length;
  MemoryType type;
  RemoteMemInfo() : addr(0), length(0), type(MemoryType::HOST) {}

  RemoteMemInfo(uint64_t a, size_t len, RKeyArray const& rkey, MemoryType t)
      : addr(a), length(len), type(t) {
    rkey_array.copyFrom(rkey);
  }
  RemoteMemInfo(uint64_t a, size_t len, MRArray const& mrs, MemoryType t)
      : addr(a), length(len), type(t) {
    copyRKeyArrayFromMRArray(mrs, rkey_array);
  }

  RemoteMemInfo(RegMemBlock const& block)
      : addr(reinterpret_cast<uint64_t>(block.addr)),
        length(block.size),
        type(block.type) {
    copyRKeyArrayFromMRArray(block.mr_array, rkey_array);
  }

  RemoteMemInfo(std::shared_ptr<RegMemBlock> const block)
      : addr(reinterpret_cast<uint64_t>(block->addr)),
        length(block->size),
        type(block->type) {
    copyRKeyArrayFromMRArray(block->mr_array, rkey_array);
  }

  inline uint32_t getKeyByChannelID(uint32_t channel_id) const {
    return rkey_array.getKeyByChannelID(channel_id);
  }

  // Get rkey by context ID (direct index access)
  inline uint32_t getKeyByContextID(size_t context_id) const {
    return rkey_array.getKeyByContextID(context_id);
  }

  friend std::ostream& operator<<(std::ostream& os, RemoteMemInfo const& info) {
    os << "RemoteMemInfo{addr: 0x" << std::hex << info.addr << std::dec
       << ", length: " << info.length
       << ", type: " << (info.type == MemoryType::HOST ? "HOST" : "GPU")
       << "rkey_array " << info.rkey_array << "}" << std::endl;
    return os;
  }
} RemoteMemInfo;

typedef struct RDMARecvRequest {
  uint32_t from_rank_id;
  uint32_t to_rank_id;
  uint32_t channel_id;
  int64_t wr_id;
  std::shared_ptr<RegMemBlock> local_mem;
  std::shared_ptr<RegMemBlock> local_compression_mem;
  CompressCtx compress_ctx;

  // Constructor
  RDMARecvRequest(std::shared_ptr<RegMemBlock> local) : local_mem(local) {}

  // Getter methods
  inline uint32_t getLocalKey() const {
    return local_mem->getKeyByContextID(channel_id);
  }

  inline uint64_t getLocalAddress() const {
    return reinterpret_cast<uint64_t>(local_mem->addr);
  }

  inline uint32_t getLocalLen() const { return local_mem->size; }

  friend std::ostream& operator<<(std::ostream& os,
                                  RDMARecvRequest const& req) {
    os << "RDMARecvRequest{";
    os << "from_rank_id: " << req.from_rank_id
       << ", to_rank_id: " << req.to_rank_id
       << ", channel_id: " << req.channel_id;
    if (req.local_mem) {
      os << ", local_mem: " << *req.local_mem;
    } else {
      os << ", local_mem: nullptr";
    }
    os << "}";
    return os;
  }
} RDMARecvRequest;

enum class ReqFlag : int16_t { PENDING = 2, IN_PROGRESS = 3, IS_DONE = 4 };

struct alignas(64) SendReqMeta {
  uint32_t rank_id;
  uint32_t channel_id;
  RemoteMemInfo remote_mem;
  RegMemBlock local_mem;
  uccl::FloatType float_type = uccl::FloatType::kUndefined;
  uint32_t expected_chunk_count;  // Expected number of chunks to receive
  uint32_t received_chunk_count;  // Number of chunks already received

  SendReqMeta()
      : rank_id(0),
        channel_id(0),
        remote_mem{},
        expected_chunk_count(0),
        received_chunk_count(0) {}

  SendReqMeta(uint32_t rid, uint32_t cid, RemoteMemInfo const& rmem,
              uint32_t expected = 0, uint32_t received = 0)
      : rank_id(rid),
        channel_id(cid),
        remote_mem(rmem),
        expected_chunk_count(expected),
        received_chunk_count(received) {}

  SendReqMeta(std::shared_ptr<RDMARecvRequest> rev_req) {
    rank_id = rev_req->from_rank_id;
    channel_id = rev_req->channel_id;
    remote_mem = rev_req->local_mem;
    local_mem =
        *(rev_req->local_compression_mem ? rev_req->local_compression_mem
                                         : rev_req->local_mem);
    float_type = rev_req->compress_ctx->getFloatType();
    expected_chunk_count =
        ChunkSplitStrategy::getMessageChunkCount(local_mem.size);
    received_chunk_count = 0;
  }

  friend std::ostream& operator<<(std::ostream& os, SendReqMeta const& meta) {
    os << "SendReqMeta{rank_id: " << meta.rank_id
       << ", channel_id: " << meta.channel_id
       << ", remote_mem: " << meta.remote_mem
       << ", expected_chunk_count: " << meta.expected_chunk_count
       << ", received_chunk_count: " << meta.received_chunk_count << "}";
    return os;
  }
};

struct alignas(64) SendReqMetaOnRing {
  SendReqMeta meta;
  std::atomic<ReqFlag> flag;

  SendReqMetaOnRing() : meta{}, flag{ReqFlag::PENDING} {}

  SendReqMetaOnRing(SendReqMetaOnRing const& other)
      : meta(other.meta), flag{other.flag.load(std::memory_order_relaxed)} {}

  SendReqMetaOnRing& operator=(SendReqMetaOnRing const& other) {
    meta = other.meta;
    flag.store(other.flag.load(std::memory_order_relaxed),
               std::memory_order_relaxed);
    return *this;
  }

  // Get SendReqMeta from SendReqMetaOnRing
  SendReqMeta getSendReqMeta() const { return meta; }

  // Set SendReqMeta part
  void setSendReqMeta(SendReqMeta const& m) { meta = m; }

  friend std::ostream& operator<<(std::ostream& os,
                                  SendReqMetaOnRing const& ring) {
    auto flag_val = ring.flag.load(std::memory_order_relaxed);
    os << "SendReqMetaOnRing{meta: " << ring.meta << ", flag: "
       << (flag_val == ReqFlag::PENDING       ? "PENDING"
           : flag_val == ReqFlag::IN_PROGRESS ? "IN_PROGRESS"
                                              : "IS_DONE")
       << "}";
    return os;
  }
};

// Ring buffer size for control channel
static constexpr size_t kRingBufferSize =
    sizeof(SendReqMetaOnRing) * kRingCapacity;

// Helper functions for SendReqMetaOnRing to be used with
// modify_and_advance_write
inline auto check_in_progress = [](SendReqMetaOnRing const& item) {
  return item.flag.load(std::memory_order_acquire) == ReqFlag::IN_PROGRESS;
};

inline auto check_is_done = [](SendReqMetaOnRing const& item) {
  return item.flag.load(std::memory_order_acquire) == ReqFlag::IS_DONE;
};

inline auto set_in_progress = [](SendReqMetaOnRing& item) {
  item.flag.store(ReqFlag::IN_PROGRESS, std::memory_order_release);
};

inline auto set_is_done = [](SendReqMetaOnRing& item) {
  item.flag.store(ReqFlag::IS_DONE, std::memory_order_release);
};

// Check if all chunks have been received
inline auto check_all_chunks_received = [](SendReqMetaOnRing const& item) {
  return item.meta.received_chunk_count == item.meta.expected_chunk_count;
};

// Increment the received chunk count
inline auto increment_received_chunk = [](SendReqMetaOnRing& item) {
  item.meta.received_chunk_count++;
};

// Conversion functions between SendReqMeta and SendReqMetaOnRing
inline auto to_ring_meta = [](SendReqMeta const& src, SendReqMetaOnRing& dst) {
  dst.meta = src;
  dst.flag.store(ReqFlag::PENDING, std::memory_order_relaxed);
};

inline auto from_ring_meta = [](SendReqMetaOnRing const& src,
                                SendReqMeta& dst) { dst = src.meta; };

enum class SendType { Send, Write, Read };
struct RDMASendRequest {
  std::shared_ptr<RegMemBlock> local_mem;
  std::shared_ptr<RemoteMemInfo> remote_mem;
  uint32_t from_rank_id;
  uint32_t to_rank_id;
  uint32_t channel_id;
  ImmData imm_data = 0;  // immediate data with chunk_count (high 16 bits) and
                         // index (low 16 bits)
  int64_t wr_id;
  bool need_signaled;  // Whether to use IBV_SEND_SIGNALED flag
  SendType send_type = SendType::Send;
  CompressCtx compress_ctx;
  // Constructor
  RDMASendRequest(std::shared_ptr<RegMemBlock> local,
                  std::shared_ptr<RemoteMemInfo> remote,
                  ImmData imm = ImmData(), bool signaled = true)
      : local_mem(local),
        remote_mem(remote),
        imm_data(imm),
        need_signaled(signaled) {}

  // Constructor from shared_ptr<RDMASendRequest>
  RDMASendRequest(std::shared_ptr<RDMASendRequest> other,
                  std::shared_ptr<RegMemBlock> local, ImmData imm = ImmData(),
                  bool signaled = true)
      : local_mem(local),
        remote_mem(other->remote_mem),
        imm_data(imm),
        need_signaled(signaled) {}

  // Constructor from const RDMASendRequest&
  RDMASendRequest(RDMASendRequest const& other,
                  std::shared_ptr<RegMemBlock> local, ImmData imm = ImmData(),
                  bool signaled = true)
      : local_mem(local),
        remote_mem(other.remote_mem),
        imm_data(imm),
        need_signaled(signaled) {}

  // Getter methods
  inline uint32_t getLocalKey() const {
    return local_mem->getKeyByChannelID(channel_id);
  }

  inline uint32_t getRemoteKey() const {
    return remote_mem->getKeyByChannelID(channel_id);
  }

  inline uint64_t getLocalAddress() const {
    return reinterpret_cast<uint64_t>(local_mem->addr);
  }

  inline uint64_t getRemoteAddress() const { return remote_mem->addr; }

  inline ImmData getImm() const { return imm_data; }

  inline uint32_t getLocalLen() const { return local_mem->size; }

  friend std::ostream& operator<<(std::ostream& os,
                                  RDMASendRequest const& req) {
    os << "RDMASendRequest{";
    os << "from_rank_id: " << req.from_rank_id
       << ", to_rank_id: " << req.to_rank_id
       << ", channel_id: " << req.channel_id << ", imm_data: " << req.imm_data
       << ", need_signaled: " << req.need_signaled;
    if (req.local_mem) {
      os << ", local_mem: " << *req.local_mem;
    } else {
      os << ", local_mem: nullptr";
    }
    if (req.remote_mem) {
      os << ", remote_mem: " << *req.remote_mem;
    } else {
      os << ", remote_mem: nullptr";
    }
    os << "}";
    return os;
  }
};

template <typename T>
std::string serialize(T const& obj) {
  std::string s(sizeof(T), '\0');
  std::memcpy(&s[0], reinterpret_cast<char const*>(&obj), sizeof(T));
  return s;
}

template <typename T>
T deserialize(std::string const& s) {
  T obj{};
  size_t copy_len = std::min(s.size(), sizeof(T));
  std::memcpy(reinterpret_cast<char*>(&obj), s.data(), copy_len);
  return obj;
}

// Example MetaInfo struct for metadata exchange
struct MetaInfoToExchange {
  int32_t rank_id;
  int32_t channel_id;
  ChannelType flag;
  ChannelMetaData channel_meta;
  RemoteMemInfo mem_meta;
  int gpu_id;
  uint16_t oob_port;  // OOB server port for back-connection (notifications)

  // Default constructor
  MetaInfoToExchange()
      : rank_id(0),
        channel_id(0),
        flag(ChannelType::Normal),
        channel_meta{},
        mem_meta{},
        gpu_id(0),
        oob_port(0) {}

  // Constructor with required rank_id and channel_id, optional channel_meta and
  // mem_meta
  MetaInfoToExchange(int32_t rid, int32_t cid,
                     std::shared_ptr<ChannelMetaData> ch_meta = nullptr,
                     std::shared_ptr<RemoteMemInfo> mem_meta_ptr = nullptr,
                     ChannelType flag_in = ChannelType::Normal, int gid = 0,
                     uint16_t oob_p = 0)
      : rank_id(rid),
        channel_id(cid),
        channel_meta{},
        mem_meta{},
        flag(flag_in),
        gpu_id(gid),
        oob_port(oob_p) {
    if (ch_meta) {
      channel_meta = *ch_meta;
    }
    if (mem_meta_ptr) {
      mem_meta = *mem_meta_ptr;
    }
  }

  // Overload << operator for printing
  friend std::ostream& operator<<(std::ostream& os,
                                  MetaInfoToExchange const& meta) {
    os << "=== MetaInfoToExchange ===" << std::endl;
    os << "  rank_id: " << meta.rank_id << std::endl;
    os << "  channel_id: " << meta.channel_id << std::endl;

    os << "  channel_meta:" << std::endl;
    os << "    qpn: " << meta.channel_meta.qpn << std::endl;
    os << "    gid: ";

    // Save stream flags
    std::ios_base::fmtflags flags = os.flags();

    for (int i = 0; i < 16; ++i) {
      os << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<int>(meta.channel_meta.gid.raw[i]);
      if (i == 7) os << ":";
    }

    // Restore stream flags
    os.flags(flags);
    os << std::endl;

    os << "  mem_meta:" << meta.mem_meta << std::endl;
    os << "  gpu_id: " << meta.gpu_id << std::endl;
    os << "=========================" << std::endl;

    return os;
  }
};

// Helper function: make socket non-blocking
static inline int make_socket_non_blocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) return -1;
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) return -1;
  return 0;
}

// Helper function: try send entire buffer, return bytes_sent
static inline ssize_t try_send(int fd, char const* buf, size_t len) {
  ssize_t n = ::send(fd, buf, len, MSG_NOSIGNAL);
  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
    return -1;
  }
  return n;
}

// Hash support for RegMemBlock (based on size and type only)
namespace std {
template <>
struct hash<RegMemBlock> {
  size_t operator()(RegMemBlock const& block) const {
    // Combine hash of size and type
    size_t h1 = hash<size_t>{}(block.size);
    size_t h2 = hash<int>{}(static_cast<int>(block.type));

    // Use a simple hash combining algorithm
    return h1 ^ (h2 << 1);
  }
};

}  // namespace std

enum class CompressStrategy {
  kNone,         // no compression
  kSplitOnly,    // only split, no encode
  kSplitEncode,  // split + encode (default)
};

inline CompressStrategy getCompressStrategyFromEnv() {
  char const* env = std::getenv("P2P_COMPRESS_STRATEGY");

  // default strategy
  if (!env || env[0] == '\0') {
    return CompressStrategy::kNone;
  }

  std::string s(env);
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);

  // ---- accepted values ----
  if (s == "none" || s == "off" || s == "0") {
    return CompressStrategy::kNone;
  }

  if (s == "split" || s == "split_only") {
    return CompressStrategy::kSplitOnly;
  }

  if (s == "encode" || s == "split_encode" || s == "full" || s == "1") {
    return CompressStrategy::kSplitEncode;
  }

  // ---- fallback ----
  // unknown value -> default
  return CompressStrategy::kSplitEncode;
}