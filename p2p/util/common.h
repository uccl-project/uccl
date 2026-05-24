#ifndef COMMON_H
#define COMMON_H

#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util/util.h"
#include <arpa/inet.h>
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
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <errno.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

typedef uint64_t FlowID;
struct ConnID {
  void* context;
  int sock_fd;
  int dev;
  FlowID flow_id;
};

typedef struct AcceptedMeta {
  std::string ip;
  uint16_t port;
  int gpu_id;
  uint64_t rank_id;
} AcceptedMeta;

// Notifications
static constexpr uint32_t NOTIFY_MSG_MAGIC = 0xDEADDEAD;
static constexpr size_t NOTIFY_MSG_SIZE = 256;

struct NotifyMsg {
  uint32_t magic;
  uint32_t msg_type;
  char name[NOTIFY_MSG_SIZE];
  char msg[NOTIFY_MSG_SIZE];
};

inline std::vector<NotifyMsg> notify_list;
inline std::mutex notify_mutex;

// FifoItem struct for RDMA operations (64-byte layout)
struct FifoItem {
  uint64_t addr;
  uint32_t size;
  uint32_t rkey;
  uint32_t nmsgs;
  uint32_t rid;
  uint64_t idx;
  char padding[32];
};
static_assert(sizeof(FifoItem) == 64, "FifoItem size must be 64 bytes");

void serialize_fifo_item(FifoItem const& item, char* buf);

void deserialize_fifo_item(char const* buf, FifoItem* item);
enum class MemoryType { HOST, GPU };

namespace uccl {
enum class FloatType : uint32_t {
  kUndefined = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kFloat32 = 3,
  kFloat8E4M3FN = 4,
  kFloat8E5M2 = 5,
};
}

struct CompressionContext {
  virtual ~CompressionContext() = default;
  virtual uccl::FloatType getFloatType() const = 0;
  virtual size_t getMaxSize() const = 0;
};

using CompressCtx = std::shared_ptr<CompressionContext>;

static constexpr uint8_t kPortNum = 1;

enum class NicMode { EFA, IB };

inline bool is_nic_usable(std::string const& nic_name, NicMode mode) {
  if (mode == NicMode::EFA && strncmp(nic_name.c_str(), "rdmap", 5) != 0) {
    return false;
  }

  int dev_count = 0;
  ibv_device** dev_list = ibv_get_device_list(&dev_count);
  if (!dev_list) {
    return false;
  }
  bool usable = false;
  for (int i = 0; i < dev_count; ++i) {
    if (std::strcmp(ibv_get_device_name(dev_list[i]), nic_name.c_str()) != 0) {
      continue;
    }
    ibv_context* ctx = ibv_open_device(dev_list[i]);
    if (!ctx) break;

    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, kPortNum, &port_attr) == 0) {
      if (mode == NicMode::EFA) {
        if (port_attr.state == IBV_PORT_ACTIVE &&
            (port_attr.link_layer == IBV_LINK_LAYER_UNSPECIFIED ||
             port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
             port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND)) {
          usable = true;
        }
      } else {
        if (port_attr.state == IBV_PORT_ACTIVE &&
            (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
             port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) &&
            (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET ||
             port_attr.gid_tbl_len > 0)) {
          usable = true;
        }
      }
    }
    ibv_close_device(ctx);
    break;
  }
  ibv_free_device_list(dev_list);
  return usable;
}

static constexpr int kNumEngines = 4;
static constexpr int kNumGpuRtStreams = 4;

static constexpr int kRankIDPlaceHolder = 9999;

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

static constexpr uint32_t INVALID_RANK_ID =
    std::numeric_limits<uint32_t>::max();
static constexpr uint32_t INVALID_GPU = std::numeric_limits<uint32_t>::max();

size_t channelIdToContextId(uint32_t channel_id);

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

  // Set index (preserves chunk_count + write_meta flag)
  void set_index(uint16_t index) {
    data_ =
        (data_ & 0xFFFF0000) | (index & kIndexMask) | (data_ & kWriteMetaBit);
  }

  // Distinguishes WriteReqMeta entries from SendReqMeta entries on the
  // control channel. kRingCapacity = 16384 only needs 14 bits, so bit 15 of
  // the low halfword is free.
  static constexpr uint32_t kWriteMetaBit = 1u << 15;
  static constexpr uint32_t kIndexMask = 0x7FFFu;
  constexpr bool is_write_meta() const { return data_ & kWriteMetaBit; }
  void set_write_meta() { data_ |= kWriteMetaBit; }
  constexpr uint16_t plain_index() const {
    return static_cast<uint16_t>(data_ & kIndexMask);
  }

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

  static size_t getMessageChunkCount(size_t message_size);

  static std::vector<MessageChunk> splitMessageToChunks(size_t message_size);

  // Given message_size and chunk_count, return the uniform chunk size used for
  // all but the (potentially smaller) last chunk.
  static size_t getRegularChunkSize(size_t message_size, size_t chunk_count);
};

#define LOG_EVERY_N_ENDPOINT(severity, freq)             \
  static std::atomic<int> LOG_OCCURRENCES_##__LINE__(0); \
  if (++LOG_OCCURRENCES_##__LINE__ % (freq) == 0)        \
  UCCL_LOG(severity, UCCL_RDMA)                          \
      << "[count=" << LOG_OCCURRENCES_##__LINE__ << "] "

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

void copyRKeyArrayFromMRArray(MRArray const& mr_array, RKeyArray& rkey_array);

void copyRKeysFromMRArrayToBytes(MRArray const& mr_array, char* dst,
                                 size_t dst_size);

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
  uint32_t from_rank_id = INVALID_RANK_ID;
  uint32_t to_rank_id = INVALID_RANK_ID;
  uint32_t channel_id = 0;
  int64_t wr_id = -1;
  std::shared_ptr<RegMemBlock> local_mem;
  std::shared_ptr<RegMemBlock> local_compression_mem;
  CompressCtx compress_ctx;

  // Constructor
  RDMARecvRequest(std::shared_ptr<RegMemBlock> local) : local_mem(local) {}

  // Getter methods
  inline uint32_t getLocalKey() const {
    return local_mem->getKeyByChannelID(channel_id);
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
    float_type = rev_req->compress_ctx ? rev_req->compress_ctx->getFloatType()
                                       : uccl::FloatType::kUndefined;
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

// ── Compressed write infra ──────────────────────────────────────────────────
// One entry per in-flight compressed write. Pushed by sender to receiver via
// the control QP (RDMA WRITE WITH IMM, IMM carries the ring index + write_meta
// flag bit).
struct alignas(64) WriteReqMeta {
  uint64_t user_remote_addr;   // user-visible WRITE destination (GPU VA)
  uint64_t decompress_offset;  // byte offset within remote decompress_buffer
  uint32_t user_rkey;          // user-provided rkey (for logging/validation)
  uint32_t total_uncomp_size;  // original message size in bytes
  uint32_t compressed_size;    // split + encode output total bytes
  uint32_t float_type;         // uccl::FloatType
  uint32_t ack_slot;  // sender expects ack written here in its ack_ring
  uint32_t _pad;
  int64_t wr_id;
};

// A small on-host ack slot. Receiver writes |wr_id| (or wr_id|kAckErrBit) once
// decompress completes. Sender polls the value to determine completion.
static constexpr uint64_t kAckErrBit = 1ull << 63;
struct alignas(64) AckSlot {
  std::atomic<uint64_t> value;
};

// Ring sizes for the WriteReqMeta channel and the ack ring. Must be ≥ max
// in-flight compressed writes so the monotonic-modulo allocators don't
// wrap onto a still-pending entry.
constexpr size_t kWriteMetaRingCapacity = 4096;
constexpr size_t kAckRingDepth = 4096;
constexpr size_t kWriteMetaRingBytes =
    sizeof(WriteReqMeta) * kWriteMetaRingCapacity;
constexpr size_t kAckRingBytes = sizeof(AckSlot) * kAckRingDepth;

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
  uint32_t from_rank_id = INVALID_RANK_ID;
  uint32_t to_rank_id = INVALID_RANK_ID;
  uint32_t channel_id = 0;
  ImmData imm_data = 0;  // immediate data with chunk_count (high 16 bits) and
                         // index (low 16 bits)
  int64_t wr_id = -1;
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
  // Compressed-write infra; populated only on the Control-channel handshake.
  RemoteMemInfo decompress_buf_meta;
  RemoteMemInfo write_meta_ring_meta;
  RemoteMemInfo ack_ring_meta;

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
int make_socket_non_blocking(int fd);

// Helper function: try send entire buffer, return bytes_sent
ssize_t try_send(int fd, char const* buf, size_t len);

// Hash support for RegMemBlock (based on size and type only)
namespace std {
template <>
struct hash<RegMemBlock> {
  size_t operator()(RegMemBlock const& block) const;
};

}  // namespace std

#endif
