#ifndef COMMON_H
#define COMMON_H

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

typedef uint64_t PeerID;
struct ConnID {
  void* context;
  int sock_fd;
  int dev;
  PeerID peer_id;
};

typedef struct AcceptedMeta {
  std::string ip;
  uint16_t port;
  int gpu_id;
  uint64_t peer_id;
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

enum class FloatType : uint32_t {
  kUndefined = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kFloat32 = 3,
  kFloat8E4M3FN = 4,
  kFloat8E5M2 = 5,
};

struct CompressionContext {
  virtual ~CompressionContext() = default;
  virtual FloatType get_float_type() const = 0;
  virtual size_t get_max_size() const = 0;
};

using CompressCtx = std::shared_ptr<CompressionContext>;

enum class NicMode { EFA, IB };

static constexpr uint8_t kPortNum = 1;
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

inline bool is_broadcom_vendor(uint32_t vendor_id) {
  return vendor_id == 0x14e4 ||  // Broadcom / bnxt_re
         vendor_id == 0x1dd8;    // Existing Broadcom-like provider path
}

inline bool is_intel_vendor(uint32_t vendor_id) { return vendor_id == 0x8086; }

inline bool uses_legacy_verbs_provider(uint32_t vendor_id) {
  return is_broadcom_vendor(vendor_id) || is_intel_vendor(vendor_id);
}

static constexpr uint32_t kBatchPostRecvWr = 32;
static constexpr uint32_t kBatchPollCqe = 32;

static constexpr size_t kTaskRingSize = 1024;
static constexpr size_t kRingCapacity = 16384;  // Must be power of 2

static constexpr size_t kInFlightMaxSizeKB =
    10240000;  // Max in-flight packets per channel

static constexpr uint32_t INVALID_PEER_ID =
    std::numeric_limits<uint32_t>::max();
static constexpr uint32_t INVALID_GPU = std::numeric_limits<uint32_t>::max();

size_t channel_id_to_context_id(uint32_t channel_id);

template <typename T = uint32_t>
struct ContextArrayT {
  T data[kNICContextNumber];

  ContextArrayT() { std::memset(data, 0, sizeof(data)); }

  inline void copy_from(ContextArrayT<T> const& other) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ContextArrayT::copy_from requires trivially copyable T");
    std::memcpy(data, other.data, sizeof(data));
  }

  inline void copy_from(char const* other) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ContextArrayT::copy_from requires trivially copyable T");
    std::memcpy(data, other, sizeof(data));
  }

  inline T get_key_by_channel_id(uint32_t channel_id) const {
    return get_key_by_context_id(channel_id_to_context_id(channel_id));
  }

  inline T get_key_by_context_id(size_t context_id) const {
    return data[context_id];
  }

  inline void set_key_by_context_id(uint32_t context_id, T key) {
    data[context_id] = key;
  }

  inline void set_key_by_channel_id(uint32_t channel_id, T key) {
    set_key_by_context_id(channel_id_to_context_id(channel_id), key);
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
  void set_chunk_count(uint16_t chunk_count);

  // Set index (preserves chunk_count + write_meta flag)
  void set_index(uint16_t index);

  // Distinguishes WriteReqMeta entries on the control channel.
  // kWriteMetaRingCapacity only needs 12 bits, so bit 15 of the low halfword
  // is free.
  static constexpr uint32_t kWriteMetaBit = 1u << 15;
  static constexpr uint32_t kIndexMask = 0x7FFFu;
  constexpr bool is_write_meta() const { return data_ & kWriteMetaBit; }
  void set_write_meta();
  constexpr uint16_t plain_index() const {
    return static_cast<uint16_t>(data_ & kIndexMask);
  }

  // Implicit conversion to uint32_t for compatibility
  constexpr operator uint32_t() const { return data_; }

  // Assignment from uint32_t
  ImmData& operator=(uint32_t raw);

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

  friend std::ostream& operator<<(std::ostream& os, ImmData const& imm);

 private:
  uint32_t data_;
};

struct MessageChunk {
  uint64_t offset;  // Offset from the start of the message
  size_t size;      // Size of this chunk in bytes

  MessageChunk(uint64_t off, size_t sz);

  friend std::ostream& operator<<(std::ostream& os, MessageChunk const& chunk);
};

struct ChunkSplitStrategy {
  static constexpr uint64_t kMessageChunkSizeKB = 512;
  static constexpr uint64_t kMaxSplitNum = 16;

  static size_t get_message_chunk_count(size_t message_size);

  static std::vector<MessageChunk> split_message_to_chunks(size_t message_size);

  // Given message_size and chunk_count, return the uniform chunk size used for
  // all but the (potentially smaller) last chunk.
  static size_t get_regular_chunk_size(size_t message_size, size_t chunk_count);
};

struct ChannelMetaData {
  uint32_t qpn;
  union ibv_gid gid;  // RoCE
  uint16_t lid;       // Infiniband

  friend std::ostream& operator<<(std::ostream& os,
                                  ChannelMetaData const& meta);
};

enum class ChannelType : int16_t { Control, Data };

void copy_rkey_array_from_mr_array(MRArray const& mr_array,
                                   RKeyArray& rkey_array);

void copy_rkeys_from_mr_array_to_bytes(MRArray const& mr_array, char* dst,
                                       size_t dst_size);

struct OOBMetaData {
  std::string server_ip;
  int64_t server_port;
  int gpu_id;
  OOBMetaData();
  OOBMetaData(std::string conn_key_, uint16_t remote_port_);
  friend std::ostream& operator<<(std::ostream& os, OOBMetaData const& meta);
};

struct CQMeta {
  uint64_t wr_id;
  ibv_wc_opcode op_code;
  uint32_t len;
  ImmData imm;
  bool has_imm() const;

  friend std::ostream& operator<<(std::ostream& os, CQMeta const& meta);
};

// Registered memory region info
typedef struct RegMemBlock {
  void* addr;
  size_t size;
  MemoryType type;
  MRArray mr_array;  // Array of MR pointers for multiple contexts

  RegMemBlock();
  RegMemBlock(void* a, size_t s, MemoryType t);

  RegMemBlock(void* a, size_t s, MRArray const& mr_array_in, MemoryType t);

  // Equality operator for hash support (based on size and type only)
  bool operator==(RegMemBlock const& other) const;

  // Set MR by context ID
  void set_mr_by_context_id(uint32_t context_id, struct ibv_mr* mr);

  // Set MR by channel ID
  void set_mr_by_channel_id(uint32_t channel_id, struct ibv_mr* mr);

  // Get MR by channel ID
  struct ibv_mr* get_mr_by_channel_id(uint32_t channel_id) const;

  // Get MR by context ID
  struct ibv_mr* get_mr_by_context_id(uint32_t context_id) const;

  // Get rkey by channel ID (for backward compatibility)
  uint32_t get_key_by_channel_id(uint32_t channel_id) const;

  // Get rkey by context ID (for backward compatibility)
  uint32_t get_key_by_context_id(uint32_t context_id) const;

  friend std::ostream& operator<<(std::ostream& os, RegMemBlock const& block);
} RegMemBlock;

typedef struct RemoteMemInfo {
  uint64_t addr;
  RKeyArray rkey_array;  // For multiple memory regions (e.g., GPU memory)
  size_t length;
  MemoryType type;
  RemoteMemInfo();

  RemoteMemInfo(uint64_t a, size_t len, RKeyArray const& rkey, MemoryType t);
  RemoteMemInfo(uint64_t a, size_t len, MRArray const& mrs, MemoryType t);

  RemoteMemInfo(RegMemBlock const& block);

  RemoteMemInfo(std::shared_ptr<RegMemBlock> const block);

  uint32_t get_key_by_channel_id(uint32_t channel_id) const;

  // Get rkey by context ID (direct index access)
  uint32_t get_key_by_context_id(size_t context_id) const;

  friend std::ostream& operator<<(std::ostream& os, RemoteMemInfo const& info);
} RemoteMemInfo;

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
  uint32_t float_type;         // FloatType
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

enum class SendType { Send, Write, Read };

struct RDMASendRequest {
  std::shared_ptr<RegMemBlock> local_mem;
  std::shared_ptr<RemoteMemInfo> remote_mem;
  // Peer id is local to this endpoint, not necessarily global.
  uint32_t from_peer_id = INVALID_PEER_ID;
  uint32_t to_peer_id = INVALID_PEER_ID;
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
                  ImmData imm = ImmData(), bool signaled = true);

  // Constructor from shared_ptr<RDMASendRequest>
  RDMASendRequest(std::shared_ptr<RDMASendRequest> other,
                  std::shared_ptr<RegMemBlock> local, ImmData imm = ImmData(),
                  bool signaled = true);

  // Constructor from const RDMASendRequest&
  RDMASendRequest(RDMASendRequest const& other,
                  std::shared_ptr<RegMemBlock> local, ImmData imm = ImmData(),
                  bool signaled = true);

  // Getter methods
  uint32_t get_local_key() const;

  uint32_t get_remote_key() const;

  uint64_t get_local_address() const;

  uint64_t get_remote_address() const;

  ImmData get_imm() const;

  uint32_t get_local_len() const;

  friend std::ostream& operator<<(std::ostream& os, RDMASendRequest const& req);
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
  int32_t peer_id;
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
  MetaInfoToExchange();

  // Constructor with required peer_id and channel_id, optional channel_meta and
  // mem_meta
  MetaInfoToExchange(int32_t pid, int32_t cid,
                     std::shared_ptr<ChannelMetaData> ch_meta = nullptr,
                     std::shared_ptr<RemoteMemInfo> mem_meta_ptr = nullptr,
                     ChannelType flag_in = ChannelType::Data, int gid = 0,
                     uint16_t oob_p = 0);

  // Overload << operator for printing
  friend std::ostream& operator<<(std::ostream& os,
                                  MetaInfoToExchange const& meta);
};

// Helper function: make socket non-blocking
int make_socket_non_blocking(int fd);

// Helper function: try send entire buffer, return bytes_sent
ssize_t try_send(int fd, char const* buf, size_t len);

bool is_nic_usable(std::string const& nic_name, NicMode mode);

// Hash support for RegMemBlock (based on size and type only)
namespace std {
template <>
struct hash<RegMemBlock> {
  size_t operator()(RegMemBlock const& block) const;
};

}  // namespace std

#endif
