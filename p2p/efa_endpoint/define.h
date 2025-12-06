#pragma once
#include <arpa/inet.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sys/mman.h>

// Socket programming headers
#include <netinet/in.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

// Standard library headers
#include "util/util.h"
#include <glog/logging.h>
#include <algorithm>  // for std::find
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>  // for std::cout, std::cerr
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
#include <cuda_runtime.h>

#include <transport.h>

static constexpr int kGidIndex = 0;
static constexpr int kRankIDPlaceHolder = 9999;
static constexpr int kEfaQpLowLatencyServiceLevel = 8;
static constexpr uint32_t kQKey = 0x15695;
static constexpr uint8_t kPortNum = 1;
static constexpr uint8_t kEfaRdmDefaultRnrRetry = 3;

static constexpr int kQpNumPerChannel = 4;

static constexpr int kMaxSendWr = 1024;
static constexpr int kMaxRecvWr = 1024;
static constexpr int kMaxSendSeg = 2;
static constexpr int kMaxRecvSeg = 2;
static constexpr uint64_t kMessageChunkSizeKB = 256;  // 1 MB
static constexpr uint64_t kMaxSplitNum = 16;

static constexpr size_t kTaskRingSize = 1024;


static constexpr size_t kInFlightMaxSizeKB =
    10240000;  // Max in-flight packets per channel

// Message chunk information for splitting large messages
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

// Calculate the number of chunks needed for a message
// If the message would be split into more than kMaxSplitNum chunks,
// returns kMaxSplitNum instead.
inline size_t getMessageChunkCount(size_t message_size) {
  constexpr size_t chunk_size_bytes = kMessageChunkSizeKB * 1024;
  if (message_size == 0) return 0;
  size_t chunk_count = (message_size + chunk_size_bytes - 1) / chunk_size_bytes;
  return std::min(chunk_count, static_cast<size_t>(kMaxSplitNum));
}

// Split a message into chunks based on kMessageChunkSizeKB
// Returns a vector of MessageChunk that can be used with range-based for loop
// without extra copies (use: for (const auto& chunk : chunks))
// Note: If the message would be split into more than kMaxSplitNum chunks,
// it will be split into exactly kMaxSplitNum chunks with larger chunk sizes.
inline std::vector<MessageChunk> splitMessageToChunks(size_t message_size) {
  constexpr size_t chunk_size_bytes = kMessageChunkSizeKB * 1024;
  size_t chunk_count = getMessageChunkCount(message_size);

  std::vector<MessageChunk> chunks;
  chunks.reserve(chunk_count);

  // Calculate actual chunk size based on the limited chunk count
  size_t actual_chunk_size = (message_size + chunk_count - 1) / chunk_count;

  for (size_t i = 0; i < chunk_count; ++i) {
    uint64_t offset = i * actual_chunk_size;
    size_t size = std::min(actual_chunk_size, message_size - offset);
    chunks.emplace_back(offset, size);
  }

  return chunks;
}
// Branch prediction hints for optimization

// #ifdef __GNUC__
// #define likely(x)   __builtin_expect(!!(x), 1)
// #define unlikely(x) __builtin_expect(!!(x), 0)
// #else
// #define likely(x)   (x)
// #define unlikely(x) (x)
// #endif

static constexpr size_t kRingCapacity = 16384;  // Must be power of 2

#define LOG_EVERY_N_ENDPOINT(severity, freq)             \
  static std::atomic<int> LOG_OCCURRENCES_##__LINE__(0); \
  if (++LOG_OCCURRENCES_##__LINE__ % (freq) == 0)        \
  LOG(severity) << "[count=" << LOG_OCCURRENCES_##__LINE__ << "] "

struct ChannelMetaData {
  uint32_t qpn;
  union ibv_gid gid;

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

enum class MemoryType { HOST, GPU };

enum class ChannelType : int16_t { Control, Normal };

struct OOBMetaData {
  std::string server_ip;
  int64_t server_port;
  int gpu_id;
  OOBMetaData() : server_ip(""), server_port(0), gpu_id(0) {}
  OOBMetaData(std::string conn_key_, uint16_t remote_port_)
      : server_ip(std::move(conn_key_)), server_port(remote_port_), gpu_id(0) {}
  friend std::ostream& operator<<(std::ostream& os, OOBMetaData const& meta) {
    os << "OOBMetaData{server_ip: " << meta.server_ip
       << ", server_port: " << meta.server_port
       << ", gpu_id: " << meta.gpu_id << "}";
    return os;
  }
};

struct CQMeta {
  uint64_t wr_id;
  ibv_wc_opcode op_code;
  uint32_t len;
  int imm;
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
  struct ibv_mr* mr;
  std::unordered_map<int64_t, struct ibv_mr*> mr_map;
  bool pool_allocated = false;
  uint32_t rkeys[kQpNumPerChannel +
                 1];  // For multiple memory regions (e.g., GPU memory)

  RegMemBlock(void* a, size_t s, MemoryType t, struct ibv_mr* m = nullptr, bool p = false)
      : addr(a), size(s), type(t), mr(m), pool_allocated(p){};

  // Equality operator for hash support (based on size and type only)
  bool operator==(RegMemBlock const& other) const {
    return addr == other.addr && size == other.size && type == other.type;
  }

  friend std::ostream& operator<<(std::ostream& os, RegMemBlock const& block) {
    os << "RegMemBlock{addr: " << block.addr << ", size: " << block.size
       << ", type: " << (block.type == MemoryType::HOST ? "HOST" : "GPU")
       << ", mr: " << block.mr << ", lkey: " << (block.mr ? block.mr->lkey : 0)
       << ", rkey: " << (block.mr ? block.mr->rkey : 0)
       << ", pool_allocated: " << block.pool_allocated << "}";
    return os;
  }
} RegMemBlock;

typedef struct RemoteMemInfo {
  uint64_t addr;
  uint32_t rkey;
  uint32_t rkeys[kQpNumPerChannel +
                 1];  // For multiple memory regions (e.g., GPU memory)
  size_t length;
  MemoryType type;

  // Default constructor
  RemoteMemInfo() : addr(0), rkey(0), length(0), type(MemoryType::HOST) {}

  // Constructor with all parameters
  RemoteMemInfo(uint64_t a, uint32_t k, size_t len, MemoryType t)
      : addr(a), rkey(k), length(len), type(t) {}

  // Constructor from RegMemBlock (copy constructor)
  RemoteMemInfo(RegMemBlock const& block)
      : addr(reinterpret_cast<uint64_t>(block.addr)),
        rkey(block.mr ? block.mr->rkey : 0),
        length(block.size),
        type(block.type) {
    for (int i = 0; i < kQpNumPerChannel + 1; ++i) {
      rkeys[i] = block.rkeys[i];
    }
  }

  // Constructor from shared_ptr<RegMemBlock> (copy constructor)
  RemoteMemInfo(const std::shared_ptr<RegMemBlock> block)
      : addr(reinterpret_cast<uint64_t>(block->addr)),
        rkey(block->mr ? block->mr->rkey : 0),
        length(block->size),
        type(block->type) {
    for (int i = 0; i < kQpNumPerChannel + 1; ++i) {
      rkeys[i] = block->rkeys[i];
    }
  }

  friend std::ostream& operator<<(std::ostream& os, RemoteMemInfo const& info) {
    os << "RemoteMemInfo{addr: 0x" << std::hex << info.addr << std::dec
       << ", rkey: 0x" << std::hex << info.rkey << std::dec
       << ", length: " << info.length
       << ", type: " << (info.type == MemoryType::HOST ? "HOST" : "GPU") << "}";
    return os;
  }
} RemoteMemInfo;

enum class ReqFlag : int16_t { PENDING = 2, IN_PROGRESS = 3, IS_DONE = 4 };
struct alignas(64) SendReqMeta {
  uint32_t rank_id;
  uint32_t channel_id;
  RemoteMemInfo remote_mem;
  uint32_t expected_chunk_count;  // Expected number of chunks to receive
  uint32_t received_chunk_count;  // Number of chunks already received

  // Default constructor
  SendReqMeta()
      : rank_id(0),
        channel_id(0),
        remote_mem{},
        expected_chunk_count(0),
        received_chunk_count(0) {}

  // Constructor with parameters
  SendReqMeta(uint32_t rid, uint32_t cid, RemoteMemInfo const& rmem,
              uint32_t expected = 0, uint32_t received = 0)
      : rank_id(rid),
        channel_id(cid),
        remote_mem(rmem),
        expected_chunk_count(expected),
        received_chunk_count(received) {}

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

enum class SendType { Send, Write };
struct EFASendRequest {
  std::shared_ptr<RegMemBlock> local_mem;
  std::shared_ptr<RemoteMemInfo> remote_mem;
  uint32_t from_rank_id;
  uint32_t to_rank_id;
  uint32_t channel_id;
  uint32_t imm_data;  // immediate data
  int64_t wr_id;
  bool need_signaled;  // Whether to use IBV_SEND_SIGNALED flag
  SendType send_type = SendType::Send;

  // Constructor
  EFASendRequest(std::shared_ptr<RegMemBlock> local,
                 std::shared_ptr<RemoteMemInfo> remote, uint32_t imm = 0,
                 bool signaled = true)
      : local_mem(local),
        remote_mem(remote),
        imm_data(imm),
        need_signaled(signaled) {}

  // Constructor from shared_ptr<EFASendRequest>
  EFASendRequest(std::shared_ptr<EFASendRequest> other,
                 std::shared_ptr<RegMemBlock> local, uint32_t imm = 0,
                 bool signaled = true)
      : local_mem(local),
        remote_mem(other->remote_mem),
        imm_data(imm),
        need_signaled(signaled) {}

  // Constructor from const EFASendRequest&
  EFASendRequest(EFASendRequest const& other,
                 std::shared_ptr<RegMemBlock> local, uint32_t imm = 0,
                 bool signaled = true)
      : local_mem(local),
        remote_mem(other.remote_mem),
        imm_data(imm),
        need_signaled(signaled) {}

  // Getter methods
  inline uint32_t getLocalKey() const { return local_mem->mr->lkey; }

  inline uint32_t getRemoteKey() const { return remote_mem->rkey; }

  inline uint64_t getLocalAddress() const {
    return reinterpret_cast<uint64_t>(local_mem->addr);
  }

  inline uint64_t getRemoteAddress() const { return remote_mem->addr; }

  inline uint32_t getImm() const { return imm_data; }

  inline uint32_t getLocalLen() const { return local_mem->size; }

  friend std::ostream& operator<<(std::ostream& os, EFASendRequest const& req) {
    os << "EFASendRequest{";
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

struct EFARecvRequest {
  uint32_t from_rank_id;
  uint32_t to_rank_id;
  uint32_t channel_id;
  int64_t wr_id;
  std::shared_ptr<RegMemBlock> local_mem;

  // Constructor
  EFARecvRequest(std::shared_ptr<RegMemBlock> local) : local_mem(local) {}

  // Getter methods
  inline uint32_t getLocalKey() const { return local_mem->mr->lkey; }

  inline uint64_t getLocalAddress() const {
    return reinterpret_cast<uint64_t>(local_mem->addr);
  }

  inline uint32_t getLocalLen() const { return local_mem->size; }

  friend std::ostream& operator<<(std::ostream& os, EFARecvRequest const& req) {
    os << "EFARecvRequest{";
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
};

template <typename T>
std::string serialize(const T& obj) {
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

  // Default constructor
  MetaInfoToExchange()
      : rank_id(0),
        channel_id(0),
        flag(ChannelType::Normal),
        channel_meta{},
        mem_meta{},
        gpu_id(0) {}

  // Constructor with required rank_id and channel_id, optional channel_meta and
  // mem_meta
  MetaInfoToExchange(int32_t rid, int32_t cid,
                     std::shared_ptr<ChannelMetaData> ch_meta = nullptr,
                     std::shared_ptr<RemoteMemInfo> mem_meta_ptr = nullptr,
                     ChannelType flag_in = ChannelType::Normal,
                     int gid = 0)
      : rank_id(rid),
        channel_id(cid),
        channel_meta{},
        mem_meta{},
        flag(flag_in),
        gpu_id(gid) {
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

    os << "  mem_meta:" << std::endl;
    os << "    addr: 0x" << std::hex << meta.mem_meta.addr << std::dec
       << std::endl;
    os << "    rkey: 0x" << std::hex << meta.mem_meta.rkey << std::dec
       << std::endl;
    os << "    length: " << meta.mem_meta.length << " bytes" << std::endl;
    os << "    type: "
       << (meta.mem_meta.type == MemoryType::HOST ? "HOST" : "GPU")
       << std::endl;

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

typedef struct AcceptedMeta {
  std::string ip;
  uint16_t port;
  int gpu_id;
  uint64_t rank_id;
} AcceptedMeta;

