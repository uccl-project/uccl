#pragma once
#include <infiniband/verbs.h>
#include <cstdint>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>
#include <memory>
#include <cstring>
#include <iostream>
#include <cassert>
#include <arpa/inet.h>

#include <sys/mman.h>

// Socket programming headers
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

// Standard library headers
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>  // for std::find
#include <iostream>   // for std::cout, std::cerr
#include <map>
#include <unordered_map>
#include <memory>

#include <cuda_runtime.h>
#include <stdexcept>

#define GID_INDEX 0
#define EFA_QP_LOW_LATENCY_SERVICE_LEVEL 8
#define QKEY 0x12345
#define PORT_NUM 1
#define EFA_RDM_DEFAULT_RNR_RETRY (3)

constexpr size_t RING_CAPACITY = 16384;  // Must be power of 2

struct ChannelMetaData {
    uint32_t qpn;
    union ibv_gid gid;
};

enum class MemoryType {
    HOST,
    GPU
};

enum class ChannelType : int16_t {
    Control,
    Normal
};

struct OOBMetaData {
    std::string server_ip;
    int64_t server_port;
};

struct CQMeta{
    uint64_t wr_id;
    ibv_wc_opcode op_code;
    uint32_t len;
    int imm;
    inline bool hasIMM(){
        return op_code == IBV_WC_RECV_RDMA_WITH_IMM;
    };
};

// Registered memory region info
typedef struct RegMemBlock {
    void* addr;
    size_t size;
    MemoryType type;
    struct ibv_mr* mr;
    bool pool_allocated;  // true if allocated by pool, false if externally provided
    RegMemBlock(void* a, size_t s, MemoryType t, struct ibv_mr* m, bool pool_alloc)
        : addr(a), size(s), type(t), mr(m), pool_allocated(pool_alloc) {};
} RegMemBlock;

typedef struct RemoteMemInfo {
    uint64_t addr;
    uint32_t rkey;
    size_t length;
    MemoryType type;

    // Default constructor
    RemoteMemInfo() : addr(0), rkey(0), length(0), type(MemoryType::HOST) {}

    // Constructor with all parameters
    RemoteMemInfo(uint64_t a, uint32_t k, size_t len, MemoryType t)
        : addr(a), rkey(k), length(len), type(t) {}

    // Constructor from RegMemBlock (copy constructor)
    RemoteMemInfo(const RegMemBlock& block)
        : addr(reinterpret_cast<uint64_t>(block.addr)),
          rkey(block.mr ? block.mr->rkey : 0),
          length(block.size),
          type(block.type) {}

    // Constructor from shared_ptr<RegMemBlock> (copy constructor)
    RemoteMemInfo(const std::shared_ptr<RegMemBlock> block)
        : addr(reinterpret_cast<uint64_t>(block->addr)),
          rkey(block->mr ? block->mr->rkey : 0),
          length(block->size),
          type(block->type) {}
} RemoteMemInfo;

enum class ReqFlag : int16_t {
    PENDING = 2,
    IN_PROGRESS = 3,
    IS_DONE = 4
};
struct alignas(64) SendReqMeta {
    uint32_t rank_id;
    uint32_t channel_id;
    RemoteMemInfo remote_mem;
};
struct alignas(64) SendReqMetaOnRing {
    SendReqMeta meta;
    std::atomic<ReqFlag> flag;

    SendReqMetaOnRing() : meta{}, flag{ReqFlag::PENDING} {}

    SendReqMetaOnRing(const SendReqMetaOnRing& other)
        : meta(other.meta), flag{other.flag.load(std::memory_order_relaxed)} {}

    SendReqMetaOnRing& operator=(const SendReqMetaOnRing& other) {
        meta = other.meta;
        flag.store(other.flag.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    // Get SendReqMeta from SendReqMetaOnRing
    SendReqMeta getSendReqMeta() const {
        return meta;
    }

    // Set SendReqMeta part
    void setSendReqMeta(const SendReqMeta& m) {
        meta = m;
    }
};

// Ring buffer size for control channel
constexpr size_t RING_BUFFER_SIZE = sizeof(SendReqMetaOnRing) * RING_CAPACITY;

// Helper functions for SendReqMetaOnRing to be used with modify_and_advance_write
inline auto check_in_progress = [](const SendReqMetaOnRing& item) {
    return item.flag.load(std::memory_order_relaxed) == ReqFlag::IN_PROGRESS;
};

inline auto check_is_done = [](const SendReqMetaOnRing& item) {
    return item.flag.load(std::memory_order_relaxed) == ReqFlag::IS_DONE;
};

inline auto set_in_progress = [](SendReqMetaOnRing& item) {
    item.flag.store(ReqFlag::IN_PROGRESS, std::memory_order_relaxed);
};

inline auto set_is_done = [](SendReqMetaOnRing& item) {
    item.flag.store(ReqFlag::IS_DONE, std::memory_order_relaxed);
};

// Conversion functions between SendReqMeta and SendReqMetaOnRing
inline auto to_ring_meta = [](const SendReqMeta& src, SendReqMetaOnRing& dst) {
    dst.meta = src;
    dst.flag.store(ReqFlag::PENDING, std::memory_order_relaxed);
};

inline auto from_ring_meta = [](const SendReqMetaOnRing& src, SendReqMeta& dst) {
    dst = src.meta;
};

struct EFASendRequest {
    std::shared_ptr<RegMemBlock> local_mem;
    std::shared_ptr<RemoteMemInfo> remote_mem;
    uint32_t from_rank_id;
    uint32_t to_rank_id;
    uint32_t channel_id;
    uint32_t imm_data;  // immediate data

    // Constructor
    EFASendRequest(std::shared_ptr<RegMemBlock> local,
                   std::shared_ptr<RemoteMemInfo> remote,
                   uint32_t imm = 0)
        : local_mem(local), remote_mem(remote), imm_data(imm) {}

    // Constructor from shared_ptr<EFASendRequest>
    EFASendRequest(std::shared_ptr<EFASendRequest> other,
                   std::shared_ptr<RegMemBlock> local,
                   uint32_t imm = 0)
        : local_mem(local), remote_mem(other->remote_mem), imm_data(imm) {}

    // Constructor from const EFASendRequest&
    EFASendRequest(const EFASendRequest& other,
                   std::shared_ptr<RegMemBlock> local,
                   uint32_t imm = 0)
        : local_mem(local), remote_mem(other.remote_mem), imm_data(imm) {}

    // Getter methods
    inline uint32_t getLocalKey() const {
        return local_mem->mr->lkey;
    }

    inline uint32_t getRemoteKey() const {
        return remote_mem->rkey;
    }

    inline uint64_t getLocalAddress() const {
        return reinterpret_cast<uint64_t>(local_mem->addr);
    }

    inline uint64_t getRemoteAddress() const {
        return remote_mem->addr;
    }

    inline uint32_t getImm() const {
        return imm_data;
    }

    inline uint32_t getLocalLen() const{
        return local_mem->size;
    } 
};

struct EFARecvRequest {
    uint32_t from_rank_id;
    uint32_t to_rank_id;
    uint32_t channel_id;
    std::shared_ptr<RegMemBlock> local_mem;

    // Constructor
    EFARecvRequest(std::shared_ptr<RegMemBlock> local)
        : local_mem(local){}

    // Getter methods
    inline uint32_t getLocalKey() const {
        return local_mem->mr->lkey;
    }

    inline uint64_t getLocalAddress() const {
        return reinterpret_cast<uint64_t>(local_mem->addr);
    }

    inline uint32_t getLocalLen() const{
        return local_mem->size;
    } 
};



template <typename T>
std::string serialize(const T& obj) {
    std::string s(sizeof(T), '\0');
    std::memcpy(&s[0], reinterpret_cast<const char*>(&obj), sizeof(T));
    return s;
}

template <typename T>
T deserialize(const std::string& s) {
    T obj{};
    size_t copy_len = std::min(s.size(), sizeof(T));
    std::memcpy(reinterpret_cast<char*>(&obj), s.data(), copy_len);
    return obj;
}

// ---------------------------
// Epoll server/client common definitions
// ---------------------------

// Example MetaInfo struct for metadata exchange
struct MetaInfoToExchange {
    int32_t rank_id;
    int32_t channel_id;
    ChannelType flag;
    ChannelMetaData channel_meta;
    RemoteMemInfo mem_meta;

    // Default constructor
    MetaInfoToExchange() : rank_id(0), channel_id(0),flag(ChannelType::Normal),channel_meta{}, mem_meta{} {}

    // Constructor with required rank_id and channel_id, optional channel_meta and mem_meta
    MetaInfoToExchange(int32_t rid, int32_t cid,
                       std::shared_ptr<ChannelMetaData> ch_meta = nullptr,
                       std::shared_ptr<RemoteMemInfo> mem_meta_ptr = nullptr, ChannelType flag_in = ChannelType::Normal)
        : rank_id(rid), channel_id(cid), channel_meta{}, mem_meta{}, flag(flag_in) {
        if (ch_meta) {
            channel_meta = *ch_meta;
        }
        if (mem_meta_ptr) {
            mem_meta = *mem_meta_ptr;
        }
    }

    // Overload << operator for printing
    friend std::ostream& operator<<(std::ostream& os, const MetaInfoToExchange& meta) {
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
        os << "    addr: 0x" << std::hex << meta.mem_meta.addr << std::dec << std::endl;
        os << "    rkey: 0x" << std::hex << meta.mem_meta.rkey << std::dec << std::endl;
        os << "    length: " << meta.mem_meta.length << " bytes" << std::endl;
        os << "    type: " << (meta.mem_meta.type == MemoryType::HOST ? "HOST" : "GPU") << std::endl;
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
static inline ssize_t try_send(int fd, const char* buf, size_t len) {
    ssize_t n = ::send(fd, buf, len, MSG_NOSIGNAL);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        return -1;
    }
    return n;
}



