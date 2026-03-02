#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

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

inline void serialize_fifo_item(FifoItem const& item, char* buf) {
  std::memcpy(buf + 0, &item.addr, sizeof(uint64_t));
  std::memcpy(buf + 8, &item.size, sizeof(uint32_t));
  std::memcpy(buf + 12, &item.rkey, sizeof(uint32_t));
  std::memcpy(buf + 16, &item.nmsgs, sizeof(uint32_t));
  std::memcpy(buf + 20, &item.rid, sizeof(uint32_t));
  std::memcpy(buf + 24, &item.idx, sizeof(uint64_t));
  std::memcpy(buf + 32, &item.padding, sizeof(item.padding));
}

inline void deserialize_fifo_item(char const* buf, FifoItem* item) {
  std::memcpy(&item->addr, buf + 0, sizeof(uint64_t));
  std::memcpy(&item->size, buf + 8, sizeof(uint32_t));
  std::memcpy(&item->rkey, buf + 12, sizeof(uint32_t));
  std::memcpy(&item->nmsgs, buf + 16, sizeof(uint32_t));
  std::memcpy(&item->rid, buf + 20, sizeof(uint32_t));
  std::memcpy(&item->idx, buf + 24, sizeof(uint64_t));
  std::memcpy(item->padding, buf + 32, sizeof(item->padding));
}
enum class MemoryType { HOST, GPU };

// IPC token constants.
// IPC_TOKEN_SIZE must equal sizeof(IpcTransferInfo) — verified by static_assert
// in uccl_engine.cc.  Layout: gpuIpcMemHandle_t(64) + uintptr_t(8) + size_t(8)
// + uint32_t(4) + bool(1) + 3-byte padding = 88 bytes.
#define IPC_TOKEN_SIZE 88
// CUDA IPC requires the base address of a cudaMalloc allocation to be aligned
// to this boundary (1 MiB) before calling cudaIpcGetMemHandle.
#define IPC_ALIGNMENT (1ul << 20)

// Unified memory token carrying both an RDMA FifoItem and a CUDA IPC handle.
// Exactly one of the two sub-tokens is used at transfer time, selected by
// uccl_engine based on conn->is_intra_node and has_ipc.
struct uccl_mem_token_t {
  char     fifo_buf[FIFO_SIZE];     // serialised RDMA FifoItem — always valid
  char     ipc_buf[IPC_TOKEN_SIZE]; // serialised IpcTransferInfo — valid iff has_ipc
  uint64_t base_addr;               // registered region base (used for IPC sub-buf patching)
  bool     has_ipc;                 // true iff buffer is VRAM and IPC handle was obtained
};

#endif