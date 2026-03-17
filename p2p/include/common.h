#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// Minimal GPU runtime includes for gpuIpcMemHandle_t (used by IpcTransferInfo)
#ifndef __HIP_PLATFORM_AMD__
#include <cuda_runtime.h>
#ifndef gpuIpcMemHandle_t
#define gpuIpcMemHandle_t cudaIpcMemHandle_t
#endif
#else
#include <hip/hip_runtime.h>
#ifndef gpuIpcMemHandle_t
#define gpuIpcMemHandle_t hipIpcMemHandle_t
#endif
#endif

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

// IPC transfer info for cross-process and same-process GPU transfers
struct IpcTransferInfo {
  gpuIpcMemHandle_t handle;
  uintptr_t offset;
  size_t size;
  uint32_t operation;  // 0 = send_ipc request, 1 = recv_ipc response
  bool is_host;        // true if this side's buffer is CPU memory
  int gpu_idx = -1;    // target GPU index; -1 = use conn->remote_gpu_idx_
  uintptr_t direct_addr = 0;  // same-process: skip IPC, use this virtual addr
};

#endif