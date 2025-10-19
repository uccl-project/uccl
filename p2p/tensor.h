#pragma once
#include "engine.h"
#include "rdma_io.h"
#include "transport.h"
#include "util/gpu_rt.h"
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

static constexpr size_t kIpcAlignment = 1ul << 20;

struct MR {
  uint64_t mr_id_;
  uccl::Mhandle* mhandle_;
};

struct IPCMemHandle {
  uint64_t id;
  gpuIpcMemHandle_t handle;
  size_t size;
  size_t offset;
};

// internal
extern std::shared_mutex mr_mapping_mu_;
extern std::unordered_map<uint64_t, std::unique_ptr<MR>> mr_mapping_;

extern std::shared_mutex ipc_handle_mapping_mu_;
extern std::unordered_map<uint64_t, std::unique_ptr<IPCMemHandle>>
    ipc_handle_mapping_;

extern std::atomic<uint64_t> next_mem_id_;

int reg_dma_mr(uccl::FactoryDevice* dev, void* addr, size_t len, int offset,
               int fd, struct uccl::Mhandle** mhandle);

int reg_mr(uccl::FactoryDevice* dev, void* addr, size_t len,
           struct uccl::Mhandle** mhandle);

void dereg_mr(struct uccl::Mhandle* mhandle);

int get_ipc_handle(void* addr, struct IPCMemHandle* ipchandle);

int open_ipc_handle(void* addr, struct IPCMemHandle* ipchandle);

// API: for pybind
void reg_mem(void* addr, size_t size, uint64_t& mem_id, bool is_device = true,
             int gpu_id = 0);
void dereg_mem(uint64_t mem_id);

// for engine Endpoint
uccl::Mhandle* get_mr_handle_by_mem_id(uint64_t mem_id);
gpuIpcMemHandle_t get_ipc_mem_handle_by_mem_id(uint64_t mem_id);