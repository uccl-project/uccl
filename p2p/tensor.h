#pragma once
#include <glog/logging.h>
#undef CHECK
#undef DCHECK
#undef VLOG_IS_ON
#undef LOG
#undef VLOG
#undef LOG_IF
#undef VLOG_IF
#include <torch/torch.h>
#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include "transport.h"
#include "rdma_io.h"
#include "util/gpu_rt.h"
#include "engine.h"

#ifndef __HIP_PLATFORM_AMD__
#define torch_dev torch::kCUDA
#else
#define torch_dev torch::kHIP
#endif

static constexpr size_t kIpcAlignment = 1ul << 20;

struct IPCMemHandle {
    uint64_t id;
    gpuIpcMemHandle_t handle;
    size_t size;
};

// internal
extern std::shared_mutex mr_mapping_mu_;
extern std::unordered_map<uint64_t, std::unique_ptr<MR>> mr_mapping_;
extern std::atomic<uint64_t> next_mr_id_;

extern std::shared_mutex ipc_handle_mapping_mu_;
extern std::unordered_map<uint64_t, std::unique_ptr<IPCMemHandle>> ipc_handle_mapping_;
extern std::atomic<uint64_t> next_ipc_id_;

int reg_dma_mr(uccl::FactoryDevice* dev, void* addr, size_t len, int type, int offset,
               int fd, struct uccl::Mhandle** mhandle);

int reg_mr(uccl::FactoryDevice* dev, void* addr, size_t len, struct uccl::Mhandle** mhandle);

void dereg_mr(struct uccl::Mhandle* mhandle);

int get_ipc_handle(void* addr, struct IPCMemHandle* ipchandle);

int open_ipc_handle(void* addr, struct IPCMemHandle* ipchandle);

torch::Dtype torch_dtype_from_size(size_t dtype_size);

// API: for pybind
torch::Tensor create_tensor(int gpu_index, size_t num_elems, size_t dtype_size,
                           uint64_t& mr_id, uint64_t& ipc_id, bool requires_grad = false);
void free_tensor(torch::Tensor& tensor, uint64_t mr_id, uint64_t ipc_id);

// for engine Endpoint
ibv_mr* get_mr_ibv_mr(uint64_t mr_id);
gpuIpcMemHandle_t get_ipc_mem_handle(uint64_t ipc_id);