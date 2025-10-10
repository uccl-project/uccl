#include "tensor.h"
#include <cerrno>
#include <iostream>
#include <stdexcept>

std::shared_mutex mr_mapping_mu_;
std::unordered_map<uint64_t, std::unique_ptr<MR>> mr_mapping_;

std::shared_mutex ipc_handle_mapping_mu_;
std::unordered_map<uint64_t, std::unique_ptr<IPCMemHandle>> ipc_handle_mapping_;

std::atomic<uint64_t> next_mem_id_{0};

int reg_dma_mr(uccl::FactoryDevice* dev, void* addr, size_t len, int offset,
               int fd, struct uccl::Mhandle** mhandle) {
  bool ib_relaxed_ordering_enabled_ = uccl::ncclIbRelaxedOrderingCapable();

  unsigned int flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ib_relaxed_ordering_enabled_) flags |= IBV_ACCESS_RELAXED_ORDERING;

  *mhandle = new uccl::Mhandle();
  (*mhandle)->mr =
      ibv_reg_dmabuf_mr(dev->pd, offset, len, (uint64_t)addr, fd, flags);
  return 0;
}

int reg_mr(uccl::FactoryDevice* dev, void* addr, size_t len,
           struct uccl::Mhandle** mhandle) {
  bool ib_relaxed_ordering_enabled_ = uccl::ncclIbRelaxedOrderingCapable();

  unsigned int flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ib_relaxed_ordering_enabled_) flags |= IBV_ACCESS_RELAXED_ORDERING;

  *mhandle = new uccl::Mhandle();
  if (ib_relaxed_ordering_enabled_) {
    (*mhandle)->mr =
        ibv_reg_mr_iova2(dev->pd, addr, len, (uint64_t)addr, flags);
  } else {
    (*mhandle)->mr = ibv_reg_mr(dev->pd, addr, len, flags);
  }
  if (!(*mhandle)->mr) {
    std::cerr << "ibv_reg_mr failed (" << strerror(errno) << "), len=" << len
              << " addr=" << addr << "\n";
    delete *mhandle;
    *mhandle = nullptr;
    return -1;
  }
  return 0;
}

void dereg_mr(struct uccl::Mhandle* mhandle) {
  ibv_dereg_mr(mhandle->mr);
  delete mhandle;
}

int get_ipc_handle(void* addr, struct IPCMemHandle* ipchandle) {
  GPU_RT_CHECK(
      gpuIpcGetMemHandle(&ipchandle->handle, reinterpret_cast<void*>(addr)));
  return 0;
}

int open_ipc_handle(void* addr, struct IPCMemHandle* ipchandle) {
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&addr, ipchandle->handle,
                                   gpuIpcMemLazyEnablePeerAccess));
  return 0;
}

void reg_mem(int gpu_id, void* addr, size_t size, uint64_t& mem_id) {
  if (gpu_id < 0 || gpu_id >= kMaxNumGPUs) {
    throw std::invalid_argument("[reg_mem] Invalid gpu_id: " +
                                std::to_string(gpu_id));
  }

  if (gpu_to_dev[gpu_id] == 0) {
    throw std::runtime_error(
        "You must initialize UCCL collective context or Endpoint first");
  }

  GPU_RT_CHECK(gpuSetDevice(gpu_id));

  uccl::FactoryDevice* factory_dev =
      uccl::RDMAFactory::get_factory_dev(gpu_to_dev[gpu_id]);

  mem_id = next_mem_id_.fetch_add(1);
  // MR
  std::unique_ptr<MR> mr = std::make_unique<MR>();
  int ret = reg_mr(factory_dev, addr, size, &mr->mhandle_);
  if (ret != 0) {
    throw std::runtime_error("MR registration failed");
  }
  mr->mr_id_ = mem_id;
  {
    std::unique_lock<std::shared_mutex> lock(mr_mapping_mu_);
    mr_mapping_[mr->mr_id_] = std::move(mr);
  }

  // IPC
  auto addr_aligned = reinterpret_cast<uintptr_t>(addr) & ~(kIpcAlignment - 1);
  auto addr_offset = reinterpret_cast<uintptr_t>(addr) - addr_aligned;
  //   std::cout << "[reg_mem] Aligned pointer=" << addr_aligned << std::endl;

  std::unique_ptr<IPCMemHandle> ipc = std::make_unique<IPCMemHandle>();
  ret = get_ipc_handle(reinterpret_cast<void*>(addr_aligned), ipc.get());
  if (ret != 0) {
    {
      std::unique_lock<std::shared_mutex> lock(mr_mapping_mu_);
      mr_mapping_.erase(mem_id);
    }
    throw std::runtime_error("[reg_mem] IPC handle creation failed");
  }
  ipc->size = size;
  ipc->offset = addr_offset;
  ipc->id = mem_id;
  {
    std::unique_lock<std::shared_mutex> lock(ipc_handle_mapping_mu_);
    ipc_handle_mapping_[ipc->id] = std::move(ipc);
  }
}

void dereg_mem(uint64_t mem_id) {
  {
    std::unique_lock<std::shared_mutex> lock(mr_mapping_mu_);
    auto it = mr_mapping_.find(mem_id);
    if (it != mr_mapping_.end()) {
      dereg_mr(it->second->mhandle_);
      mr_mapping_.erase(it);
    } else {
      std::cerr << "[free_tensor] MR id " << mem_id << " not found!\n";
    }
  }
  {
    std::unique_lock<std::shared_mutex> lock(ipc_handle_mapping_mu_);
    auto it = ipc_handle_mapping_.find(mem_id);
    if (it != ipc_handle_mapping_.end()) {
      ipc_handle_mapping_.erase(it);
    } else {
      std::cerr << "[free_tensor] IPC id " << mem_id << " not found!\n";
    }
  }
}

uccl::Mhandle* get_mr_handle_by_mem_id(uint64_t mem_id) {
  std::shared_lock<std::shared_mutex> lock(mr_mapping_mu_);
  auto it = mr_mapping_.find(mem_id);
  if (it != mr_mapping_.end()) {
    return it->second->mhandle_;
  }
  return nullptr;
}

gpuIpcMemHandle_t get_ipc_mem_handle_by_mem_id(uint64_t mem_id) {
  std::shared_lock<std::shared_mutex> lock(ipc_handle_mapping_mu_);
  auto it = ipc_handle_mapping_.find(mem_id);
  if (it != ipc_handle_mapping_.end()) {
    return it->second->handle;
  }
  gpuIpcMemHandle_t handle = {};
  return handle;
}