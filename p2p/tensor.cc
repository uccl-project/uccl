#include "tensor.h"
#include <cerrno>
#include <stdexcept>
#include <iostream>

std::shared_mutex mr_mapping_mu_;
std::unordered_map<uint64_t, std::unique_ptr<MR>> mr_mapping_;
std::atomic<uint64_t> next_mr_id_{0};

std::shared_mutex ipc_handle_mapping_mu_;
std::unordered_map<uint64_t, std::unique_ptr<IPCMemHandle>> ipc_handle_mapping_;
std::atomic<uint64_t> next_ipc_id_{0};

int reg_dma_mr(uccl::FactoryDevice* dev, void* addr, size_t len, int type, int offset,
               int fd, struct uccl::Mhandle** mhandle) {
    bool ib_relaxed_ordering_enabled_ = uccl::ncclIbRelaxedOrderingCapable();

    unsigned int flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ib_relaxed_ordering_enabled_) flags |= IBV_ACCESS_RELAXED_ORDERING;

    *mhandle = new uccl::Mhandle();
    (*mhandle)->mr = ibv_reg_dmabuf_mr(dev->pd, offset, len,
                                        (uint64_t)addr, fd, flags);
    return 0;
}

int reg_mr(uccl::FactoryDevice* dev, void* addr, size_t len, struct uccl::Mhandle** mhandle) {
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
    GPU_RT_CHECK(gpuIpcGetMemHandle(&ipchandle->handle, reinterpret_cast<void*>(addr)));
    return 0;
}

int open_ipc_handle(void* addr, struct IPCMemHandle* ipchandle) {
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&addr, ipchandle->handle,
                                   gpuIpcMemLazyEnablePeerAccess));
    return 0;
}

torch::Dtype torch_dtype_from_size(size_t dtype_size) {
    switch (dtype_size) {
    case 1:
        return torch::kInt8;
    case 2:
        return torch::kInt16;
    case 4:
        return torch::kInt32;
    case 8:
        return torch::kInt64;
    default:
        throw std::runtime_error("Unsupported dtype size: " +
                                 std::to_string(dtype_size));
    }
}

torch::Tensor create_tensor(int gpu_index, size_t num_elems, size_t dtype_size,
                           uint64_t& mr_id, uint64_t& ipc_id, bool requires_grad) {
    std::cout << "[create_tensor] gpu_index=" << gpu_index
              << " num_elems=" << num_elems
              << " dtype_size=" << dtype_size
              << " requires_grad=" << requires_grad << std::endl;

    GPU_RT_CHECK(gpuSetDevice(gpu_index));
    uccl::FactoryDevice* factory_dev = uccl::RDMAFactory::get_factory_dev(gpu_to_dev[gpu_index]);
    std::cout << "[create_tensor] Got factory_dev for gpu_index " << gpu_index << std::endl;

    size_t bytes = num_elems * dtype_size;
    size_t alignment = kIpcAlignment;
    std::cout << "[create_tensor] Allocating " << bytes << " bytes (aligned to " << alignment << ")" << std::endl;

    void* raw_ptr;
    GPU_RT_CHECK(gpuMalloc(&raw_ptr, bytes + alignment));
    std::cout << "[create_tensor] gpuMalloc success, raw_ptr=" << raw_ptr << std::endl;

    uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(raw_ptr) + alignment - 1) & ~(alignment - 1);
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
    std::cout << "[create_tensor] Aligned pointer=" << aligned_ptr << std::endl;

    // Tensor
    auto dtype_ = torch_dtype_from_size(dtype_size);
    auto dev = torch::Device(torch_dev, gpu_index);
    auto options = torch::TensorOptions().dtype(dtype_).device(dev).requires_grad(requires_grad);
    auto deleter = [raw_ptr](void* ptr) {
        std::cout << "[create_tensor] Deleter freeing raw_ptr=" << raw_ptr << std::endl;
        GPU_RT_CHECK(gpuFree(raw_ptr));
    };
    torch::Tensor tensor = torch::from_blob(aligned_ptr, {static_cast<long>(num_elems)}, deleter, options);
    std::cout << "[create_tensor] Torch tensor created: sizes=" << tensor.sizes() << std::endl;

    // MR
    std::unique_ptr<MR> mr = std::make_unique<MR>();
    int ret = reg_mr(factory_dev, aligned_ptr, bytes, &mr->mhandle_);
    if (ret != 0) {
        GPU_RT_CHECK(gpuFree(raw_ptr));
        throw std::runtime_error("MR registration failed");
    }
    mr->mr_id_ = next_mr_id_.fetch_add(1);
    {
        std::unique_lock<std::shared_mutex> lock(mr_mapping_mu_);
        mr_mapping_[mr->mr_id_] = std::move(mr);
    }
    mr_id = mr->mr_id_;
    std::cout << "[create_tensor] MR registered, mr_id=" << mr_id << std::endl;

    // IPC
    std::unique_ptr<IPCMemHandle> ipc = std::make_unique<IPCMemHandle>();
    ret = get_ipc_handle(aligned_ptr, ipc.get());
    if (ret != 0) {
        {
            std::unique_lock<std::shared_mutex> lock(mr_mapping_mu_);
            mr_mapping_.erase(mr_id);
        }
        GPU_RT_CHECK(gpuFree(raw_ptr));
        throw std::runtime_error("IPC handle creation failed");
    }
    ipc->id = next_ipc_id_.fetch_add(1);
    ipc->size = bytes;
    {
        std::unique_lock<std::shared_mutex> lock(ipc_handle_mapping_mu_);
        ipc_handle_mapping_[ipc->id] = std::move(ipc);
    }
    ipc_id = ipc->id;
    std::cout << "[create_tensor] IPC handle created, ipc_id=" << ipc_id
              << " size=" << bytes << std::endl;

    std::cout << "[create_tensor] SUCCESS: returning tensor with mr_id=" << mr_id
              << " ipc_id=" << ipc_id << std::endl;

    return tensor;
}


void free_tensor(torch::Tensor& tensor, uint64_t mr_id, uint64_t ipc_id) {
    if (tensor.defined()) {
        tensor.reset();
    }

    {
        std::unique_lock<std::shared_mutex> lock(mr_mapping_mu_);
        auto it = mr_mapping_.find(mr_id);
        if (it != mr_mapping_.end()) {
            dereg_mr(it->second->mhandle_);
            mr_mapping_.erase(it);
        } else {
            std::cerr << "[free_tensor] MR id " << mr_id << " not found!\n";
        }
    }

    {
        std::unique_lock<std::shared_mutex> lock(ipc_handle_mapping_mu_);
        auto it = ipc_handle_mapping_.find(ipc_id);
        if (it != ipc_handle_mapping_.end()) {
            ipc_handle_mapping_.erase(it);
        } else {
            std::cerr << "[free_tensor] IPC id " << ipc_id << " not found!\n";
        }
    }
}

ibv_mr* get_mr_ibv_mr(uint64_t mr_id) {
    std::shared_lock<std::shared_mutex> lock(mr_mapping_mu_);
    auto it = mr_mapping_.find(mr_id);
    if (it != mr_mapping_.end()) {
        return it->second->mhandle_->mr;
    }
    return nullptr;
}

gpuIpcMemHandle_t get_ipc_mem_handle(uint64_t ipc_id) {
    std::shared_lock<std::shared_mutex> lock(ipc_handle_mapping_mu_);
    auto it = ipc_handle_mapping_.find(ipc_id);
    if (it != ipc_handle_mapping_.end()) {
        return it->second->handle;
    }
    gpuIpcMemHandle_t handle = {};
    return handle;
}