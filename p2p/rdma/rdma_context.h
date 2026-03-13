// Simple RDMA context wrapper - no memory management
#pragma once
#include "define.h"
#include "rdma_device.h"
#include <dlfcn.h>

class RdmaContext {
 public:
  explicit RdmaContext(std::shared_ptr<RdmaDevice> dev, uint64_t context_id = 0)
      : gid_index_(-1) {
    context_id_ = context_id;
    ctx_ = dev->open();
    if (!ctx_) throw std::runtime_error("Failed to open context");

    struct ibv_device_attr dev_attr;
    assert(ibv_query_device(ctx_.get(), &dev_attr) == 0);
    vendor_id_ = dev_attr.vendor_id;

    struct ibv_pd* pd = ibv_alloc_pd(ctx_.get());
    if (!pd) throw std::runtime_error("Failed to alloc pd");

    pd_.reset(pd, [](ibv_pd* p) { ibv_dealloc_pd(p); });
  }

  // Getters
  struct ibv_context* getCtx() const {
    return ctx_.get();
  }
  struct ibv_context* ctx() const {
    return ctx_.get();
  }
  struct ibv_pd* getPD() const {
    return pd_.get();
  }
  struct ibv_pd* pd() const {
    return pd_.get();
  }

  uint32_t getVendorID() const { return vendor_id_; }

  // Query GID by index
  void getGID(int gid_index, union ibv_gid* gid, int port = 1) const {
    if (ibv_query_gid(ctx_.get(), port, gid_index, gid)) {
      perror("ibv_query_gid");
      throw std::runtime_error("query_gid failed");
    }
    auto ip = *(struct in_addr*)&gid->raw[8];
    UCCL_LOG(INFO, UCCL_RDMA) << "GID[" << gid_index << "]: " << inet_ntoa(ip);
  }

  union ibv_gid queryGid(int gid_index, int port = 1) const {
    union ibv_gid gid {};
    getGID(gid_index, &gid, port);
    return gid;
  }

  union ibv_gid detectGid(int gid_index, int port = 1) const {
    struct ibv_port_attr port_attr;

    char const* env = getenv("UCCL_P2P_RDMA_GID_INDEX");
    if (env) {
      int env_gid_index = std::atoi(env);
      UCCL_LOG(INFO, UCCL_RDMA)
          << "Using GID index from environment: " << env_gid_index;
      gid_index_ = env_gid_index;
      return queryGid(gid_index_);
    }

    if (ibv_query_port(ctx_.get(), port, &port_attr)) {
      throw std::runtime_error("query_port failed");
    }

    char const* device_name = ibv_get_device_name(ctx_.get()->device);

    if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
      union ibv_gid gid;
      if (ibv_query_gid(ctx_.get(), port, 0, &gid) == 0) {
        if (gid.global.subnet_prefix != 0 || gid.global.interface_id != 0) {
          gid_index_ = 0;
          return gid;
        }
      }
      throw std::runtime_error("query_gid failed");
    }

    if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
      for (int i = 0; i < port_attr.gid_tbl_len; i++) {
        union ibv_gid gid;
        if (ibv_query_gid(ctx_.get(), port, i, &gid) == 0) {
          if (gid_index == 0) {  // EFA
            gid_index_ = i;
            return gid;
          }

          if (gid.global.subnet_prefix != 0 || gid.global.interface_id != 0) {
            char gid_type_path[512];
            snprintf(gid_type_path, sizeof(gid_type_path),
                     "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
                     device_name, port, i);

            FILE* fp = fopen(gid_type_path, "r");
            if (fp) {
              char gid_type[64];
              if (fgets(gid_type, sizeof(gid_type), fp)) {
                if (strstr(gid_type, "RoCE v2") != nullptr) {
                  fclose(fp);
                  UCCL_LOG(INFO, UCCL_RDMA) << "RoCE v2 device " << device_name
                                            << ": using GID index " << i;
                  gid_index_ = i;
                  return gid;
                }
              }
              fclose(fp);
            }
          }
        }
      }
    }

    // On p5 EFA, the link_layer is IBV_LINK_LAYER_UNSPECIFIED.
    UCCL_LOG(INFO, UCCL_RDMA)
        << "Auto-detect GID failed, using default " << gid_index;
    gid_index_ = gid_index;
    return queryGid(gid_index_, port);
  }

  int getGidIndex(int gid_index, int port = 1) const {
    union ibv_gid gid;
    // Return cached value if available
    if (gid_index_ >= 0) {
      return gid_index_;
    }

    gid = detectGid(gid_index, port);

    return gid_index_;
  }

  uint16_t queryLid(int port = 1) const {
    struct ibv_port_attr port_attr;
    assert(ibv_query_port(ctx_.get(), port, &port_attr) == 0);
    return port_attr.lid;
  }

  // Create address handle from remote GID
  struct ibv_ah* createAH(union ibv_gid remote_gid, int port = 1) const {
    struct ibv_ah_attr attr = {};
    attr.is_global = 1;
    attr.port_num = port;
    attr.grh.dgid = remote_gid;
    struct ibv_ah* ah = ibv_create_ah(pd_.get(), &attr);
    if (!ah) throw std::runtime_error("create_ah failed");
    return ah;
  }

  // Check if a pointer refers to GPU device memory.
  static bool isGpuPointer(void* ptr) {
    gpuPointerAttribute_t attrs = {};
    gpuError_t err = gpuPointerGetAttributes(&attrs, ptr);
    if (err != gpuSuccess) {
      (void)gpuGetLastError();  // clear sticky error
      return false;
    }
    return (attrs.type == gpuMemoryTypeDevice);
  }

  // Register GPU memory via DMA-BUF for GPUDirect RDMA.
  // Uses kernel DMA-BUF subsystem instead of nvidia_peermem.
  // Returns nullptr on failure so the caller can report the error.
  struct ibv_mr* regMemGpuDmabuf(void* addr, size_t size) const {
    // GPU page granularity for DMA-BUF export (2 MiB on modern GPUs).
    static constexpr size_t kDmabufGranularity = 2ULL << 20;  // 2 MiB

    static gpuMemGetHandleForAddressRange_fn gpuGetHandleForRange_func =
        nullptr;
    static std::once_flag init_flag;

    std::call_once(init_flag, []() {
      void* handle = dlopen(GPU_DRIVER_LIB_NAME, RTLD_LAZY);
      if (!handle) handle = dlopen(GPU_DRIVER_LIB_NAME_FALLBACK, RTLD_LAZY);
      if (handle) {
        gpuGetHandleForRange_func = (gpuMemGetHandleForAddressRange_fn)dlsym(
            handle, GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME);
      }
    });

    if (!gpuGetHandleForRange_func) {
      UCCL_LOG(ERROR) << "GPU Driver API not available for DMA-BUF";
      return nullptr;
    }

    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ;

    // The DMA-BUF export API requires the address to be at the start of a
    // GPU allocation and the size aligned to GPU page granularity. The caller
    // may pass a sub-region, so we find the allocation base and compute offset.
    void* alloc_base_ptr = nullptr;
    size_t alloc_size = 0;
    gpuError_t gpu_err =
        gpuMemGetAddressRange(&alloc_base_ptr, &alloc_size, addr);
    if (gpu_err != gpuSuccess) {
      UCCL_LOG(ERROR) << "gpuMemGetAddressRange failed ("
                      << gpuGetErrorString(gpu_err) << ")";
      return nullptr;
    }

    gpuDevicePtr_t alloc_base = (gpuDevicePtr_t)(uintptr_t)alloc_base_ptr;
    size_t offset_in_alloc = (uintptr_t)addr - (uintptr_t)alloc_base;

    // Export the entire allocation as a DMA-BUF fd (aligned to granularity).
    size_t export_size =
        ((alloc_size + kDmabufGranularity - 1) / kDmabufGranularity) *
        kDmabufGranularity;

    UCCL_LOG(INFO, UCCL_RDMA)
        << "DMA-BUF: gpu_buf=" << addr << " bytes=" << size << " alloc_base=0x"
        << std::hex << (uintptr_t)alloc_base << std::dec
        << " alloc_size=" << alloc_size << " offset=" << offset_in_alloc
        << " export_size=" << export_size;

    // Export DMA-BUF fd from the allocation base.
    int dmabuf_fd = -1;
    gpuDriverResult_t drv_err =
        gpuGetHandleForRange_func(&dmabuf_fd, alloc_base, export_size,
                                  GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);

    if (drv_err != gpuDriverSuccess) {
      UCCL_LOG(ERROR) << "gpuMemGetHandleForAddressRange failed (error="
                      << (int)drv_err << "), DMA-BUF unavailable";
      return nullptr;
    }

    // Register only the sub-region we need using fd_offset.
    uint64_t iova = (uint64_t)addr;
    ibv_mr* mr = ibv_reg_dmabuf_mr(pd_.get(), offset_in_alloc, size, iova,
                                   dmabuf_fd, access_flags);
    int saved_errno = errno;
    close(dmabuf_fd);  // fd can be closed after registration

    if (!mr) {
      UCCL_LOG(ERROR) << "ibv_reg_dmabuf_mr failed (errno=" << saved_errno
                      << ": " << strerror(saved_errno) << ")";
      return nullptr;
    }

    // ibv_reg_dmabuf_mr sets mr->addr to the offset, not the GPU VA.
    // Override to match ibv_reg_mr_iova2 behavior.
    mr->addr = reinterpret_cast<void*>(iova);

    UCCL_LOG(INFO, UCCL_RDMA)
        << "Registered GPU memory via DMA-BUF (addr=" << addr
        << ", len=" << size << ", rkey=0x" << std::hex << mr->rkey << std::dec
        << ") via DMA-BUF GPUDirect RDMA";
    return mr;
  }

  struct ibv_mr* regMem(void* addr, size_t size) const {
    // Intel irdma (0x8086) uses DMA-BUF for GPUDirect RDMA
    // instead of nvidia_peermem.
    if (vendor_id_ == 0x8086 && isGpuPointer(addr)) {
      UCCL_LOG(INFO, UCCL_RDMA)
          << "GPU memory detected on irdma NIC (vendor=0x" << std::hex
          << vendor_id_ << std::dec << "), using DMA-BUF registration";
      return regMemGpuDmabuf(addr, size);
    }

    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ;
    return ibv_reg_mr(pd_.get(), addr, size, access_flags);
  }

  static void deregMem(struct ibv_mr* mr) {
    if (mr) ibv_dereg_mr(mr);
  }
  inline const uint64_t getContextID() const { return context_id_; }

 private:
  std::shared_ptr<struct ibv_context> ctx_;
  std::shared_ptr<struct ibv_pd> pd_;
  uint64_t context_id_;
  uint32_t vendor_id_;
  mutable int gid_index_;
};
