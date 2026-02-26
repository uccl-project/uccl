// Simple RDMA context wrapper - no memory management
#pragma once
#include "define.h"
#include "rdma_device.h"

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
    LOG(INFO) << "GID[" << gid_index << "]: " << inet_ntoa(ip);
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
      LOG(INFO) << "Using GID index from environment: " << env_gid_index;
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
                  LOG(INFO) << "RoCE v2 device " << device_name
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
    LOG(INFO) << "Auto-detect GID failed, using default " << gid_index;
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

  struct ibv_mr* regMem(void* addr, size_t size) const {
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
