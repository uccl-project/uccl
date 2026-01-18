// Simple RDMA context wrapper - no memory management
#pragma once
#include "define.h"
#include "rdma_device.h"

class RdmaContext {
 public:
  explicit RdmaContext(std::shared_ptr<RdmaDevice> dev,
                       uint64_t context_id = 0) {
    context_id_ = context_id;
    ctx_ = dev->open();
    if (!ctx_) throw std::runtime_error("Failed to open context");

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

  // Query GID by index
  void getGID(int gid_index, union ibv_gid* gid, int port = 1) const {
    if (ibv_query_gid(ctx_.get(), port, gid_index, gid)) {
      perror("ibv_query_gid");
      throw std::runtime_error("query_gid failed");
    }
    auto ip = *(struct in_addr*)&gid->raw[8];
    std::cout << "GID[" << gid_index << "]: " << inet_ntoa(ip) << std::endl;
  }

  union ibv_gid queryGid(int gid_index, int port = 1) const {
    union ibv_gid gid {};
    getGID(gid_index, &gid, port);
    return gid;
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

  struct ibv_ah* createAh(union ibv_gid remote_gid, int port = 1) const {
    return createAH(remote_gid, port);
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
};
