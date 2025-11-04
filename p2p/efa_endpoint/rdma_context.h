// rdma_context.h
#pragma once
#include "rdma_device.h"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <memory>

class RdmaContext {
 public:
  explicit RdmaContext(std::shared_ptr<RdmaDevice> dev) {
    ctx_ = dev->open();
    if (!ctx_) throw std::runtime_error("Failed to open context");

    struct ibv_pd* pd = ibv_alloc_pd(ctx_.get());
    if (!pd) throw std::runtime_error("Failed to alloc pd");

    pd_.reset(pd, [](ibv_pd* p) { ibv_dealloc_pd(p); });
  }

  struct ibv_context* ctx() const { return ctx_.get(); }
  struct ibv_pd* pd() const { return pd_.get(); }

  union ibv_gid queryGid(int gid_index) {
    union ibv_gid gid{};
    if (ibv_query_gid(ctx_.get(), 1, gid_index, &gid)) {
      perror("ibv_query_gid");
      throw std::runtime_error("query_gid failed");
    }
    auto ip = *(struct in_addr*)&gid.raw[8];
    std::cout << "GID[" << gid_index << "]: " << inet_ntoa(ip) << std::endl;
    return gid;
  }

  struct ibv_ah* createAh(union ibv_gid remote_gid, int gid_index) {
    struct ibv_ah_attr attr = {};
    attr.is_global = 1;
    attr.port_num = 1;
    attr.grh.dgid = remote_gid;
    struct ibv_ah* ah = ibv_create_ah(pd_.get(), &attr);
    if (!ah) throw std::runtime_error("create_ah failed");
    return ah;
  }

 private:
  std::shared_ptr<struct ibv_context> ctx_;
  std::shared_ptr<struct ibv_pd> pd_;
};
