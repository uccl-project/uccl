#pragma once
#include "transport/endpoint_ops.h"
#include "transport/rdma_request.h"
template <>
struct EndpointOps<RDMAEndpoint> : DirectEndpointOps<RDMAEndpoint> {
  static bool uccl_regmr(std::shared_ptr<RDMAEndpoint> const& s, void* data,
                         size_t len, struct P2PMhandle* mhandle) {
    return s->uccl_regmr(data, len, mhandle->mr_array, mhandle->cache_refs,
                         mhandle->compress_ctx) >= 0;
  }

  static bool uccl_prepare_compress_ctx(std::shared_ptr<RDMAEndpoint> const& s,
                                        void* data, size_t len,
                                        CompressCtx const& compress_ctx) {
    if (!data) return false;
    s->prepare_compress_ctx(data, len, compress_ctx);
    return true;
  }

  static SendConnection* uccl_resolve_send_group(
      std::shared_ptr<RDMAEndpoint> const& s, uint64_t peer_id) {
    return s->get_send_group_raw(peer_id);
  }

  static int uccl_read_async(std::shared_ptr<RDMAEndpoint> const& s, Conn* conn,
                             P2PMhandle* local_mh, void* dst, size_t size,
                             FifoItem const& slot_item, UcclRequest* ureq) {
    ureq->type = ReqType::ReqRead;
    return set_request(s, conn, local_mh, dst, size, slot_item, ureq);
  }

  static int uccl_write_async(std::shared_ptr<RDMAEndpoint> const& s,
                              Conn* conn, P2PMhandle* local_mh, void* src,
                              size_t size, FifoItem const& slot_item,
                              UcclRequest* ureq) {
    ureq->type = ReqType::ReqWrite;
    return set_request(s, conn, local_mh, src, size, slot_item, ureq);
  }

  static int prepare_fifo_metadata(std::shared_ptr<RDMAEndpoint> const& s,
                                   P2PMhandle* mhandle, void const* data,
                                   size_t size, char* out_buf) {
    FifoItem remote_mem_info{};
    remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
    remote_mem_info.size = size;
    copy_rkeys_from_mr_array_to_bytes(
        mhandle->mr_array, static_cast<char*>(remote_mem_info.padding),
        sizeof(remote_mem_info.padding));
    serialize_fifo_item(remote_mem_info, out_buf);
    return 0;
  }

  static void uccl_deregmr(std::shared_ptr<RDMAEndpoint> const& s,
                           P2PMhandle* mhandle) {
    s->uccl_deregmr(mhandle->cache_refs);
  }

  static int uccl_get_numa_node(std::shared_ptr<RDMAEndpoint> const& s,
                                int gpu_idx) {
    return RdmaDeviceManager::instance().get_numa_node(
        RdmaDeviceManager::instance().get_best_dev_idx(gpu_idx)[0]);
  }

  static int uccl_send_notification(std::shared_ptr<RDMAEndpoint> const& s,
                                    uint64_t peer_id,
                                    NotifyMsg const& notification) {
    return -1;
  }
};
