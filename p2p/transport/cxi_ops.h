#pragma once
#include "transport/endpoint_ops.h"
template <>
struct EndpointOps<CxiEndpoint> : DirectEndpointOps<CxiEndpoint> {
  static bool uccl_regmr(std::shared_ptr<CxiEndpoint> const& s, void* data,
                         size_t len, struct P2PMhandle* mhandle) {
    return s->uccl_regmr(data, len, mhandle->cxi_region) >= 0;
  }

  static bool uccl_prepare_compress_ctx(std::shared_ptr<CxiEndpoint> const& s,
                                        void* data, size_t len,
                                        CompressCtx const& compress_ctx) {
    (void)s;
    (void)data;
    (void)len;
    (void)compress_ctx;
    return false;
  }

  static SendConnection* uccl_resolve_send_group(
      std::shared_ptr<CxiEndpoint> const& s, uint64_t peer_id) {
    return nullptr;
  }

  // The target NIC completes RMA without software, and the CQ holds only
  // completions of this endpoint's own operations, which the waiting threads
  // poll. So the idle receive proxy polls only when the target needs progress.
  static void uccl_drive_recv_idle(std::shared_ptr<CxiEndpoint> const& s) {
    if (s->needs_target_progress()) s->recv_routine();
  }

  static int uccl_read_async(std::shared_ptr<CxiEndpoint> const& s, Conn* conn,
                             P2PMhandle* local_mh, void* dst, size_t size,
                             FifoItem const& slot_item, UcclRequest* ureq) {
    ureq->type = ReqType::ReqRead;
    return s->uccl_read_async(conn->uccl_conn_id_, local_mh->cxi_region, dst,
                              size, slot_item, ureq);
  }

  static int uccl_write_async(std::shared_ptr<CxiEndpoint> const& s, Conn* conn,
                              P2PMhandle* local_mh, void* src, size_t size,
                              FifoItem const& slot_item, UcclRequest* ureq) {
    ureq->type = ReqType::ReqWrite;
    return s->uccl_write_async(conn->uccl_conn_id_, local_mh->cxi_region, src,
                               size, slot_item, ureq);
  }

  static int prepare_fifo_metadata(std::shared_ptr<CxiEndpoint> const& s,
                                   P2PMhandle* mhandle, void const* data,
                                   size_t size, char* out_buf) {
    (void)s;
    if (!mhandle->cxi_region) return -1;
    FifoItem remote_mem_info;
    remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
    remote_mem_info.size = size;
    remote_mem_info.rkey = 0;
    remote_mem_info.nmsgs = 0;
    remote_mem_info.rid = 0;
    remote_mem_info.idx = 0;
    encode_cxi_fifo_metadata(*mhandle->cxi_region, remote_mem_info);
    serialize_fifo_item(remote_mem_info, out_buf);
    return 0;
  }

  static void uccl_deregmr(std::shared_ptr<CxiEndpoint> const& s,
                           P2PMhandle* mhandle) {
    s->uccl_deregmr(mhandle->cxi_region);
  }

  // The proxy threads are pinned to this node. Use the node of the CXI NIC
  // the endpoint opened, as the verbs path does; the socket interface is on
  // one node for all GPUs.
  static int uccl_get_numa_node(std::shared_ptr<CxiEndpoint> const& s,
                                int gpu_idx) {
    (void)gpu_idx;
    int const node = s ? s->nic_numa_node() : -1;
    if (node >= 0) return node;
    UCCL_LOG(WARN) << "CXI: NUMA node of the NIC is unknown; pinning proxy "
                      "threads to the node of the socket interface";
    return get_numa_node_from_iface();
  }

  static int uccl_send_notification(std::shared_ptr<CxiEndpoint> const& s,
                                    uint64_t peer_id,
                                    NotifyMsg const& notification) {
    if (!s) return -1;

    return s->send_notification(peer_id, notification);
  }
};
