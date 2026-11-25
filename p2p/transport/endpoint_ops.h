#pragma once
#include "engine.h"
#include "util/debug.h"
// Deliberately no primary implementation: a new transport must provide its
// adapter explicitly. Shared direct-endpoint operations can be inherited.
template <typename T>
struct EndpointOps;
template <typename T>
struct DirectEndpointOps {
  static ConnID uccl_connect(std::shared_ptr<T> const& s, int remote_gpuidx,
                             std::string const& remote_ip,
                             uint16_t remote_port) {
    return s->uccl_connect(remote_gpuidx, remote_ip, remote_port);
  }

  static uint16_t get_p2p_listen_port(std::shared_ptr<T> const& s) {
    return s->get_p2p_listen_port();
  }

  static int get_p2p_listen_fd(std::shared_ptr<T> const& s) {
    return s->get_p2p_listen_fd();
  }

  static ConnID uccl_accept(std::shared_ptr<T> const& s, std::string& remote_ip,
                            int* remote_gpuidx) {
    return s->uccl_accept(remote_ip, remote_gpuidx);
  }

  static bool uccl_poll_ureq_once(std::shared_ptr<T> const& s,
                                  UcclRequest* ureq) {
    if (ureq->type == ReqType::ReqWrite || ureq->type == ReqType::ReqRead) {
      s->send_routine();
      return s->check_send_complete_once(ureq->peer_id, ureq->engine_idx);
    }
    UCCL_LOG(ERROR) << "Invalid request type: " << ureq->type;
    return false;
  }

  static void uccl_drive_send(std::shared_ptr<T> const& s) {
    s->send_routine();
  }

  static void uccl_flush_send(std::shared_ptr<T> const& s) {
    s->flush_all_sends();
  }

  static void uccl_drive_recv(std::shared_ptr<T> const& s) {
    s->recv_routine();
  }

  static bool uccl_check_ureq_once(std::shared_ptr<T> const& s,
                                   UcclRequest* ureq) {
    if (ureq->type == ReqType::ReqWrite || ureq->type == ReqType::ReqRead) {
      return s->check_send_complete_once(ureq->peer_id, ureq->engine_idx);
    }
    UCCL_LOG(ERROR) << "Invalid request type: " << ureq->type;
    return false;
  }

  static bool initialize_rdma_ctx_for_gpu(std::shared_ptr<T> const& s,
                                          int dev) {
    return s->initialize_rdma_ctx_for_gpu(dev);
  }

  static void create_unified_p2p_socket(std::shared_ptr<T> const& s) {
    s->create_unified_p2p_socket();
  }
};
