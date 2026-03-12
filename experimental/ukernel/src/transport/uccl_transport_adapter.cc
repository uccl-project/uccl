#include "uccl_transport_adapter.h"
#include "collective/rdma/transport.h"

namespace UKernel {
namespace Transport {

UcclTransportAdapter::UcclTransportAdapter(int local_rank, int world_size, UcclTransportConfig config)
    : local_rank_(local_rank), world_size_(world_size), config_(config) {
  endpoint_ = std::make_unique<::uccl::RDMAEndpoint>(config_.num_engines);
  endpoint_->initialize_resources(config_.num_engines * world_size);

  int dev_idx = endpoint_->get_best_dev_idx(local_rank_);
  endpoint_->initialize_engine_by_dev(dev_idx, true);
}

UcclTransportAdapter::~UcclTransportAdapter() {
  endpoint_.reset();
}

bool UcclTransportAdapter::connect_to_peer(int peer_rank, std::string remote_ip, uint16_t remote_port) {
  std::lock_guard<std::mutex> lk(mu_);

  int dev_idx = endpoint_->get_best_dev_idx(local_rank_);
  ::uccl::ConnID conn_id = endpoint_->uccl_connect(
      dev_idx, local_rank_, peer_rank, peer_rank, remote_ip, remote_port);

  PeerContext ctx;
  ctx.peer_rank = peer_rank;
  peer_contexts_[peer_rank] = std::move(ctx);
  return true;
}

bool UcclTransportAdapter::accept_from_peer(int peer_rank) {
  std::lock_guard<std::mutex> lk(mu_);

  int dev_idx = endpoint_->get_best_dev_idx(local_rank_);
  std::string remote_ip;
  int remote_dev = 0;
  int remote_gpuidx = 0;
  ::uccl::ConnID conn_id = endpoint_->uccl_accept(
      dev_idx, endpoint_->get_p2p_listen_fd(dev_idx), local_rank_,
      remote_ip, &remote_dev, &remote_gpuidx);

  PeerContext ctx;
  ctx.peer_rank = peer_rank;
  peer_contexts_[peer_rank] = std::move(ctx);
  return true;
}

uint64_t UcclTransportAdapter::register_memory(void* ptr, size_t len) {
  uint64_t mr_id = next_mr_id_.fetch_add(1);

  ::uccl::Mhandle* mhandle = nullptr;
  int dev_idx = endpoint_->get_best_dev_idx(local_rank_);
  endpoint_->uccl_regmr(dev_idx, ptr, len, 0, &mhandle);

  {
    std::lock_guard<std::mutex> lk(mu_);
    mr_id_to_mhandle_[mr_id] = mhandle;
  }

  return mr_id;
}

void UcclTransportAdapter::deregister_memory(uint64_t mr_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = mr_id_to_mhandle_.find(mr_id);
  if (it != mr_id_to_mhandle_.end()) {
    endpoint_->uccl_deregmr(it->second);
    mr_id_to_mhandle_.erase(it);
  }
}

int UcclTransportAdapter::send_async(int peer_rank, void* local_ptr, size_t len,
                                     uint64_t local_mr_id, uint64_t remote_mr_id) {
  auto it = peer_contexts_.find(peer_rank);
  if (it == peer_contexts_.end()) return -1;

  ::uccl::Mhandle* local_mh = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto mh_it = mr_id_to_mhandle_.find(local_mr_id);
    if (mh_it != mr_id_to_mhandle_.end()) {
      local_mh = mh_it->second;
    }
  }

  if (!local_mh) return -1;

  ::uccl::ucclRequest* ureq = new ::uccl::ucclRequest();
  int ret = endpoint_->uccl_send_async(it->second.flow, local_mh, local_ptr, len, ureq);

  return ret == 0 ? 0 : -1;
}

int UcclTransportAdapter::recv_async(int peer_rank, void* local_ptr, size_t len, uint64_t local_mr_id) {
  auto it = peer_contexts_.find(peer_rank);
  if (it == peer_contexts_.end()) return -1;

  ::uccl::Mhandle* local_mh = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto mh_it = mr_id_to_mhandle_.find(local_mr_id);
    if (mh_it != mr_id_to_mhandle_.end()) {
      local_mh = mh_it->second;
    }
  }

  if (!local_mh) return -1;

  ::uccl::ucclRequest* ureq = new ::uccl::ucclRequest();
  ::uccl::Mhandle* mh_array[1] = {local_mh};
  void* data_array[1] = {local_ptr};
  int size_array[1] = {static_cast<int>(len)};

  int ret = endpoint_->uccl_recv_async(it->second.flow, mh_array, data_array, size_array, 1, ureq);

  return ret == 0 ? 0 : -1;
}

bool UcclTransportAdapter::poll_completion(int* out_peer_rank, uint64_t* out_mr_id) {
  return false;
}

bool UcclTransportAdapter::wait_completion(int peer_rank, uint64_t mr_id) {
  return true;
}

}  // namespace Transport
}  // namespace UKernel
