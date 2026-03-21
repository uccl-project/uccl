#include "uccl_transport_adapter.h"
#include "collective/rdma/transport.h"
#include <chrono>
#include <iostream>
#include <thread>

namespace UKernel {
namespace Transport {

namespace {
constexpr auto kUcclRetrySleep = std::chrono::microseconds(50);
constexpr auto kUcclAsyncRetryTimeout = std::chrono::seconds(30);
}  // namespace

UcclTransportAdapter::UcclTransportAdapter(int local_gpu_idx, int world_size,
                                           UcclTransportConfig config)
    : local_gpu_idx_(local_gpu_idx), world_size_(world_size), config_(config) {
  int num_engines = static_cast<int>(::ucclParamNUM_ENGINES());
  if (config_.num_engines > 0 && config_.num_engines != num_engines) {
    std::cout << "[WARN] UCCL engine count mismatch: requested "
              << config_.num_engines << ", runtime " << num_engines
              << ". Using runtime value." << std::endl;
  }
  config_.num_engines = num_engines;

  endpoint_ = std::make_unique<::uccl::RDMAEndpoint>(config_.num_engines);
  endpoint_->initialize_resources(config_.num_engines * world_size);

  int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
  endpoint_->initialize_engine_by_dev(dev_idx, true);
}

UcclTransportAdapter::~UcclTransportAdapter() { endpoint_.reset(); }

uint16_t UcclTransportAdapter::get_p2p_listen_port(int dev_idx) const {
  return endpoint_->get_p2p_listen_port(dev_idx);
}

std::string UcclTransportAdapter::get_p2p_listen_ip(int dev_idx) const {
  return endpoint_->get_p2p_listen_ip(dev_idx);
}

int UcclTransportAdapter::get_best_dev_idx(int gpu_idx) const {
  return endpoint_->get_best_dev_idx(gpu_idx);
}

bool UcclTransportAdapter::has_send_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second.send_flow != nullptr;
}

bool UcclTransportAdapter::has_recv_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second.recv_flow != nullptr;
}

bool UcclTransportAdapter::connect_to_peer(int peer_rank, std::string remote_ip,
                                           uint16_t remote_port,
                                           int local_dev_idx, int local_gpu_idx,
                                           int remote_dev_idx,
                                           int remote_gpu_idx) {
  if (has_send_peer(peer_rank)) return true;

  ::uccl::ConnID conn_id =
      endpoint_->uccl_connect(local_dev_idx, local_gpu_idx, remote_dev_idx,
                              remote_gpu_idx, remote_ip, remote_port);

  std::lock_guard<std::mutex> lk(mu_);
  auto& ctx = peer_contexts_[peer_rank];
  ctx.peer_rank = peer_rank;
  ctx.send_flow = static_cast<::uccl::UcclFlow*>(conn_id.context);
  ctx.remote_ip = std::move(remote_ip);
  ctx.remote_dev_idx = remote_dev_idx;
  ctx.remote_gpu_idx = remote_gpu_idx;
  return true;
}

bool UcclTransportAdapter::accept_from_peer(
    int peer_rank, std::string const& expected_remote_ip,
    int expected_remote_dev_idx, int expected_remote_gpu_idx,
    AcceptedPeer* accepted_peer) {
  if (has_recv_peer(peer_rank)) return true;

  int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
  std::string remote_ip;
  int remote_dev = 0;
  int remote_gpuidx = 0;
  ::uccl::ConnID conn_id = endpoint_->uccl_accept(
      dev_idx, endpoint_->get_p2p_listen_fd(dev_idx), local_gpu_idx_, remote_ip,
      &remote_dev, &remote_gpuidx);

  if ((!expected_remote_ip.empty() && remote_ip != expected_remote_ip) ||
      (expected_remote_dev_idx >= 0 && remote_dev != expected_remote_dev_idx) ||
      (expected_remote_gpu_idx >= 0 &&
       remote_gpuidx != expected_remote_gpu_idx)) {
    std::cerr << "[ERROR] UCCL accept peer mismatch for rank " << peer_rank
              << ": expected ip/dev/gpu=" << expected_remote_ip << "/"
              << expected_remote_dev_idx << "/" << expected_remote_gpu_idx
              << ", got " << remote_ip << "/" << remote_dev << "/"
              << remote_gpuidx << std::endl;
    endpoint_->discard_conn(conn_id);
    return false;
  }

  if (accepted_peer != nullptr) {
    accepted_peer->remote_ip = remote_ip;
    accepted_peer->remote_dev_idx = remote_dev;
    accepted_peer->remote_gpu_idx = remote_gpuidx;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto& ctx = peer_contexts_[peer_rank];
  ctx.peer_rank = peer_rank;
  ctx.recv_flow = static_cast<::uccl::UcclFlow*>(conn_id.context);
  ctx.remote_ip = std::move(remote_ip);
  ctx.remote_dev_idx = remote_dev;
  ctx.remote_gpu_idx = remote_gpuidx;
  return true;
}

bool UcclTransportAdapter::register_memory(uint64_t mr_id, void* ptr,
                                           size_t len) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (mr_id_to_mhandle_.find(mr_id) != mr_id_to_mhandle_.end()) {
      return true;
    }
  }

  ::uccl::Mhandle* mhandle = nullptr;
  int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
  if (endpoint_->uccl_regmr(dev_idx, ptr, len, 0, &mhandle) != 0 || !mhandle) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    mr_id_to_mhandle_[mr_id] = mhandle;
  }

  return true;
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
                                      uint64_t local_mr_id,
                                      uint64_t remote_mr_id,
                                      uint64_t request_id) {
  (void)remote_mr_id;
  ::uccl::UcclFlow* flow = nullptr;
  ::uccl::Mhandle* local_mh = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(peer_rank);
    if (peer_it == peer_contexts_.end()) return -1;
    flow = peer_it->second.send_flow;

    auto mh_it = mr_id_to_mhandle_.find(local_mr_id);
    if (mh_it != mr_id_to_mhandle_.end()) {
      local_mh = mh_it->second;
    }
  }

  if (!flow || !local_mh) return -1;

  auto ureq = std::make_unique<::uccl::ucclRequest>();
  auto deadline = std::chrono::steady_clock::now() + kUcclAsyncRetryTimeout;
  int ret = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    ret =
        endpoint_->uccl_send_async(flow, local_mh, local_ptr, len, ureq.get());
    if (ret == 0) break;
    std::this_thread::sleep_for(kUcclRetrySleep);
  }
  if (ret != 0) return -1;

  {
    std::lock_guard<std::mutex> lk(mu_);
    pending_requests_[request_id] = std::move(ureq);
  }

  return 0;
}

int UcclTransportAdapter::recv_async(int peer_rank, void* local_ptr, size_t len,
                                      uint64_t local_mr_id,
                                      uint64_t request_id) {
  ::uccl::UcclFlow* flow = nullptr;
  ::uccl::Mhandle* local_mh = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(peer_rank);
    if (peer_it == peer_contexts_.end()) return -1;
    flow = peer_it->second.recv_flow;

    auto mh_it = mr_id_to_mhandle_.find(local_mr_id);
    if (mh_it != mr_id_to_mhandle_.end()) {
      local_mh = mh_it->second;
    }
  }

  if (!flow || !local_mh) return -1;

  auto ureq = std::make_unique<::uccl::ucclRequest>();
  ::uccl::Mhandle* mh_array[1] = {local_mh};
  void* data_array[1] = {local_ptr};
  int size_array[1] = {static_cast<int>(len)};

  auto deadline = std::chrono::steady_clock::now() + kUcclAsyncRetryTimeout;
  int ret = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    ret = endpoint_->uccl_recv_async(flow, mh_array, data_array, size_array, 1,
                                     ureq.get());
    if (ret == 0) break;
    std::this_thread::sleep_for(kUcclRetrySleep);
  }
  if (ret != 0) return -1;

  {
    std::lock_guard<std::mutex> lk(mu_);
    pending_requests_[request_id] = std::move(ureq);
  }

  return 0;
}

bool UcclTransportAdapter::poll_completion(uint64_t request_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = pending_requests_.find(request_id);
  if (it == pending_requests_.end()) return true;
  if (!endpoint_->uccl_poll_ureq_once(it->second.get())) return false;
  pending_requests_.erase(it);
  return true;
}

bool UcclTransportAdapter::wait_completion(uint64_t request_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = pending_requests_.find(request_id);
  if (it == pending_requests_.end()) return false;
  endpoint_->uccl_poll_ureq(it->second.get());
  pending_requests_.erase(it);
  return true;
}

}  // namespace Transport
}  // namespace UKernel
