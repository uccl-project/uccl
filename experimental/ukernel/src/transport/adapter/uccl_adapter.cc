#include "uccl_adapter.h"
#include "../util/utils.h"
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace Transport {

namespace {
template <typename T>
bool enqueue_elem(jring_t* ring, T const& elem, std::atomic<bool> const& stop) {
  while (!stop.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(ring, &elem, 1, nullptr) != 1)
    std::this_thread::yield();
  return !stop.load(std::memory_order_acquire);
}
}

UcclTransportAdapter::UcclTransportAdapter(int gpu_id, int world_size,
                                           UcclTransportConfig config)
    : gpu_id_(gpu_id) {
  (void)world_size; (void)config;
  send_ring_ = create_ring(sizeof(RingElem), 1024);
  recv_ring_ = create_ring(sizeof(RingElem), 1024);
  if (!send_ring_ || !recv_ring_) throw std::runtime_error("UCCL ring alloc failed");
  send_th_ = std::thread([this] { send_worker(); });
  recv_th_ = std::thread([this] { recv_worker(); });
}

UcclTransportAdapter::~UcclTransportAdapter() {
  stop_.store(true);
  send_th_.join(); recv_th_.join();
  free(send_ring_); free(recv_ring_);
}

bool UcclTransportAdapter::is_memory_registered(uint32_t id) const {
  std::lock_guard<std::mutex> lk(mu_);
  return buffer_id_to_mhandle_.count(id) > 0;
}
bool UcclTransportAdapter::register_memory(uint32_t id, void* ptr, size_t len) {
  (void)id; (void)ptr; (void)len; return true;
}
void UcclTransportAdapter::deregister_memory(uint32_t id) {
  std::lock_guard<std::mutex> lk(mu_);
  buffer_id_to_mhandle_.erase(id);
}

bool UcclTransportAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  std::lock_guard<std::mutex> lk(mu_);
  peers_[spec.peer_rank].send_flow = reinterpret_cast<::uccl::UcclFlow*>(1);
  return true;
}
bool UcclTransportAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  std::lock_guard<std::mutex> lk(mu_);
  peers_[spec.peer_rank].recv_flow = reinterpret_cast<::uccl::UcclFlow*>(1);
  return true;
}
bool UcclTransportAdapter::has_put_path(int peer) const {
  std::lock_guard<std::mutex> lk(mu_);
  return peers_.count(peer) && peers_.at(peer).send_flow;
}
bool UcclTransportAdapter::has_wait_path(int peer) const {
  std::lock_guard<std::mutex> lk(mu_);
  return peers_.count(peer) && peers_.at(peer).recv_flow;
}
uint16_t UcclTransportAdapter::get_p2p_listen_port(int d) const { (void)d; return 0; }
std::string UcclTransportAdapter::get_p2p_listen_ip(int d) const { (void)d; return ""; }
int UcclTransportAdapter::get_best_dev_idx(int g) const { (void)g; return 0; }

unsigned UcclTransportAdapter::put_async(int peer, void* local_ptr, uint32_t,
                                         void*, uint32_t, size_t bytes,
                                         unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  RingElem e{comm_rid, peer, Kind::DataPut, local_ptr, bytes};
  return enqueue_elem(send_ring_, e, stop_) ? 1 : 0;
}
unsigned UcclTransportAdapter::signal_async(int peer, uint64_t tag, unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  RingElem e{comm_rid, peer, Kind::Signal, (void*)(uintptr_t)tag, 0};
  return enqueue_elem(send_ring_, e, stop_) ? 1 : 0;
}
unsigned UcclTransportAdapter::wait_async(int peer, uint64_t tag,
                                          std::optional<WaitTarget> target,
                                          unsigned comm_rid) {
  if (!has_wait_path(peer)) return 0;
  RingElem e{comm_rid, peer, Kind::DataWait,
             target ? target->local_ptr : nullptr,
             target ? target->len : 0};
  (void)tag;
  return enqueue_elem(recv_ring_, e, stop_) ? 1 : 0;
}

void UcclTransportAdapter::send_worker() {
  RingElem e;
  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(send_ring_, &e, 1, nullptr) != 1)
      { std::this_thread::yield(); continue; }
    publish_completion(e.comm_rid, false);
  }
}
void UcclTransportAdapter::recv_worker() {
  RingElem e;
  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(recv_ring_, &e, 1, nullptr) != 1)
      { std::this_thread::yield(); continue; }
    publish_completion(e.comm_rid, false);
  }
}

}  // namespace Transport
}  // namespace UKernel
