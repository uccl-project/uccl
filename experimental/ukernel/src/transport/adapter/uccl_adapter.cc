#include "uccl_adapter.h"
#include "../util/utils.h"
#include "collective/rdma/transport.h"
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

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

constexpr auto kUcclAsyncRetryTimeout = std::chrono::seconds(30);
constexpr uint32_t kUcclRetryYieldThreshold = 4;
constexpr uint32_t kUcclRetrySleepThreshold = 256;
constexpr auto kUcclRetryBackoff = std::chrono::microseconds(2);

inline void backoff_retry(uint32_t retries) {
  if (retries < kUcclRetryYieldThreshold) return;
  if (retries < kUcclRetrySleepThreshold) {
    std::this_thread::yield();
    return;
  }
  std::this_thread::sleep_for(kUcclRetryBackoff);
}
}  // namespace

UcclTransportAdapter::UcclTransportAdapter(int gpu_id, int world_size,
                                           UcclTransportConfig config)
    : gpu_id_(gpu_id) {
  send_ring_ = create_ring(sizeof(RingElem), 1024);
  recv_ring_ = create_ring(sizeof(RingElem), 1024);
  if (!send_ring_ || !recv_ring_)
    throw std::runtime_error("UCCL ring alloc failed");

  // Try to create real UCCL endpoint. If UCCL is not available, workers
  // will fall back to publish_completion-only behaviour.
  int num_engines = static_cast<int>(::ucclParamNUM_ENGINES());
  if (num_engines <= 0) {
    std::cerr << "[WARN] UCCL NUM_ENGINES=" << num_engines
              << ", UCCL adapter operating in fallback mode" << std::endl;
  } else {
    if (config.num_engines > 0 && config.num_engines != num_engines) {
      std::cout << "[WARN] UCCL engine count mismatch: requested "
                << config.num_engines << ", runtime " << num_engines
                << ". Using runtime value." << std::endl;
    }
    endpoint_ = std::make_unique<::uccl::RDMAEndpoint>(num_engines);
    endpoint_->initialize_resources(num_engines * world_size);

    local_dev_idx_ = endpoint_->get_best_dev_idx(gpu_id_);
    if (local_dev_idx_ < 0) {
      std::cerr << "[ERROR] UCCL get_best_dev_idx failed for gpu " << gpu_id_
                << std::endl;
      endpoint_.reset();
    } else {
      endpoint_->initialize_engine_by_dev(local_dev_idx_, true);
    }
  }

  send_th_ = std::thread([this] { send_worker(); });
  recv_th_ = std::thread([this] { recv_worker(); });
}

UcclTransportAdapter::~UcclTransportAdapter() {
  stop_.store(true);
  send_th_.join();
  recv_th_.join();
  free(send_ring_);
  free(recv_ring_);

  // Deregister all control_mhandle memory.
  if (endpoint_) {
    for (auto& kv : peers_) {
      if (kv.second.control_mhandle) {
        endpoint_->uccl_deregmr(kv.second.control_mhandle);
        kv.second.control_mhandle = nullptr;
      }
    }
  }
}

uint16_t UcclTransportAdapter::get_p2p_listen_port(int dev_idx) const {
  if (!endpoint_ || dev_idx < 0) return 0;
  return endpoint_->get_p2p_listen_port(dev_idx);
}

std::string UcclTransportAdapter::get_p2p_listen_ip(int dev_idx) const {
  if (!endpoint_ || dev_idx < 0) return {};
  return endpoint_->get_p2p_listen_ip(dev_idx);
}

int UcclTransportAdapter::get_best_dev_idx(int gpu_idx) const {
  if (!endpoint_) return -1;
  return endpoint_->get_best_dev_idx(gpu_idx);
}

bool UcclTransportAdapter::is_memory_registered(uint32_t id) const {
  std::lock_guard<std::mutex> lk(mu_);
  return buffer_id_to_mhandle_.count(id) > 0;
}

bool UcclTransportAdapter::register_memory(uint32_t id, void* ptr, size_t len) {
  if (!endpoint_ || ptr == nullptr || len == 0) return false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (buffer_id_to_mhandle_.count(id)) return true;
  }

  int dev_idx = local_dev_idx_;
  if (dev_idx < 0) return false;

  ::uccl::Mhandle* mhandle = nullptr;
  if (endpoint_->uccl_regmr(dev_idx, ptr, len, 0, &mhandle) != 0 || !mhandle)
    return false;

  {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_id_to_mhandle_[id] = mhandle;
  }
  return true;
}

void UcclTransportAdapter::deregister_memory(uint32_t id) {
  if (!endpoint_) return;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = buffer_id_to_mhandle_.find(id);
  if (it != buffer_id_to_mhandle_.end()) {
    endpoint_->uccl_deregmr(it->second);
    buffer_id_to_mhandle_.erase(it);
  }
}

bool UcclTransportAdapter::connect_to_peer(int peer_rank,
                                           std::string const& remote_ip,
                                           uint16_t remote_port,
                                           int local_dev_idx, int local_gpu_idx,
                                           int remote_dev_idx,
                                           int remote_gpu_idx) {
  if (has_put_path(peer_rank)) return true;
  if (!endpoint_) return false;
  if (local_dev_idx < 0 || remote_dev_idx < 0 || remote_port == 0) return false;

  ::uccl::ConnID conn_id =
      endpoint_->uccl_connect(local_dev_idx, local_gpu_idx, remote_dev_idx,
                              remote_gpu_idx, remote_ip, remote_port);
  if (conn_id.context == nullptr) {
    std::cerr << "[ERROR] UCCL connect returned null context for peer "
              << peer_rank << std::endl;
    return false;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto& ctx = peers_[peer_rank];
  ctx.send_flow = static_cast<::uccl::UcclFlow*>(conn_id.context);

  // Register control memory for signal tag exchange.
  if (!ctx.control_mhandle) {
    int dev_idx = local_dev_idx_;
    if (dev_idx >= 0) {
      ::uccl::Mhandle* ctrl_mh = nullptr;
      endpoint_->uccl_regmr(dev_idx, &ctx.control_tag, sizeof(ctx.control_tag),
                            0, &ctrl_mh);
      ctx.control_mhandle = ctrl_mh;
    }
  }

  return true;
}

bool UcclTransportAdapter::accept_from_peer(
    int peer_rank, std::string const& expected_remote_ip,
    int expected_remote_dev_idx, int expected_remote_gpu_idx,
    uint16_t expected_remote_port) {
  if (has_wait_path(peer_rank)) return true;
  (void)expected_remote_port;
  if (!endpoint_) return false;

  int dev_idx = local_dev_idx_;
  if (dev_idx < 0) {
    std::cerr << "[ERROR] UCCL accept: invalid local dev for gpu " << gpu_id_
              << std::endl;
    return false;
  }
  int listen_fd = endpoint_->get_p2p_listen_fd(dev_idx);
  if (listen_fd < 0) {
    std::cerr << "[ERROR] UCCL accept: invalid listen fd for dev " << dev_idx
              << std::endl;
    return false;
  }

  std::string remote_ip;
  int remote_dev = 0;
  int remote_gpuidx = 0;
  ::uccl::ConnID conn_id = endpoint_->uccl_accept(
      dev_idx, listen_fd, gpu_id_, remote_ip, &remote_dev, &remote_gpuidx);
  if (conn_id.context == nullptr) {
    std::cerr << "[ERROR] UCCL accept returned null context for peer "
              << peer_rank << std::endl;
    return false;
  }

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

  std::lock_guard<std::mutex> lk(mu_);
  auto& ctx = peers_[peer_rank];
  ctx.recv_flow = static_cast<::uccl::UcclFlow*>(conn_id.context);

  // Register control memory for signal tag exchange.
  if (!ctx.control_mhandle) {
    if (dev_idx >= 0) {
      ::uccl::Mhandle* ctrl_mh = nullptr;
      endpoint_->uccl_regmr(dev_idx, &ctx.control_tag, sizeof(ctx.control_tag),
                            0, &ctrl_mh);
      ctx.control_mhandle = ctrl_mh;
    }
  }

  return true;
}

bool UcclTransportAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_put_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Connect) return false;
  if (!std::holds_alternative<UcclPeerConnectSpec>(spec.detail)) return false;
  auto const& u = std::get<UcclPeerConnectSpec>(spec.detail);
  return connect_to_peer(spec.peer_rank, u.remote_ip, u.remote_port,
                         u.local_dev_idx, u.local_gpu_idx, u.remote_dev_idx,
                         u.remote_gpu_idx);
}

bool UcclTransportAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_wait_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Accept) return false;
  if (!std::holds_alternative<UcclPeerConnectSpec>(spec.detail)) return false;
  auto const& u = std::get<UcclPeerConnectSpec>(spec.detail);
  return accept_from_peer(spec.peer_rank, u.remote_ip, u.remote_dev_idx,
                          u.remote_gpu_idx, u.remote_port);
}

bool UcclTransportAdapter::has_put_path(int peer) const {
  std::lock_guard<std::mutex> lk(mu_);
  return peers_.count(peer) && peers_.at(peer).send_flow;
}

bool UcclTransportAdapter::has_wait_path(int peer) const {
  std::lock_guard<std::mutex> lk(mu_);
  return peers_.count(peer) && peers_.at(peer).recv_flow;
}

unsigned UcclTransportAdapter::send_put_async(int peer, void* local_ptr,
                                              uint32_t local_buf, void*,
                                              uint32_t, size_t bytes,
                                              unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  RingElem e{comm_rid, peer, Kind::DataPut, local_ptr, bytes, local_buf, 0};
  return enqueue_elem(send_ring_, e, stop_) ? 1 : 0;
}

unsigned UcclTransportAdapter::send_signal_async(int peer, uint64_t tag,
                                                 unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  RingElem e{comm_rid, peer, Kind::Signal, nullptr, 0, 0, tag};
  return enqueue_elem(send_ring_, e, stop_) ? 1 : 0;
}

unsigned UcclTransportAdapter::wait_signal_async(
    int peer, uint64_t tag, std::optional<WaitTarget> target,
    unsigned comm_rid) {
  if (!has_wait_path(peer)) return 0;
  if (target.has_value()) {
    RingElem e{comm_rid,
               peer,
               Kind::DataWait,
               target->local_ptr,
               target->len,
               target->local_buffer_id,
               0};
    return enqueue_elem(recv_ring_, e, stop_) ? 1 : 0;
  } else {
    RingElem e{comm_rid, peer, Kind::SignalWait, nullptr, 0, 0, tag};
    return enqueue_elem(recv_ring_, e, stop_) ? 1 : 0;
  }
}

void UcclTransportAdapter::send_worker() {
  RingElem e;
  ::uccl::ucclRequest ureq;

  struct Pending {
    unsigned comm_rid;
    ::uccl::ucclRequest request;
  };
  std::vector<Pending> pending;

  while (!stop_.load(std::memory_order_acquire)) {
    bool did_work = false;

    // ── Step 1: try to dequeue a new task (non-blocking) ──
    if (jring_sc_dequeue_bulk(send_ring_, &e, 1, nullptr) == 1) {
      did_work = true;

      // Fallback: no UCCL endpoint — just publish completion immediately.
      if (!endpoint_) {
        publish_completion(e.comm_rid, false);
      } else {
        bool failed = false;
        ::uccl::UcclFlow* flow = nullptr;
        ::uccl::Mhandle* mh = nullptr;
        void* data_ptr = nullptr;
        size_t data_len = 0;

        // Resolve flow and memory handle under the lock.
        {
          std::lock_guard<std::mutex> lk(mu_);
          auto it = peers_.find(e.peer);
          if (it == peers_.end() || !it->second.send_flow) {
            failed = true;
          } else {
            flow = it->second.send_flow;
            switch (e.kind) {
              case Kind::DataPut: {
                auto mh_it = buffer_id_to_mhandle_.find(e.buffer_id);
                if (mh_it != buffer_id_to_mhandle_.end()) {
                  mh = mh_it->second;
                  data_ptr = e.ptr;
                  data_len = e.len;
                } else {
                  failed = true;
                }
                break;
              }
              case Kind::Signal: {
                if (it->second.control_mhandle) {
                  it->second.control_tag = e.tag;
                  mh = it->second.control_mhandle;
                  data_ptr = &it->second.control_tag;
                  data_len = sizeof(uint64_t);
                } else {
                  failed = true;
                }
                break;
              }
              default:
                failed = true;
                break;
            }
          }
        }

        if (!failed && flow && mh && data_ptr && data_len > 0) {
          int ret = -1;
          auto deadline =
              std::chrono::steady_clock::now() + kUcclAsyncRetryTimeout;
          uint32_t retries = 0;
          while (std::chrono::steady_clock::now() < deadline) {
            std::memset(&ureq, 0, sizeof(ureq));
            ret =
                endpoint_->uccl_send_async(flow, mh, data_ptr, data_len, &ureq);
            if (ret == 0) break;

            // Every 16 retries, poll pending to help UCCL engine drain
            // completions
            if ((retries & 15) == 0) {
              for (auto it = pending.begin(); it != pending.end();) {
                if (endpoint_->uccl_poll_ureq_once(&it->request)) {
                  did_work = true;
                  publish_completion(it->comm_rid, false);
                  it = pending.erase(it);
                } else {
                  ++it;
                }
              }
            }

            backoff_retry(retries++);
          }
          if (ret == 0) {
            pending.push_back({e.comm_rid, ureq});
          } else {
            failed = true;
          }
        }

        if (failed) {
          publish_completion(e.comm_rid, true);
        }
      }
    }

    // ── Step 2: poll all pending sends (non-blocking) ──
    for (auto it = pending.begin(); it != pending.end();) {
      if (endpoint_ && endpoint_->uccl_poll_ureq_once(&it->request)) {
        did_work = true;
        publish_completion(it->comm_rid, false);
        it = pending.erase(it);
      } else {
        ++it;
      }
    }

    // ── Step 3: yield if idle ──
    if (!did_work) {
      std::this_thread::yield();
    }
  }

  // Drain on shutdown: publish failure for any remaining pending.
  for (auto& p : pending) {
    publish_completion(p.comm_rid, true);
  }
  RingElem drain;
  while (jring_mc_dequeue_bulk(send_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

void UcclTransportAdapter::recv_worker() {
  RingElem e;
  ::uccl::ucclRequest ureq;

  struct Pending {
    unsigned comm_rid;
    ::uccl::ucclRequest request;
  };
  std::vector<Pending> pending;

  while (!stop_.load(std::memory_order_acquire)) {
    bool did_work = false;

    // ── Step 1: try to dequeue a new task (non-blocking) ──
    if (jring_sc_dequeue_bulk(recv_ring_, &e, 1, nullptr) == 1) {
      did_work = true;

      // Fallback: no UCCL endpoint — just publish completion immediately.
      if (!endpoint_) {
        publish_completion(e.comm_rid, false);
      } else {
        bool failed = false;
        ::uccl::UcclFlow* flow = nullptr;
        ::uccl::Mhandle* mh = nullptr;
        void* data_ptr = nullptr;
        size_t data_len = 0;

        // Resolve flow and memory handle under the lock.
        {
          std::lock_guard<std::mutex> lk(mu_);
          auto it = peers_.find(e.peer);
          if (it == peers_.end() || !it->second.recv_flow) {
            failed = true;
          } else {
            flow = it->second.recv_flow;
            switch (e.kind) {
              case Kind::DataWait: {
                auto mh_it = buffer_id_to_mhandle_.find(e.buffer_id);
                if (mh_it != buffer_id_to_mhandle_.end()) {
                  mh = mh_it->second;
                  data_ptr = e.ptr;
                  data_len = e.len;
                } else {
                  failed = true;
                }
                break;
              }
              case Kind::SignalWait: {
                if (it->second.control_mhandle) {
                  mh = it->second.control_mhandle;
                  data_ptr = &it->second.control_tag;
                  data_len = sizeof(uint64_t);
                } else {
                  failed = true;
                }
                break;
              }
              default:
                failed = true;
                break;
            }
          }
        }

        if (!failed && flow && mh && data_ptr && data_len > 0) {
          std::memset(&ureq, 0, sizeof(ureq));
          ::uccl::Mhandle* mh_array[1] = {mh};
          void* data_array[1] = {data_ptr};
          int size_array[1] = {static_cast<int>(data_len)};
          int ret = -1;
          auto deadline =
              std::chrono::steady_clock::now() + kUcclAsyncRetryTimeout;
          uint32_t retries = 0;
          while (std::chrono::steady_clock::now() < deadline) {
            std::memset(&ureq, 0, sizeof(ureq));
            ret = endpoint_->uccl_recv_async(flow, mh_array, data_array,
                                             size_array, 1, &ureq);
            if (ret == 0) break;

            // Every 16 retries, poll pending to help UCCL engine drain
            // completions
            if ((retries & 15) == 0) {
              for (auto it = pending.begin(); it != pending.end();) {
                if (endpoint_->uccl_poll_ureq_once(&it->request)) {
                  did_work = true;
                  publish_completion(it->comm_rid, false);
                  it = pending.erase(it);
                } else {
                  ++it;
                }
              }
            }

            backoff_retry(retries++);
          }
          if (ret == 0) {
            pending.push_back({e.comm_rid, ureq});
          } else {
            failed = true;
          }
        }

        if (failed) {
          publish_completion(e.comm_rid, true);
        }
      }
    }

    // ── Step 2: poll all pending recvs (non-blocking) ──
    for (auto it = pending.begin(); it != pending.end();) {
      if (endpoint_ && endpoint_->uccl_poll_ureq_once(&it->request)) {
        did_work = true;
        publish_completion(it->comm_rid, false);
        it = pending.erase(it);
      } else {
        ++it;
      }
    }

    // ── Step 3: yield if idle ──
    if (!did_work) {
      std::this_thread::yield();
    }
  }

  // Drain on shutdown: publish failure for any remaining pending.
  for (auto& p : pending) {
    publish_completion(p.comm_rid, true);
  }
  RingElem drain;
  while (jring_mc_dequeue_bulk(recv_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

}  // namespace Transport
}  // namespace UKernel
