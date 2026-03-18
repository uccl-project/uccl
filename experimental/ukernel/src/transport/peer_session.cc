#include "peer_session.h"

namespace UKernel {
namespace Transport {

PeerSession::PeerSession(int peer_rank) : peer_rank_(peer_rank) {}

bool PeerSession::has_meta() const {
  std::lock_guard<std::mutex> lk(mu_);
  return has_meta_;
}

CommunicatorMeta const& PeerSession::meta() const {
  std::lock_guard<std::mutex> lk(mu_);
  return meta_;
}

void PeerSession::set_meta(CommunicatorMeta const& meta) {
  std::lock_guard<std::mutex> lk(mu_);
  meta_ = meta;
  has_meta_ = true;
}

PeerTransportKind PeerSession::transport_kind() const {
  std::lock_guard<std::mutex> lk(mu_);
  return kind_;
}

void PeerSession::set_transport_kind(PeerTransportKind kind) {
  std::lock_guard<std::mutex> lk(mu_);
  kind_ = kind;
}

bool PeerSession::send_ready() const {
  std::lock_guard<std::mutex> lk(mu_);
  return send_ready_;
}

bool PeerSession::recv_ready() const {
  std::lock_guard<std::mutex> lk(mu_);
  return recv_ready_;
}

void PeerSession::set_send_ready(bool ready) {
  std::lock_guard<std::mutex> lk(mu_);
  send_ready_ = ready;
}

void PeerSession::set_recv_ready(bool ready) {
  std::lock_guard<std::mutex> lk(mu_);
  recv_ready_ = ready;
}

void PeerSession::reset() {
  std::lock_guard<std::mutex> lk(mu_);
  has_meta_ = false;
  kind_ = PeerTransportKind::Ipc;
  send_ready_ = false;
  recv_ready_ = false;
}

PeerSessionManager::PeerSessionManager(int world_size)
    : world_size_(world_size) {
  peers_.reserve(world_size);
  for (int i = 0; i < world_size; ++i) {
    peers_.push_back(std::make_unique<PeerSession>(i));
  }
}

PeerSession* PeerSessionManager::get(int rank) {
  if (rank < 0 || rank >= world_size_) return nullptr;
  return peers_[rank].get();
}

PeerSession const* PeerSessionManager::get(int rank) const {
  if (rank < 0 || rank >= world_size_) return nullptr;
  return peers_[rank].get();
}

bool PeerSessionManager::all_have_meta() const {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto const& p : peers_) {
    if (!p->has_meta()) return false;
  }
  return true;
}

bool PeerSessionManager::all_ready() const {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto const& p : peers_) {
    if (!p->send_ready() || !p->recv_ready()) return false;
  }
  return true;
}

bool PeerSessionManager::has_peer_send_path(int rank) const {
  if (rank < 0 || rank >= world_size_) return false;
  return peers_[rank]->send_ready();
}

bool PeerSessionManager::has_peer_recv_path(int rank) const {
  if (rank < 0 || rank >= world_size_) return false;
  return peers_[rank]->recv_ready();
}

}  // namespace Transport
}  // namespace UKernel
