#pragma once
#include "shared_host_packets.h"

namespace uccl_lite::multinode {
// Host-only progress after preparation. Every NUMA pair has one IB leader;
// its two rows are multicast to both remote NUMA groups. Uses the same MSCCL++
// CPU endpoint/ordered host notification mechanism as the recorded HostRdma.
// Calls on a leader are serialized. Storage must outlive this object.
class HostPacketTransport {
  static constexpr int kSlots = SharedHostPackets::slots;
  static constexpr int upload = 0, localAck = 2, remoteAck = 6;
  static constexpr int credit = 8, arrived = 10, retired = 12;
  SharedHostPackets& packets_;
  int rank_, group_;
  bool leader_;
  mscclpp::RegisteredMemory memory_;
  std::array<mscclpp::RegisteredMemory, 2> remote_;
  std::array<mscclpp::Connection, 2> connections_;
  std::array<std::array<uint64_t, kSlots>, 2> grants_{}, sent_{};
  std::array<uint64_t, kSlots> pending_{};
  static uint64_t load(uint64_t const* p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
  }
  static void store(uint64_t* p, uint64_t value) {
    __atomic_store_n(p, value, __ATOMIC_RELEASE);
  }
  size_t flagOffset(int slot, int field) const {
    return reinterpret_cast<char*>(packets_.flag(group_, slot, field)) -
           packets_.host(group_);
  }
  void leaderOnly() const {
    if (!leader_)
      throw std::logic_error("host packet operation requires NUMA leader");
  }

 public:
  HostPacketTransport(mscclpp::Communicator& comm, SharedHostPackets& packets,
                      int rank)
      : packets_(packets),
        rank_(rank),
        group_((rank % 4) / 2),
        leader_((rank % 2) == 0) {
    if (rank < 0 || rank >= 8) throw std::invalid_argument("packet rank");
    if (leader_) {
      auto transport = static_cast<mscclpp::Transport>(
          int(mscclpp::Transport::IB0) + group_);
      memory_ = comm.registerMemory(packets_.host(group_), packets_.bytes(),
                                    transport);
      std::array<std::shared_future<mscclpp::Connection>, 2> connection;
      std::array<std::shared_future<mscclpp::RegisteredMemory>, 2> memory;
      for (int peer = 0; peer < 2; ++peer) {
        int other = (1 - rank / 4) * 4 + 2 * peer;
        mscclpp::EndpointConfig endpoint(transport, mscclpp::DeviceType::CPU);
        endpoint.ib.mode = mscclpp::EndpointConfig::Ib::Mode::Host;
        connection[peer] = comm.connect(endpoint, other, 710);
        comm.sendMemory(memory_, other, 711);
        memory[peer] = comm.recvMemory(other, 711);
      }
      for (int peer = 0; peer < 2; ++peer) {
        connections_[peer] = connection[peer].get();
        remote_[peer] = memory[peer].get();
      }
    }
    comm.bootstrap()->barrier();
  }
  ~HostPacketTransport() {
    if (leader_) flush();
  }
  HostPacketTransport(HostPacketTransport const&) = delete;
  HostPacketTransport& operator=(HostPacketTransport const&) = delete;
  bool readersConsumed(int slot, uint64_t epoch) const {
    for (int r = 0; r < 4; ++r)
      if (load(packets_.flag(group_, slot, localAck + r)) < epoch) return false;
    for (int r = 0; r < 2; ++r)
      if (load(packets_.flag(group_, slot, remoteAck + r)) < epoch)
        return false;
    return true;
  }
  bool canReuse(int slot, uint64_t epoch) const {
    if (!epoch) throw std::invalid_argument("zero packet epoch");
    return readersConsumed(slot, epoch - 1) &&
           load(packets_.flag(group_, slot, retired)) >= epoch - 1;
  }
  void uploaded(int slot, uint64_t epoch) {
    if (!epoch) throw std::invalid_argument("zero packet epoch");
    store(packets_.flag(group_, slot, upload + rank_ % 2), epoch);
  }
  bool localReady(int slot, uint64_t epoch) const {
    for (int group = 0; group < 2; ++group)
      for (int r = 0; r < 2; ++r)
        if (load(packets_.flag(group, slot, upload + r)) != epoch) return false;
    return true;
  }
  void grant(int slot, uint64_t epoch) {
    leaderOnly();
    if (!epoch) throw std::invalid_argument("zero packet epoch");
    for (int r = 0; r < 2; ++r)
      if (load(packets_.flag(group_, slot, remoteAck + r)) < epoch - 1)
        throw std::logic_error("receiver still reading packet slot");
    for (int peer = 0; peer < 2; ++peer) {
      if (grants_[peer].at(slot) + 1 != epoch)
        throw std::logic_error("nonsequential receive credit");
      connections_[peer].updateAndSync(remote_[peer],
                                       flagOffset(slot, credit + group_),
                                       &grants_[peer][slot], epoch);
    }
  }
  bool canSend(int slot, uint64_t epoch) const {
    leaderOnly();
    for (int r = 0; r < 2; ++r)
      if (load(packets_.flag(group_, slot, upload + r)) != epoch ||
          load(packets_.flag(group_, slot, credit + r)) != epoch)
        return false;
    return true;
  }
  void send(int slot, size_t length, uint64_t epoch) {
    leaderOnly();
    if (!canSend(slot, epoch) || pending_.at(slot) + 1 != epoch)
      throw std::logic_error("packet send without uploads/credits");
    size_t destination = packets_.remoteOffset(slot, group_, length);
    for (int peer = 0; peer < 2; ++peer) {
      connections_[peer].write(remote_[peer], destination, memory_,
                               packets_.payloadOffset(slot), 2 * length);
      connections_[peer].updateAndSync(remote_[peer],
                                       flagOffset(slot, arrived + group_),
                                       &sent_[peer][slot], epoch);
    }
    pending_[slot] = epoch;
  }
  bool received(int slot, uint64_t epoch) const {
    for (int peer = 0; peer < 2; ++peer)
      if (load(packets_.flag(group_, slot, arrived + peer)) != epoch)
        return false;
    return true;
  }
  void consumed(int slot, uint64_t epoch) {
    for (int group = 0; group < 2; ++group)
      store(packets_.flag(group, slot, localAck + rank_ % 4), epoch);
    store(packets_.flag(group_, slot, remoteAck + rank_ % 2), epoch);
  }
  void flush() {
    leaderOnly();
    for (auto& connection : connections_) connection.flush();
    for (int slot = 0; slot < kSlots; ++slot)
      store(packets_.flag(group_, slot, retired), pending_[slot]);
  }
};
}  // namespace uccl_lite::multinode
