#pragma once

#include "dpdk/dpdk.h"
#include "dpdk/packet.h"
#include "dpdk/packet_pool.h"
#include "dpdk/pmd.h"
#include <bitset>
#include <cstdint>
#include <sstream>
#include <string>

namespace uccl {
class PacketBuf {
 private:
  juggler::dpdk::Packet* packet_;
  PacketBuf* next_;
#define UCCL_MSGBUF_FLAGS_SYN (1 << 0)
#define UCCL_MSGBUF_FLAGS_FIN (1 << 1)
#define UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE (1 << 2)
  uint8_t msg_flags_;

  PacketBuf(juggler::dpdk::Packet* pkt)
      : packet_(pkt), next_(nullptr), msg_flags_(0) {}

 public:
  static PacketBuf* Create(juggler::dpdk::Packet* pkt) {
    return new PacketBuf(pkt);
  }

  juggler::dpdk::Packet* get_packet() const { return packet_; }

  uint8_t msg_flags() const { return msg_flags_; }
  bool is_first() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_SYN) != 0; }
  bool is_last() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_FIN) != 0; }
  FrameBuf* next() const { return next_; }
  void set_next(FrameBuf* next) { next_ = next; }
  void mark_first() { msg_flags_ |= UCCL_MSGBUF_FLAGS_SYN; }
  void mark_last() { msg_flags_ |= UCCL_MSGBUF_FLAGS_FIN; }

  void mark_txpulltime_free() {
    msg_flags_ |= UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE;
  }
  void mark_not_txpulltime_free() {
    msg_flags_ &= ~UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE;
  }
  bool is_txpulltime_free() const {
    return (msg_flags_ & UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE) != 0;
  }
  void set_msg_flags(uint16_t flags) { msg_flags_ = flags; }
  void add_msg_flags(uint16_t flags) { msg_flags_ |= flags; }
  void clear_fields() {
    next_ = nullptr;
    msg_flags_ = 0;
  }

  std::string to_string() const {
    std::stringstream s;
    s << " packet: 0x" << std::hex << reinterpret_cast<uint64_t>(packet_)
      << " packet_len: " << std::dec << packet_->length()
      << " msg_flags: " << std::bitset<8>(msg_flags_);
    return s.str();
  }

  std::string print_chain() {
    std::stringstream s;
    auto* cur = this;
    while (cur && !cur->is_last()) {
      s << cur->to_string();
      cur = cur->next_;
    }
    return s.str();
  }
};

class DpdkSocket {
  std::shared_ptr<juggler::dpdk::PmdPort> pmd_port_;
  int queue_id_;
  juggler::dpdk::RxRing* rx_ring_;
  juggler::dpdk::TxRing* tx_ring_;
  juggler::dpdk::PacketPool* packet_pool_;

  DpdkSocket(std::shared_ptr<juggler::dpdk::PmdPort> pmd_port,
             uint32_t rx_queue_id, uint32_t tx_queue_id) {
    pmd_port_ = CHECK_NOTNULL(pmd_port);
    rx_ring_ = pmd_port_->GetRing<juggler::dpdk::RxRing>(rx_queue_id);
    tx_ring_ = pmd_port_->GetRing<juggler::dpdk::TxRing>(tx_queue_id);
    packet_pool_ = CHECK_NOTNULL(tx_ring_->GetPacketPool());
  }

 public:
  inline void push_packet(juggler::dpdk::Packet* pkt) {
    juggler::dpdk::Packet::Free(pkt);
  }

  inline juggler::dpdk::Packet* pop_packet() {
    auto* pkt = packet_pool_->PacketAlloc();
    if (!pkt) {
      return nullptr;
    }
    return pkt;
  }

  uint32_t send_packet(juggler::dpdk::Packet* pkt) {
    std::vector<juggler::dpdk::Packet*> pkts;
    pkts.push_back(pkt);
    return send_packets(pkts);
  }

  uint32_t send_packets(std::vector<juggler::dpdk::Packet*>& pkts) {
    return tx_ring_->TrySendPackets(pkts.data(), pkts.size());
  }

  std::vector<juggler::dpdk::Packet*> recv_packets(uint32_t nb_pkts) {
    juggler::dpdk::Packet** pkts_ptr = new juggler::dpdk::Packet*[nb_pkts];
    uint16_t nb_recvd = rx_ring_->RecvPackets(pkts_ptr, nb_pkts);
    std::vector<juggler::dpdk::Packet*> pkts(pkts_ptr, pkts_ptr + nb_recvd);
    delete[] pkts_ptr;
    return pkts;
  }

  void shutdown() = delete;
  inline uint32_t unpulled_tx_pkts() const = delete;
  inline uint32_t send_queue_free_entries(uint32_t nb = UINT32_MAX) = delete;
  inline uint32_t kick_tx_and_pull() = delete;
  inline uint64_t send_queue_estimated_latency_ns() = delete;
};

class DpdkFactory {
 public:
  DpdkFactory(DpdkFactory const&) = delete;
  DpdkFactory& operator=(DpdkFactory const&) = delete;

  static DpdkFactory* Create(std::string const& conf_file) {
    if (instance_ == nullptr) {
      instance_ = new DpdkFactory(conf_file);
    }
    return instance_;
  }

 private:
  explicit DpdkFactory(std::string const& conf_file)
      : config_processor_{conf_file} {
    if (config_processor_.interfaces_config().empty()) {
      LOG(ERROR) << "No interfaces configured. Exiting.";
      return;
    }

    // Initialize DPDK.
    dpdk_.InitDpdk(config_processor_.GetEalOpts());
    if (dpdk_.GetNumPmdPortsAvailable() == 0) {
      LOG(ERROR) << "Error: No DPDK-capable ports found. This can be due to "
                    "several reasons.";
      LOG(ERROR)
          << "1. On Azure, the accelerated NIC must first be unbound from "
             "the kernel driver with driverctl";
      LOG(ERROR)
          << "2. The user libraries for the NIC are not installed, e.g., "
             "libmlx5 for Mellanox NICs";
      LOG(ERROR) << "3. The NIC is not supported by DPDK";
      return;
    }

    // Find the PMD port id for each interface.
    for (auto const& interface : config_processor_.interfaces_config()) {
      auto pmd_port_id = dpdk_.GetPmdPortIdByMac(interface.l2_addr());
      if (!pmd_port_id) {
        LOG(ERROR) << "Cannot find PMD port id for interface with L2 address: "
                   << interface.l2_addr().ToString();
        return;
      }
      const_cast<NetworkInterfaceConfig&>(interface).set_dpdk_port_id(
          pmd_port_id.value());

      // Initialize the PMD port.
      uint16_t const rx_rings_nr = NUM_QUEUES, tx_rings_nr = NUM_QUEUES;
      pmd_ports_.emplace_back(std::make_shared<juggler::dpdk::PmdPort>(
          interface.dpdk_port_id().value(), rx_rings_nr, tx_rings_nr,
          dpdk::PmdRing::kDefaultRingDescNr,
          dpdk::PmdRing::kDefaultRingDescNr));
      pmd_ports_.back()->InitDriver();

      break;
    }
  }

  static DpdkSocket* CreateSocket(int queue_id) {
    if (instance_ == nullptr) {
      LOG(ERROR) << "DpdkFactory not initialized";
      return nullptr;
    }

    if (instance_->pmd_ports_.empty()) {
      LOG(ERROR) << "No PMD ports available";
      return nullptr;
    }

    std::lock_guard<std::mutex> lock(instance_->socket_q_lock_);

    // Create new socket
    auto pmd_port = instance_->pmd_ports_[0];  // Using first port
    auto socket = new DpdkSocket(pmd_port, queue_id, queue_id);
    instance_->socket_q_.push_back(socket);

    return socket;
  }

  static void shutdown() {
    std::lock_guard<std::mutex> lock(instance_->socket_q_lock_);
    
    for (auto socket : instance_->socket_q_) {
      delete socket;
    }
    instance_->socket_q_.clear();

    for (auto& pmd_port : instance_->pmd_ports_) {
      pmd_port->DeInit();
    }
    instance_->pmd_ports_.clear();

    instance_->dpdk_.DeInitDpdk();

    delete instance_;
    instance_ = nullptr;
  }

 private:
  static inline DpdkFactory* instance_;
  juggler::dpdk::MachnetConfigProcessor config_processor_;
  juggler::dpdk::Dpdk dpdk_{};
  std::vector<std::shared_ptr<juggler::dpdk::PmdPort>> pmd_ports_{};
  std::deque<DpdkSocket*> socket_q_;
};
}  // namespace uccl