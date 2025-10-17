/**
 * @file  machnet_engine.h
 * @brief Contains the class definition for the main Machnet engine.
 */
#ifndef SRC_INCLUDE_MACHNET_ENGINE_H_
#define SRC_INCLUDE_MACHNET_ENGINE_H_

#include <arp.h>
#include <channel.h>
#include <common.h>
#include <ether.h>
#include <flow.h>
#include <icmp.h>
#include <ipv4.h>
#include <pmd.h>
#include <rte_thash.h>
#include <udp.h>

#include <concepts>
#include <cstddef>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace juggler {

/**
 * @brief Class `MachnetEngineSharedState' contains any state that might need to
 * be shared among different engines. This might be required when multiple
 * engine threads operate on a single PMD, sharing one or more IP addresses.
 */
class MachnetEngineSharedState {
public:
  static const size_t kSrcPortMin = (1 << 10);     // 1024
  static const size_t kSrcPortMax = (1 << 16) - 1; // 65535
  static constexpr size_t kSrcPortBitmapSize =
      (kSrcPortMax + 1) / sizeof(uint64_t) / 8;
  explicit MachnetEngineSharedState(std::vector<uint8_t> rss_key,
                                    net::Ethernet::Address l2addr,
                                    std::vector<net::Ipv4::Address> ipv4_addrs)
      : rss_key_(rss_key), arp_handler_(l2addr, ipv4_addrs) {
    for (const auto &addr : ipv4_addrs) {
      CHECK(ipv4_port_bitmap_.find(addr) == ipv4_port_bitmap_.end());
      ipv4_port_bitmap_.insert({addr, std::vector<uint64_t>(1, UINT64_MAX)});
    }
  }

  const std::unordered_map<net::Ipv4::Address, std::vector<uint64_t>> &
  GetIpv4PortBitmap() const {
    return ipv4_port_bitmap_;
  }

  bool IsLocalIpv4Address(const net::Ipv4::Address &ipv4_addr) const {
    return ipv4_port_bitmap_.find(ipv4_addr) != ipv4_port_bitmap_.end();
  }

  /**
   * @brief Allocates a source UDP port for a given IPv4 address based on a
   * specified predicate.
   *
   * This function searches for an available source UDP port in the range of
   * [kSrcPortMin, kSrcPortMax] that satisfies the provided predicate (lambda
   * function). If a suitable port is found, it is marked as used and returned
   * as an std::optional<net::Udp::Port> value. If no suitable port is found,
   * std::nullopt is returned.
   *
   * @tparam F A type satisfying the Predicate concept, invocable with a
   * uint16_t argument.
   * @param ipv4_addr The net::Ipv4::Address for which the source port is being
   * allocated.
   * @param lambda A predicate (lambda function) taking a uint16_t as input and
   * returning a bool. The function should return true if the input port number
   * is suitable for allocation. Default behavior is to accept any port.
   * @return std::optional<net::Udp::Port> containing the allocated source port
   * if found, or std::nullopt otherwise.
   *
   * Example usage:
   * @code
   * net::Ipv4::Address ipv4_addr = ...;
   * auto port = SrcPortAlloc(ipv4_addr, [](uint16_t port) { return port % 2 ==
   * 0; }); // Allocate an even port if (port.has_value()) {
   *   // Port successfully allocated, proceed with usage
   * } else {
   *   // No suitable port found
   * }
   * @endcode
   */
  std::optional<net::Udp::Port>
  SrcPortAlloc(const net::Ipv4::Address &ipv4_addr,
               std::predicate<uint16_t> auto &&lambda) {
    constexpr size_t bits_per_slot = sizeof(uint64_t) * 8;
    auto it = ipv4_port_bitmap_.find(ipv4_addr);
    if (it == ipv4_port_bitmap_.end()) {
      return std::nullopt;
    }

    // Helper lambda to find a free port.
    // Given a 64-bit wide slot in a bitmap (vector of uint64_t), find the first
    // available port that satisfies the lambda condition.
    auto find_free_port = [&lambda, &bits_per_slot](
                              auto &bitmap,
                              size_t index) -> std::optional<net::Udp::Port> {
      auto mask = ~0ULL;
      do {
        auto pos = __builtin_ffsll(bitmap[index] & mask);
        if (pos == 0)
          break; // This slot is fully used.
        const size_t candidate_port = index * bits_per_slot + pos - 1;
        if (candidate_port > kSrcPortMax)
          break; // Illegal port.
        if (lambda(candidate_port)) {
          bitmap[index] &= ~(1ULL << (pos - 1));
          return net::Udp::Port(candidate_port);
        }
        // If we reached the end of the slot and the port is not suitable,
        // abort.
        if (pos == sizeof(uint64_t) * 8)
          break;
        // Update the mask to skip the bits checked already.
        mask = (~0ULL) << pos;
      } while (true);

      return std::nullopt;
    };

    const std::lock_guard<std::mutex> lock(mtx_);
    auto &bitmap = it->second;
    // Calculate how many bitmap elements are required to cover port
    // kSrcPortMin.
    const size_t first_valid_slot = kSrcPortMin / bits_per_slot;
    if (bitmap.size() < first_valid_slot + 1) {
      bitmap.resize(first_valid_slot + 1, ~0ULL);
    }

    for (size_t i = first_valid_slot; i < bitmap.size(); i++) {
      if (bitmap[i] == 0)
        continue; // This slot is fully used.
      auto port = find_free_port(bitmap, i);
      if (port.has_value()) {
        return port;
      }
    }

    while (bitmap.size() < kSrcPortBitmapSize) {
      // We have exhausted the current bitmap, but there is still space to
      // allocate more ports.
      bitmap.emplace_back(~0ULL);
      auto port = find_free_port(bitmap, bitmap.size() - 1);
      if (port.has_value())
        return port;
    }

    return std::nullopt;
  }

  /**
   * @brief Releases a previously allocated UDP source port for the given IPv4
   * address.
   *
   * This function releases a source port that was previously allocated using
   * the SrcPortAlloc function. After releasing the port, it becomes available
   * for future allocation.
   *
   * @param ipv4_addr The IPv4 address for which the source port was allocated.
   * @param port The net::Udp::Port instance representing the allocated source
   * port to be released.
   *
   * @note Thread-safe, as it uses a lock_guard to protect concurrent access to
   * the shared data.
   *
   * Example usage:
   * @code
   * net::Ipv4::Address ipv4_addr = ...;
   * net::Udp::Port port = ...; // Previously allocated port
   * SrcPortRelease(ipv4_addr, port); // Release the allocated port
   * @endcode
   */
  void SrcPortRelease(const net::Ipv4::Address &ipv4_addr,
                      const net::Udp::Port &port) {
    const std::lock_guard<std::mutex> lock(mtx_);
    SrcPortReleaseLocked(ipv4_addr, port);
  }

  /**
   * @brief Registers a listener on a specific IPv4 address and UDP port,
   * associating the port with a receive queue.
   *
   * This function attempts to register a listener on the provided IPv4 address
   * and UDP port. If the port is available and not already in use, it will be
   * associated with the specified receive queue, and the function will return
   * true. If the port is already in use or the provided address and port are
   * not valid, the function returns false.
   *
   * @param ipv4_addr The IPv4 address on which to register the listener.
   * @param port The net::Udp::Port instance representing the source port to
   * listen on.
   * @param rx_queue_id The ID of the receive queue to associate with the
   * registered listener.
   *
   * @return A `bool` indicating whether the listener registration was
   * successful. Returns `true` if the port is available and the registration
   * operation is successful, `false` otherwise.
   *
   * @note Thread-safe, as it uses a lock_guard to protect concurrent access to
   * the shared data.
   *
   * Example usage:
   * @code
   * net::Ipv4::Address ipv4_addr = ...;
   * net::Udp::Port port = ...;
   * size_t rx_queue_id = ...;
   * bool success = RegisterListener(ipv4_addr, port, rx_queue_id); // Attempt
   * to register listener on the specified address and port
   * @endcode
   */
  bool RegisterListener(const net::Ipv4::Address &ipv4_addr,
                        const net::Udp::Port &port, size_t rx_queue_id) {
    const std::lock_guard<std::mutex> lock(mtx_);
    if (ipv4_port_bitmap_.find(ipv4_addr) == ipv4_port_bitmap_.end()) {
      return false;
    }

    constexpr size_t bits_per_slot = sizeof(uint64_t) * 8;
    auto p = port.port.value();
    const auto slot = p / bits_per_slot;
    const auto bit = p % bits_per_slot;
    auto &bitmap = ipv4_port_bitmap_[ipv4_addr];
    if (slot >= bitmap.size()) {
      // Allocate more elements in the bitmap.
      bitmap.resize(slot + 1, ~0ULL);
    }
    // Check if the port is already in use (i.e, bit is unset).
    if (!(bitmap[slot] & (1ULL << bit)))
      return false;
    bitmap[slot] &= ~(1ULL << bit);

    // Add the port and engine to the listeners.
    DCHECK(listeners_to_rxq.find({ipv4_addr, port}) == listeners_to_rxq.end());
    listeners_to_rxq[{ipv4_addr, port}] = rx_queue_id;

    return true;
  }

  /**
   * @brief Unregisters a listener on a specific IPv4 address and UDP port,
   * releasing the associated port.
   *
   * This function unregisters a listener on the provided IPv4 address and UDP
   * port, and releases the associated port. If the listener is not found, the
   * function does nothing.
   *
   * @param ipv4_addr The IPv4 address on which the listener was registered.
   * @param port The net::Udp::Port instance representing the source port
   * associated with the listener.
   *
   * @note Thread-safe, as it uses a lock_guard to protect concurrent access to
   * the shared data.
   *
   * Example usage:
   * @code
   * net::Ipv4::Address ipv4_addr = ...;
   * net::Udp::Port port = ...; // Previously registered port
   * UnregisterListener(ipv4_addr, port); // Unregister the listener and release
   * the associated port
   * @endcode
   */
  void UnregisterListener(const net::Ipv4::Address &ipv4_addr,
                          const net::Udp::Port &port) {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto it = listeners_to_rxq.find({ipv4_addr, port});
    if (it == listeners_to_rxq.end()) {
      return;
    }

    listeners_to_rxq.erase(it);
    SrcPortReleaseLocked(ipv4_addr, port);
  }

  std::optional<net::Ethernet::Address>
  GetL2Addr(const dpdk::TxRing *txring, const net::Ipv4::Address &local_ip,
            const net::Ipv4::Address &target_ip) {
    const std::lock_guard<std::mutex> lock(mtx_);
    return arp_handler_.GetL2Addr(txring, local_ip, target_ip);
  }

  void ProcessArpPacket(dpdk::TxRing *txring, net::Arp *arph) {
    const std::lock_guard<std::mutex> lock(mtx_);
    arp_handler_.ProcessArpPacket(txring, arph);
  }

  std::vector<std::tuple<std::string, std::string>> GetArpTableEntries() {
    const std::lock_guard<std::mutex> lock(mtx_);
    return arp_handler_.GetArpTableEntries();
  }

private:
  struct hash_ip_port_pair {
    template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U> &x) const {
      return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
    }
  };

  /**
   * @brief Private method to release a previously allocated UDP source port.
   *
   * @attention This method is not thread-safe, and should only be called when
   * locked.
   */
  void SrcPortReleaseLocked(const net::Ipv4::Address &ipv4_addr,
                            const net::Udp::Port &port) {
    auto it = ipv4_port_bitmap_.find(ipv4_addr);
    if (it == ipv4_port_bitmap_.end()) {
      return;
    }

    constexpr size_t bits_per_slot = sizeof(uint64_t) * 8;
    auto p = port.port.value();
    const auto slot = p / bits_per_slot;
    const auto bit = p % bits_per_slot;
    auto &bitmap = ipv4_port_bitmap_[ipv4_addr];
    bitmap[slot] |= (1ULL << bit);
  }

  const std::vector<uint8_t> rss_key_;
  ArpHandler arp_handler_;
  std::mutex mtx_{};
  std::unordered_map<net::Ipv4::Address, std::vector<uint64_t>>
      ipv4_port_bitmap_{};
  std::unordered_map<std::pair<net::Ipv4::Address, net::Udp::Port>, size_t,
                     hash_ip_port_pair>
      listeners_to_rxq{};
};

/**
 * @brief Class `MachnetEngine' abstracts the main Machnet engine. This engine
 * contains all the functionality need to be run by the stack's threads.
 */
class MachnetEngine {
public:
  using Ethernet = net::Ethernet;
  using Arp = net::Arp;
  using Ipv4 = net::Ipv4;
  using Udp = net::Udp;
  using Icmp = net::Icmp;
  using Flow = net::flow::Flow;
  using PmdPort = juggler::dpdk::PmdPort;
  // Slow timer (periodic processing) interval in microseconds.
  const size_t kSlowTimerIntervalUs = 1000000; // 2ms
  const size_t kPendingRequestTimeoutSlowTicks = 3;
  // Flow creation timeout in slow ticks (# of periodic executions since
  // flow creation request).
  const size_t kFlowCreationTimeoutSlowTicks = 3;
  MachnetEngine() = delete;
  MachnetEngine(MachnetEngine const &) = delete;

  /**
   * @brief Construct a new MachnetEngine object.
   *
   * @param pmd_port      Pointer to the PMD port to be used by the engine. The
   *                      PMD port must be initialized (i.e., call
   * InitDriver()).
   * @param rx_queue_id   RX queue index to be used by the engine.
   * @param tx_queue_id   TX queue index to be used by the engine. The TXRing
   *                      associated should be initialized with a packet pool.
   * @param channels      (optional) Machnet channels the engine will be
   *                      responsible for (if any).
   */
  MachnetEngine(std::shared_ptr<PmdPort> pmd_port, uint16_t rx_queue_id,
                uint16_t tx_queue_id,
                std::shared_ptr<MachnetEngineSharedState> shared_state,
                std::vector<std::shared_ptr<shm::Channel>> channels = {})
      : pmd_port_(CHECK_NOTNULL(pmd_port)),
        rxring_(pmd_port_->GetRing<dpdk::RxRing>(rx_queue_id)),
        txring_(pmd_port_->GetRing<dpdk::TxRing>(tx_queue_id)),
        packet_pool_(CHECK_NOTNULL(txring_->GetPacketPool())),
        shared_state_(CHECK_NOTNULL(shared_state)), channels_(channels),
        last_periodic_timestamp_(0), periodic_ticks_(0) {
    for (const auto &[ipv4_addr, _] : shared_state_->GetIpv4PortBitmap()) {
      listeners_.emplace(
          ipv4_addr,
          std::unordered_map<Udp::Port, std::shared_ptr<shm::Channel>>());
    }
  }

  /**
   * @brief Get the PMD port used by this engine.
   */
  std::shared_ptr<PmdPort> GetPmdPort() const { return pmd_port_; }

  // Adds a channel to be served by this engine.
  void AddChannel(std::shared_ptr<shm::Channel> channel,
                  std::promise<bool> &&status) {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto channel_info =
        std::make_tuple(std::move(CHECK_NOTNULL(channel)), std::move(status));
    channels_to_enqueue_.emplace_back(std::move(channel_info));
  }

  // Removes a channel from the engine.
  void RemoveChannel(std::shared_ptr<shm::Channel> channel) {
    const std::lock_guard<std::mutex> lock(mtx_);
    channels_to_dequeue_.emplace_back(std::move(channel));
  }

  /**
   * @brief This is the main event cycle of the Machnet engine.
   * It is called repeatedly by the main thread of the Machnet engine.
   * On each cycle, the engine processes incoming packets in the RX queue and
   * enqueued messages in all channels that it is responsible for.
   * This method is not thread-safe.
   *
   * @param now The current TSC.
   */
  void Run(uint64_t now) {
    // Calculate the time elapsed since the last periodic processing.
    const auto elapsed = time::cycles_to_us(now - last_periodic_timestamp_);
    if (elapsed >= kSlowTimerIntervalUs) {
      // Perform periodic processing.
      PeriodicProcess(now);
      last_periodic_timestamp_ = now;
    }

    juggler::dpdk::PacketBatch rx_packet_batch;
    const uint16_t nb_pkt_rx = rxring_->RecvPackets(&rx_packet_batch);
    for (uint16_t i = 0; i < nb_pkt_rx; i++) {
      const auto *pkt = rx_packet_batch.pkts()[i];
      process_rx_pkt(pkt, now);
    }

    // We have processed the RX batch; release it.
    rx_packet_batch.Release();

    // Process messages from channels.
    shm::MsgBufBatch msg_buf_batch;
    for (auto &channel : channels_) {
      // TODO(ilias): Revisit the number of messages to dequeue.
      const auto nb_msg_dequeued = channel->DequeueMessages(&msg_buf_batch);
      for (uint32_t i = 0; i < nb_msg_dequeued; i++) {
        auto *msg = msg_buf_batch.bufs()[i];
        process_msg(channel.get(), msg, now);
      }
      // We have processed the message batch; reset it.
      msg_buf_batch.Clear();
    }
  }

  /**
   * @brief Method to perform periodic processing. This is called by the main
   * engine cycle (see method `Run`).
   *
   * @param now The current TSC.
   */
  void PeriodicProcess(uint64_t now) {
    // Advance the periodic ticks counter.
    ++periodic_ticks_;
    HandleRTO();
    DumpStatus();
    ProcessControlRequests();
    // Continue the rest of management tasks locked to avoid race conditions
    // with the control plane.
    const std::lock_guard<std::mutex> lock(mtx_);
    // Refresh the list of active channels, if needed.
    ChannelsUpdate();
  }

  // Return the number of channels served by this engine.
  size_t GetChannelCount() const { return channels_.size(); }

protected:
  void DumpStatus() {
    std::string s;
    s += "[Machnet Engine Status]";
    s += "[PMD Port: " + std::to_string(pmd_port_->GetPortId()) +
         ", RX_Q: " + std::to_string(rxring_->GetRingId()) +
         ", TX_Q: " + std::to_string(txring_->GetRingId()) + "]\n";
    s += "\tLocal L2 address:\n";
    s += "\t\t" + pmd_port_->GetL2Addr().ToString() + "\n";
    s += "\tLocal IPv4 addresses:\n";
    s += "\t\t";
    for (const auto &[addr, _] : shared_state_->GetIpv4PortBitmap()) {
      s += addr.ToString();
      s += ",";
    }
    s += "\n";
    s += "\tActive channels:";
    for (const auto &channel : channels_) {
      s += "\n\t\t";
      s += "[" + channel->GetName() + "]" +
           " Total buffers: " + std::to_string(channel->GetTotalBufCount()) +
           ", Free buffers: " + std::to_string(channel->GetFreeBufCount());
    }
    s += "\n";
    s += "\tARP Table:\n";
    for (const auto &entry : shared_state_->GetArpTableEntries()) {
      s += "\t\t" + std::get<0>(entry) + " -> " + std::get<1>(entry) + "\n";
    }
    s += "\tListeners:\n";
    for (const auto &listeners_for_ip : listeners_) {
      const auto ip = listeners_for_ip.first.ToString();
      for (const auto &listener : listeners_for_ip.second) {
        s += "\t\t" + ip + ":" + std::to_string(listener.first.port.value()) +
             " <-> [" + listener.second->GetName() + "]" + "\n";
      }
    }
    s += "\tActive flows:\n";
    for (const auto &[key, flow_it] : active_flows_map_) {
      s += "\t\t";
      s += (*flow_it)->ToString();
      s += "\n";
    }
    s += "\n";
    LOG(INFO) << s;
  }

  /**
   * @brief This method curates the list of active channels under this engine.
   * It enqueues newly added channels to the list of active channels, and
   * removes/destroys channels pending for removal from the list.
   * We choose to curate the list in a separate method, so that the caller can
   * do this operation periodically, amortising the cost of locking.
   *
   * @attention This method should be executed periodically at the dataplane,
   * with the mutex held to achieve synchronization with the control plane
   * thread.
   */
  void ChannelsUpdate() {
    // TODO(ilias): For now, we assume that added channels do not carry any
    // flows (i.e., these are newly created channels).
    // If we want, dynamic load balancing (e.g., moving channels to different
    // engines, we should take care of flow migrations).
    for (auto it = channels_to_enqueue_.begin();
         it != channels_to_enqueue_.end();
         it = channels_to_enqueue_.erase(it)) {
      auto &channel_info = *it;
      const auto &channel = std::get<0>(channel_info);
      auto &status = std::get<1>(channel_info);

      channels_.emplace_back(std::move(channel));
      status.set_value(true);
    }

    // Remove channels pending for removal.
    for (auto &channel : channels_to_dequeue_) {
      const auto &it =
          std::find_if(channels_.begin(), channels_.end(),
                       [&channel](const auto &c) { return channel == c; });
      if (it == channels_.end()) {
        // This channel is not in the list of active channels.
        LOG(WARNING) << "Channel " << channel->GetName()
                     << " is not in the list of active channels";
        continue;
      }

      // Remove from the engine all listeners associated with this channel.
      const auto &channel_listeners = channel->GetListeners();
      for (const auto &ch_listener : channel_listeners) {
        const auto &local_ip = ch_listener.addr;
        const auto &local_port = ch_listener.port;

        if (listeners_.find(local_ip) == listeners_.end()) {
          LOG(ERROR) << "No listeners for IP " << local_ip.ToString();
          continue;
        }

        auto &listeners_for_ip = listeners_[local_ip];
        if (listeners_for_ip.find(local_port) == listeners_for_ip.end()) {
          LOG(ERROR) << utils::Format("Listener not found %s:%hu",
                                      local_ip.ToString().c_str(),
                                      local_port.port.value());
          continue;
        }

        shared_state_->UnregisterListener(local_ip, local_port);
        listeners_for_ip.erase(local_port);
      }

      const auto &channel_flows = channel->GetActiveFlows();
      // Remove from the engine's map all the flows associated with this
      // channel.
      for (const auto &flow : channel_flows) {
        if (active_flows_map_.find(flow->key()) != active_flows_map_.end()) {
          shared_state_->SrcPortRelease(flow->key().local_addr,
                                        flow->key().local_port);
          LOG(INFO) << "Removing flow " << flow->key().ToString();
          flow->ShutDown();
          active_flows_map_.erase(flow->key());
        } else {
          LOG(WARNING) << "Flow " << flow->key().ToString()
                       << " is not in the list of active flows";
        }
      }

      // Finally remove the channel.
      channels_.erase(it);
    }

    channels_to_dequeue_.clear();
  }

  /**
   * @brief This method polls active channels for all control plane requests and
   * processes them.
   * It is called periodically.
   */
  void ProcessControlRequests() {
    MachnetCtrlQueueEntry_t reqs[MACHNET_CHANNEL_CTRL_SQ_SLOT_NR];
    for (const auto &channel : channels_) {
      // Peek the control SQ.
      const auto nreqs =
          channel->DequeueCtrlRequests(reqs, MACHNET_CHANNEL_CTRL_SQ_SLOT_NR);
      for (auto i = 0u; i < nreqs; i++) {
        const auto &req = reqs[i];
        auto emit_completion = [&req, &channel](bool success) {
          MachnetCtrlQueueEntry_t resp;
          resp.id = req.id;
          resp.opcode = MACHNET_CTRL_OP_STATUS;
          resp.status =
              success ? MACHNET_CTRL_STATUS_OK : MACHNET_CTRL_STATUS_ERROR;
          channel->EnqueueCtrlCompletions(&resp, 1);
        };
        switch (req.opcode) {
        case MACHNET_CTRL_OP_CREATE_FLOW:
          // clang-format off
             {
               const Ipv4::Address src_addr(req.flow_info.src_ip);
               if (!shared_state_->IsLocalIpv4Address(src_addr)) {
                 LOG(ERROR) << "Source IP " << src_addr.ToString()
                            << " is not local. Cannot create flow.";
                 emit_completion(false);
                 break;
               }
               const Ipv4::Address dst_addr(req.flow_info.dst_ip);
               const Udp::Port dst_port(req.flow_info.dst_port);
               LOG(INFO) << "Request to create flow " << src_addr.ToString()
                         << " -> "
                         << dst_addr.ToString() << ":" << dst_port.port.value();
               pending_requests_.emplace_back(periodic_ticks_, req, channel);
             }
             break;
          // clang-format on
        case MACHNET_CTRL_OP_DESTROY_FLOW:
          break;
        case MACHNET_CTRL_OP_LISTEN:
          // clang-format off
             {
               const Ipv4::Address local_ip(req.listener_info.ip);
               const Udp::Port local_port(req.listener_info.port);
               if (!shared_state_->IsLocalIpv4Address(local_ip) ||
                   listeners_.find(local_ip) == listeners_.end()) {
               emit_completion(false);
               break;
               }
 
               auto &listeners_on_ip = listeners_[local_ip];
               if (listeners_on_ip.find(local_port) != listeners_on_ip.end()) {
                 LOG(ERROR) << "Cannot register listener for IP "
                            << local_ip.ToString() << " and port "
                            << local_port.port.value();
                 emit_completion(false);
                 break;
               }
 
               if (!shared_state_->RegisterListener(local_ip, local_port,
                                                    rxring_->GetRingId())) {
                 LOG(ERROR) << "Cannot register listener for IP "
                            << local_ip.ToString() << " and port "
                            << local_port.port.value();
                 emit_completion(false);
                 break;
               }
 
               listeners_on_ip.emplace(local_port, channel);
               channel->AddListener(local_ip, local_port);
               emit_completion(true);
             }
          // clang-format on
          break;
        default:
          LOG(ERROR) << "Unknown control plane request opcode: " << req.opcode;
        }
      }
    }

    for (auto it = pending_requests_.begin(); it != pending_requests_.end();) {
      const auto &[timestamp_, req, channel] = *it;
      if (periodic_ticks_ - timestamp_ > kPendingRequestTimeoutSlowTicks) {
        LOG(ERROR) << utils::Format(
            "Pending request timeout: [ID: %lu, Opcode: %u]", req.id,
            req.opcode);
        it = pending_requests_.erase(it);
        continue;
      }

      const Ipv4::Address src_addr(req.flow_info.src_ip);
      const Ipv4::Address dst_addr(req.flow_info.dst_ip);
      const Udp::Port dst_port(req.flow_info.dst_port);

      auto remote_l2_addr =
          shared_state_->GetL2Addr(txring_, src_addr, dst_addr);
      if (!remote_l2_addr.has_value()) {
        // L2 address has not been resolved yet.
        it++;
        continue;
      }

      // L2 address has been resolved. Allocate a source port.
      auto rss_lambda = [src_addr, dst_addr, dst_port,
                         rss_key = pmd_port_->GetRSSKey(), pmd_port = pmd_port_,
                         rx_queue_id =
                             rxring_->GetRingId()](uint16_t port) -> bool {
        rte_thash_tuple ipv4_l3_l4_tuple;
        ipv4_l3_l4_tuple.v4.src_addr = src_addr.address.value();
        ipv4_l3_l4_tuple.v4.dst_addr = dst_addr.address.value();
        ipv4_l3_l4_tuple.v4.sport = port;
        ipv4_l3_l4_tuple.v4.dport = dst_port.port.value();

        rte_thash_tuple reversed_ipv4_l3_l4_tuple;
        reversed_ipv4_l3_l4_tuple.v4.src_addr = dst_addr.address.value();
        reversed_ipv4_l3_l4_tuple.v4.dst_addr = src_addr.address.value();
        reversed_ipv4_l3_l4_tuple.v4.sport = dst_port.port.value();
        reversed_ipv4_l3_l4_tuple.v4.dport = port;

        auto rss_hash =
            rte_softrss(reinterpret_cast<uint32_t *>(&ipv4_l3_l4_tuple),
                        RTE_THASH_V4_L4_LEN, rss_key.data());
        auto reversed_rss_hash = rte_softrss(
            reinterpret_cast<uint32_t *>(&reversed_ipv4_l3_l4_tuple),
            RTE_THASH_V4_L4_LEN, rss_key.data());
        if (pmd_port->GetRSSRxQueue(reversed_rss_hash) != rx_queue_id) {
          return false;
        }

        if (pmd_port->GetRSSRxQueue(__builtin_bswap32(reversed_rss_hash)) !=
            rx_queue_id) {
          return false;
        }

        LOG(INFO) << "RSS hash for " << src_addr.ToString() << ":" << port
                  << " -> " << dst_addr.ToString() << ":"
                  << dst_port.port.value() << " is " << rss_hash
                  << " and reversed " << reversed_rss_hash
                  << " (queue: " << rx_queue_id << ")";

        return true;
      };

      auto src_port = shared_state_->SrcPortAlloc(src_addr, rss_lambda);
      if (!src_port.has_value()) {
        LOG(ERROR) << "Cannot allocate source port for " << src_addr.ToString();
        it = pending_requests_.erase(it);
        continue;
      }

      auto application_callback =
          [req_id = req.id](shm::Channel *channel, bool success,
                            const juggler::net::flow::Key &flow_key) {
            MachnetCtrlQueueEntry_t resp;
            resp.id = req_id;
            resp.opcode = MACHNET_CTRL_OP_STATUS;
            resp.status =
                success ? MACHNET_CTRL_STATUS_OK : MACHNET_CTRL_STATUS_ERROR;
            resp.flow_info.src_ip = flow_key.local_addr.address.value();
            resp.flow_info.src_port = flow_key.local_port.port.value();
            resp.flow_info.dst_ip = flow_key.remote_addr.address.value();
            resp.flow_info.dst_port = flow_key.remote_port.port.value();
            channel->EnqueueCtrlCompletions(&resp, 1);
          };
      const auto &flow_it =
          channel->CreateFlow(src_addr, src_port.value(), dst_addr, dst_port,
                              pmd_port_->GetL2Addr(), remote_l2_addr.value(),
                              txring_, application_callback);
      (*flow_it)->InitiateHandshake();
      active_flows_map_.emplace((*flow_it)->key(), flow_it);
      it = pending_requests_.erase(it);
    }
  }

  /**
   * @brief Iterate throught the list of flows, check and handle RTOs.
   */
  void HandleRTO() {
    for (auto it = active_flows_map_.begin(); it != active_flows_map_.end();) {
      const auto &flow_it = it->second;
      auto is_active_flow = (*flow_it)->PeriodicCheck();
      if (!is_active_flow) {
        LOG(INFO) << "Flow " << (*flow_it)->key().ToString()
                  << " is no longer active. Removing.";
        auto channel = (*flow_it)->channel();
        shared_state_->SrcPortRelease((*flow_it)->key().local_addr,
                                      (*flow_it)->key().local_port);
        channel->RemoveFlow(flow_it);
        it = active_flows_map_.erase(it);
        continue;
      }
      ++it;
    }
  }

  /**
   * @brief Process an incoming packet.
   *
   * @param pkt Pointer to the packet.
   * @param now TSC timestamp.
   */
  void process_rx_pkt(const juggler::dpdk::Packet *pkt, uint64_t now) {
    // Sanity ethernet header check.
    if (pkt->length() < sizeof(Ethernet)) [[unlikely]]
      return;

    auto *eh = pkt->head_data<Ethernet *>();
    switch (eh->eth_type.value()) {
      // clang-format off
       case Ethernet::kArp:
         {
           auto *arph = pkt->head_data<Arp *>(sizeof(*eh));
           shared_state_->ProcessArpPacket(txring_, arph);
         }
    // clang-format on
    break;
      // clang-format off
       [[likely]] case Ethernet::kIpv4:
           process_rx_ipv4(pkt, now);
         break;
    // clang-format on
    case Ethernet::kIpv6:
      // We do not support IPv6 yet.
      break;
    default:
      break;
    }
  }

  void process_rx_ipv4(const juggler::dpdk::Packet *pkt, uint64_t now) {
    // Sanity ipv4 header check.
    if (pkt->length() < sizeof(Ethernet) + sizeof(Ipv4)) [[unlikely]]
      return;

    const auto *eh = pkt->head_data<Ethernet *>();
    const auto *ipv4h = pkt->head_data<Ipv4 *>(sizeof(Ethernet));
    const auto *udph = pkt->head_data<Udp *>(sizeof(Ethernet) + sizeof(Ipv4));

    const net::flow::Key pkt_key(ipv4h->dst_addr, udph->dst_port,
                                 ipv4h->src_addr, udph->src_port);
    // Check ivp4 header length.
    // clang-format off
     if (pkt->length() != sizeof(Ethernet) + ipv4h->total_length.value()) [[unlikely]] { // NOLINT
      // clang-format on
      LOG(WARNING) << "IPv4 packet length mismatch (expected: "
                   << ipv4h->total_length.value()
                   << ", actual: " << pkt->length() << ")";
      return;
    }

    switch (ipv4h->next_proto_id) {
      // clang-format off
       [[likely]] case Ipv4::kUdp:
      // clang-format on
      if (active_flows_map_.find(pkt_key) != active_flows_map_.end()) {
        const auto &flow_it = active_flows_map_[pkt_key];
        (*flow_it)->InputPacket(pkt);
        return;
      }

      {
        // If we reach here, it means that the packet does not belong to any
        // active flow.
        // Check if there is a listener on this port.
        const auto &local_ipv4_addr = ipv4h->dst_addr;
        const auto &local_udp_port = udph->dst_port;
        if (listeners_.find(local_ipv4_addr) != listeners_.end()) {
          // We have a listener on this port.
          const auto &listeners_on_ip = listeners_[local_ipv4_addr];
          if (listeners_on_ip.find(local_udp_port) == listeners_on_ip.end()) {
            LOG(INFO) << "Dropping packet with RSS hash: " << pkt->rss_hash()
                      << " (be: " << __builtin_bswap32(pkt->rss_hash()) << ")"
                      << " because there is no listener on port "
                      << local_udp_port.port.value()
                      << " (engine @rx_q_id: " << rxring_->GetRingId() << ")";
            return;
          }

          // Create a new flow.
          const auto &channel = listeners_on_ip.at(local_udp_port);
          const auto &remote_ipv4_addr = ipv4h->src_addr;
          const auto &remote_udp_port = udph->src_port;

          // Check if it is a SYN packet.
          const auto *machneth = pkt->head_data<net::MachnetPktHdr *>(
              sizeof(Ethernet) + sizeof(Ipv4) + sizeof(Udp));
          if (machneth->net_flags != net::MachnetPktHdr::MachnetFlags::kSyn) {
            LOG(WARNING) << "Received a non-SYN packet on a listening port";
            break;
          }

          auto empty_callback = [](shm::Channel *, bool,
                                   const net::flow::Key &) {};
          const auto &flow_it = channel->CreateFlow(
              local_ipv4_addr, local_udp_port, remote_ipv4_addr,
              remote_udp_port, pmd_port_->GetL2Addr(), eh->src_addr, txring_,
              empty_callback);
          active_flows_map_.insert({pkt_key, flow_it});

          // Handle the incoming packet.
          (*flow_it)->InputPacket(pkt);
        }
      }

      break;
      // clang-format off
       case Ipv4::kIcmp:
       {
         if (pkt->length() < sizeof(Ethernet) + sizeof(Ipv4) + sizeof(Icmp))
           [[unlikely]] return;
 
         const auto *icmph =
             pkt->head_data<Icmp *>(sizeof(Ethernet) + sizeof(Ipv4));
         // Only process ICMP echo requests.
         if (icmph->type != Icmp::kEchoRequest) [[unlikely]] return;
 
         // Allocate and construct a new packet for the response, instead of
         // in-place modification.
         // If `FAST_FREE' is enabled it's unsafe to use packets from different
         // pools (the driver may put them in the wrong pool on reclaim).
         auto *response = CHECK_NOTNULL(packet_pool_->PacketAlloc());
         auto *response_eh = response->append<Ethernet *>(pkt->length());
         response_eh->dst_addr = eh->src_addr;
         response_eh->src_addr = pmd_port_->GetL2Addr();
         response_eh->eth_type = be16_t(Ethernet::kIpv4);
         response->set_l2_len(sizeof(*response_eh));
         auto *response_ipv4h = reinterpret_cast<Ipv4 *>(response_eh + 1);
         response_ipv4h->version_ihl = 0x45;
         response_ipv4h->type_of_service = 0;
         response_ipv4h->packet_id = be16_t(0x1513);
         response_ipv4h->fragment_offset = be16_t(0);
         response_ipv4h->time_to_live = 64;
         response_ipv4h->next_proto_id = Ipv4::Proto::kIcmp;
         response_ipv4h->total_length = be16_t(pkt->length() - sizeof(Ethernet));
         response_ipv4h->src_addr = ipv4h->dst_addr;
         response_ipv4h->dst_addr = ipv4h->src_addr;
         response_ipv4h->hdr_checksum = 0;
         response->set_l3_len(sizeof(*response_ipv4h));
         response->offload_ipv4_csum();
         auto *response_icmph = reinterpret_cast<Icmp *>(response_ipv4h + 1);
         response_icmph->type = Icmp::kEchoReply;
         response_icmph->code = Icmp::kCodeZero;
         response_icmph->cksum = 0;
         response_icmph->id = icmph->id;
         response_icmph->seq = icmph->seq;
 
         auto *response_data = reinterpret_cast<uint8_t *>(response_icmph + 1);
         const auto *request_data =
             reinterpret_cast<const uint8_t *>(icmph + 1);
         utils::Copy(
             response_data, request_data,
             pkt->length() - sizeof(Ethernet) - sizeof(Ipv4) - sizeof(Icmp));
         response_icmph->cksum = utils::ComputeChecksum16(
             reinterpret_cast<const uint8_t *>(response_icmph),
             pkt->length() - sizeof(Ethernet) - sizeof(Ipv4));
 
         auto nsent = txring_->TrySendPackets(&response, 1);
         LOG_IF(WARNING, nsent != 1) << "Failed to send ICMP echo reply";
       }
    // clang-format on

    break;
    default:
      LOG(WARNING) << "Unsupported IP protocol: "
                   << static_cast<uint32_t>(ipv4h->next_proto_id);
      break;
    }
  }

  /**
   * Process a message enqueued from an application to a channel.
   * @param channel A pointer to the channel that the message was enqueued to.
   * @param msg     A pointer to the `MsgBuf` containing the first buffer of the
   *                message.
   */
  void process_msg(const shm::Channel *channel, shm::MsgBuf *msg,
                   uint64_t now) {
    const auto *flow_info = msg->flow();
    const net::flow::Key msg_key(flow_info->src_ip, flow_info->src_port,
                                 flow_info->dst_ip, flow_info->dst_port);
    if (active_flows_map_.find(msg_key) == active_flows_map_.end()) {
      LOG(ERROR) << "Message received for a non-existing flow! "
                 << utils::Format("(Channel: %s, 5-tuple hash: %lu, Flow: %s)",
                                  channel->GetName().c_str(),
                                  std::hash<net::flow::Key>{}(msg_key),
                                  msg_key.ToString().c_str());
      return;
    }
    const auto &flow_it = active_flows_map_[msg_key];
    (*flow_it)->OutputMessage(msg);
  }

private:
  using channel_info =
      std::tuple<std::shared_ptr<shm::Channel>, std::promise<bool>>;
  using flow_info =
      std::tuple<uint64_t, Ipv4::Address, Udp::Port, Ipv4::Address, Udp::Port,
                 std::shared_ptr<shm::Channel>, std::promise<bool>>;
  using listener_info =
      std::tuple<Ipv4::Address, Udp::Port, std::shared_ptr<shm::Channel>,
                 std::promise<bool>>;
  static const size_t kSrcPortMin = (1 << 10);     // 1024
  static const size_t kSrcPortMax = (1 << 16) - 1; // 65535
  static constexpr size_t kSrcPortBitmapSize =
      ((kSrcPortMax - kSrcPortMin + 1) + sizeof(uint64_t) - 1) /
      sizeof(uint64_t);
  // A mutex to synchronize control plane operations.
  std::mutex mtx_;
  // A shared pointer to the PmdPort instance.
  std::shared_ptr<PmdPort> pmd_port_;
  // Designated RX queue for this engine (not shared).
  juggler::dpdk::RxRing *rxring_;
  // Designated TX queue for this engine (not shared).
  juggler::dpdk::TxRing *txring_;
  // The following packet pool is used for all TX packets; should not be shared
  // with other engines/threads.
  dpdk::PacketPool *packet_pool_;
  // Shared State instance for this engine.
  std::shared_ptr<MachnetEngineSharedState> shared_state_;
  // Local IPv4 addresses bound to this engine/interface.
  std::unordered_set<Ipv4::Address> local_ipv4_addrs_;
  // Vector of active channels this engine is serving.
  std::vector<std::shared_ptr<shm::Channel>> channels_;
  // Timestamp of last periodic process execution.
  uint64_t last_periodic_timestamp_{0};
  // Clock ticks for the slow timer.
  uint64_t periodic_ticks_{0};
  // Listeners for incoming packets.
  std::unordered_map<
      Ipv4::Address,
      std::unordered_map<Udp::Port, std::shared_ptr<shm::Channel>>>
      listeners_{};
  // Unordered map of active flows.
  std::unordered_map<net::flow::Key,
                     const std::list<std::unique_ptr<Flow>>::const_iterator>
      active_flows_map_{};
  // Vector of channels to be added to the list of active channels.
  std::vector<channel_info> channels_to_enqueue_{};
  // Vector of channels to be removed from the list of active channels.
  std::vector<std::shared_ptr<shm::Channel>> channels_to_dequeue_{};
  // List of pending control plane requests.
  std::list<std::tuple<uint64_t, MachnetCtrlQueueEntry_t,
                       const std::shared_ptr<shm::Channel>>>
      pending_requests_{};
};

} // namespace juggler

#endif // SRC_INCLUDE_MACHNET_ENGINE_H_