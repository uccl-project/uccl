#pragma once

#include <glog/logging.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "transport_cc.h"
#include "util.h"
#include "util_afxdp.h"
#include "util_endian.h"

namespace uccl {

typedef uint64_t ConnectionID;

struct ChannelMsg {
    enum ChannelOp : uint8_t {
        kTx = 0,
        kTxComp = 1,
        kRx = 2,
        kRxComp = 3,
    };
    ChannelOp opcode;
    void *data;
    size_t *len_ptr;
    ConnectionID connection_id;
};
static_assert(sizeof(ChannelMsg) % 4 == 0, "channelMsg must be 32-bit aligned");

class Channel {
    constexpr static uint32_t kChannelSize = 1024;

   public:
    Channel() {
        tx_ring_ = create_ring(sizeof(ChannelMsg), kChannelSize);
        tx_comp_ring_ = create_ring(sizeof(ChannelMsg), kChannelSize);
        rx_ring_ = create_ring(sizeof(ChannelMsg), kChannelSize);
        rx_comp_ring_ = create_ring(sizeof(ChannelMsg), kChannelSize);
    }

    ~Channel() {
        free(tx_ring_);
        free(tx_comp_ring_);
        free(rx_ring_);
        free(rx_comp_ring_);
    }

    jring_t *tx_ring_;
    jring_t *tx_comp_ring_;
    jring_t *rx_ring_;
    jring_t *rx_comp_ring_;
};

class Endpoint {
    constexpr static uint16_t kBootstrapPort = 40000;
    Channel *channel_;

   public:
    // This function bind this endpoint to a specific local network interface
    // with the IP specified by the interface. It also listens on incoming
    // Connect() requests to estabish connections. Each connection is identified
    // by a unique connection_id, and uses multiple src+dst port combinations to
    // leverage multiple paths. Under the hood, we leverage TCP to boostrap our
    // connections. We do not consider multi-tenancy for now, assuming this
    // endpoint exclusively uses the NIC and its all queues.
    Endpoint(Channel *channel) : channel_(channel) {}
    ~Endpoint() {}

    // Connecting to a remote address.
    ConnectionID Connect(uint32_t remote_ip) {
        // TODO: Using TCP to negotiate a ConnectionID.
        return 0xdeadbeaf;
    }

    ConnectionID Accept() {
        // TODO: Using TCP to negotiate a ConnectionID.
        return 0xdeadbeaf;
    }

    // Sending the data by leveraging multiple port combinations.
    bool Send(ConnectionID connection_id, const void *data, size_t len) {
        ChannelMsg msg = {
            .opcode = ChannelMsg::ChannelOp::kTx,
            .data = const_cast<void *>(data),
            .len_ptr = &len,
            .connection_id = connection_id,
        };
        while (jring_sp_enqueue_bulk(channel_->tx_ring_, &msg, 1, nullptr) !=
               1) {
            // do nothing
        }
        // Wait for the completion.
        while (jring_sc_dequeue_bulk(channel_->tx_comp_ring_, &msg, 1,
                                     nullptr) != 1) {
            // do nothing
        }
        return true;
    }

    // Receiving the data by leveraging multiple port combinations.
    bool Recv(ConnectionID connection_id, void *data, size_t *len) {
        ChannelMsg msg = {
            .opcode = ChannelMsg::ChannelOp::kRx,
            .data = data,
            .len_ptr = len,
            .connection_id = connection_id,
        };
        while (jring_sp_enqueue_bulk(channel_->rx_ring_, &msg, 1, nullptr) !=
               1) {
            // do nothing
        }
        // Wait for the completion.
        while (jring_sc_dequeue_bulk(channel_->rx_comp_ring_, &msg, 1,
                                     nullptr) != 1) {
            // do nothing
        }
        return true;
    }
};

/**
 * @struct Key
 * @brief UcclFlow key: corresponds to the 5-tuple (UDP is always the protocol).
 */
struct Key {
    Key(const Key &other) = default;
    /**
     * @brief Construct a new Key object.
     *
     * @param local_addr Local IP address (in host byte order).
     * @param local_port Local UDP port (in host byte order).
     * @param remote_addr Remote IP address (in host byte order).
     * @param remote_port Remote UDP port (in host byte order).
     */
    Key(const uint32_t local_addr, const uint16_t local_port,
        const uint32_t remote_addr, const uint16_t remote_port)
        : local_addr(local_addr),
          local_port(local_port),
          remote_addr(remote_addr),
          remote_port(remote_port) {}

    bool operator==(const Key &other) const {
        return local_addr == other.local_addr &&
               local_port == other.local_port &&
               remote_addr == other.remote_addr &&
               remote_port == other.remote_port;
    }

    std::string ToString() const {
        return Format("[%x:%hu <-> %x:%hu]", remote_addr, remote_port,
                      local_addr, local_port);
    }

    const uint32_t local_addr;
    const uint32_t remote_addr;
    const uint16_t local_port;
    const uint16_t remote_port;
};
static_assert(sizeof(Key) == 12, "UcclFlow key size is not 12 bytes.");

/**
 * Uccl Packet Header.
 */
struct __attribute__((packed)) UcclPktHdr {
    static constexpr uint16_t kMagic = 0x4e53;
    be16_t magic;  // Magic value tagged after initialization for the flow.
    enum class UcclFlags : uint8_t {
        kData = 0b0,
        kSyn = 0b1,         // SYN packet.
        kAck = 0b10,        // ACK packet.
        kSynAck = 0b11,     // SYN-ACK packet.
        kRst = 0b10000000,  // RST packet.
    };
    UcclFlags net_flags;  // Network flags.
    uint8_t msg_flags;    // Field to reflect the `FrameBuf' flags.
    be32_t seqno;  // Sequence number to denote the packet counter in the flow.
    be32_t ackno;  // Sequence number to denote the packet counter in the flow.
    be64_t sack_bitmap[4];     // Bitmap of the SACKs received.
    be16_t sack_bitmap_count;  // Length of the SACK bitmap [0-256].
    be64_t timestamp1;         // Timestamp of the packet before sending.
};
static_assert(sizeof(UcclPktHdr) == 54, "UcclPktHdr size mismatch");

inline UcclPktHdr::UcclFlags operator|(UcclPktHdr::UcclFlags lhs,
                                       UcclPktHdr::UcclFlags rhs) {
    using UcclFlagsType = std::underlying_type<UcclPktHdr::UcclFlags>::type;
    return UcclPktHdr::UcclFlags(static_cast<UcclFlagsType>(lhs) |
                                 static_cast<UcclFlagsType>(rhs));
}

inline UcclPktHdr::UcclFlags operator&(UcclPktHdr::UcclFlags lhs,
                                       UcclPktHdr::UcclFlags rhs) {
    using UcclFlagsType = std::underlying_type<UcclPktHdr::UcclFlags>::type;
    return UcclPktHdr::UcclFlags(static_cast<UcclFlagsType>(lhs) &
                                 static_cast<UcclFlagsType>(rhs));
}

class FrameBuf {
    // Pointing to the next message buffer in the chain.
    FrameBuf *next_;
    // Describing the packet frame address and length.
    uint64_t frame_offset_;
    void *umem_buffer_;
    uint32_t frame_len_;
    // Flags to denote the message buffer state.
#define UCCL_MSGBUF_FLAGS_SYN (1 << 0)
#define UCCL_MSGBUF_FLAGS_FIN (1 << 1)
    uint8_t msg_flags_;

    FrameBuf(uint64_t frame_offset, void *umem_buffer, uint32_t frame_len)
        : frame_offset_(frame_offset),
          umem_buffer_(umem_buffer),
          frame_len_(frame_len) {
        next_ = nullptr;
        msg_flags_ = 0;
    }

   public:
    static FrameBuf *Create(uint64_t frame_offset, void *umem_buffer,
                            uint32_t frame_len) {
        // The XDP_PACKET_HEADROOM bytes before frame_offset is xdp metedata,
        // and we reuse it to chain Framebufs.
        return new (reinterpret_cast<void *>(
            frame_offset + (uint64_t)umem_buffer - XDP_PACKET_HEADROOM))
            FrameBuf(frame_offset, umem_buffer, frame_len);
    }
    uint64_t get_frame_offset() const { return frame_offset_; }
    void *get_umem_buffer() const { return umem_buffer_; }
    uint32_t get_frame_len() const { return frame_len_; }
    uint8_t *get_pkt_addr() const {
        return (uint8_t *)umem_buffer_ + frame_offset_;
    }

    uint16_t msg_flags() const { return msg_flags_; }

    // Returns true if this is the first in a message.
    bool is_first() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_SYN) != 0; }
    // Returns true if this is the last in a message.
    bool is_last() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_FIN) != 0; }

    // Returns the next message buffer index in the chain.
    FrameBuf *next() const { return next_; }
    // Set the next message buffer index in the chain.
    void set_next(FrameBuf *next) { next_ = next; }
    // Link the message train to the current message train. The start and end of
    // each message are still preserved.
    void link_msg_train(FrameBuf *next) {
        // next_ = next;
        DCHECK(is_last()) << "This is not the last buffer of a message!";
        DCHECK(next->is_first())
            << "The next buffer is not the first of a message!";
        next_ = next;
    }

    void mark_first() { add_msg_flags(UCCL_MSGBUF_FLAGS_SYN); }
    void mark_last() { add_msg_flags(UCCL_MSGBUF_FLAGS_FIN); }

    void set_msg_flags(uint16_t flags) { msg_flags_ = flags; }
    void add_msg_flags(uint16_t flags) { msg_flags_ |= flags; }

    Key get_flow() const {
        const auto *pkt_addr =
            reinterpret_cast<uint8_t *>(umem_buffer_) + frame_offset_;
        const auto *ih =
            reinterpret_cast<const iphdr *>(pkt_addr + sizeof(ethhdr));
        const auto *udph = reinterpret_cast<const udphdr *>(
            pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
        return Key(ih->saddr, udph->source, ih->daddr, udph->dest);
    }
};

class TXTracking {
   public:
    TXTracking() = delete;
    TXTracking(AFXDPSocket *socket)
        : socket_(socket),
          oldest_unacked_msgbuf_(nullptr),
          oldest_unsent_msgbuf_(nullptr),
          last_msgbuf_(nullptr),
          num_unsent_msgbufs_(0),
          num_tracked_msgbufs_(0) {}

    const uint32_t NumUnsentMsgbufs() const { return num_unsent_msgbufs_; }
    FrameBuf *GetOldestUnackedMsgBuf() const { return oldest_unacked_msgbuf_; }

    void ReceiveAcks(uint32_t num_acked_pkts) {
        while (num_acked_pkts) {
            auto msgbuf = oldest_unacked_msgbuf_;
            DCHECK(msgbuf != nullptr);
            if (msgbuf != last_msgbuf_) {
                DCHECK_NE(oldest_unacked_msgbuf_, oldest_unsent_msgbuf_)
                    << "Releasing an unsent msgbuf!";
                oldest_unacked_msgbuf_ = msgbuf->next();
            } else {
                oldest_unacked_msgbuf_ = nullptr;
                last_msgbuf_ = nullptr;
            }
            // Free acked frames
            socket_->frame_pool_->push(msgbuf->get_frame_offset());
            num_tracked_msgbufs_--;
            num_acked_pkts--;
        }
    }

    void Append(FrameBuf *msgbuf) {
        // Append the message at the end of the chain of buffers, if any.
        if (last_msgbuf_ == nullptr) {
            // This is the first pending message buffer in the flow.
            DCHECK(oldest_unsent_msgbuf_ == nullptr);
            last_msgbuf_ = msgbuf;
            oldest_unsent_msgbuf_ = msgbuf;
            oldest_unacked_msgbuf_ = msgbuf;
        } else {
            // This is not the first message buffer in the flow.
            DCHECK(oldest_unacked_msgbuf_ != nullptr);
            // Let's enqueue the new message buffer at the end of the chain.
            last_msgbuf_->set_next(msgbuf);
            // Update the last buffer pointer to point to the current buffer.
            last_msgbuf_ = msgbuf;
            if (oldest_unsent_msgbuf_ == nullptr)
                oldest_unsent_msgbuf_ = msgbuf;
        }

        num_unsent_msgbufs_ += 1;
        num_tracked_msgbufs_ += 1;
    }

    std::optional<FrameBuf *> GetAndUpdateOldestUnsent() {
        if (oldest_unsent_msgbuf_ == nullptr) {
            DCHECK_EQ(NumUnsentMsgbufs(), 0);
            return std::nullopt;
        }

        auto msgbuf = oldest_unsent_msgbuf_;
        if (oldest_unsent_msgbuf_ != last_msgbuf_) {
            oldest_unsent_msgbuf_ = oldest_unsent_msgbuf_->next();
        } else {
            oldest_unsent_msgbuf_ = nullptr;
        }

        num_unsent_msgbufs_--;
        return msgbuf;
    }

   private:
    const uint32_t NumTrackedMsgbufs() const { return num_tracked_msgbufs_; }
    const FrameBuf *GetLastMsgBuf() const { return last_msgbuf_; }
    const FrameBuf *GetOldestUnsentMsgBuf() const {
        return oldest_unsent_msgbuf_;
    }

    AFXDPSocket *socket_;

    /*
     * For the linked list of FrameBufs in the channel (chain going
     * downwards), we track 3 pointers
     *
     * B   -> oldest sent but unacknowledged MsgBuf
     * ...
     * B   -> oldest unsent MsgBuf
     * ...
     * B   -> last MsgBuf, among all active messages in this flow
     */

    FrameBuf *oldest_unacked_msgbuf_;
    FrameBuf *oldest_unsent_msgbuf_;
    FrameBuf *last_msgbuf_;

    uint32_t num_unsent_msgbufs_;
    uint32_t num_tracked_msgbufs_;
};

/**
 * @class RXTracking
 * @brief Tracking for message buffers that are received from the network. This
 * class is handling out-of-order reception of packets, and delivers complete
 * messages to the application.
 */
class RXTracking {
   public:
    // 256-bit SACK bitmask => we can track up to 256 packets
    static constexpr std::size_t kReassemblyMaxSeqnoDistance =
        sizeof(sizeof(UcclPktHdr::sack_bitmap)) * 8;

    static_assert((kReassemblyMaxSeqnoDistance &
                   (kReassemblyMaxSeqnoDistance - 1)) == 0,
                  "kReassemblyMaxSeqnoDistance must be a power of two");

    struct reasm_queue_ent_t {
        FrameBuf *msgbuf;
        uint64_t seqno;

        reasm_queue_ent_t(FrameBuf *m, uint64_t s) : msgbuf(m), seqno(s) {}
    };

    RXTracking(const RXTracking &) = delete;
    RXTracking(uint32_t local_ip, uint16_t local_port, uint32_t remote_ip,
               uint16_t remote_port, AFXDPSocket *socket)
        : local_ip_(local_ip),
          local_port_(local_port),
          remote_ip_(remote_ip),
          remote_port_(remote_port),
          socket_(socket),
          cur_msg_train_head_(nullptr),
          cur_msg_train_tail_(nullptr),
          app_buf_pos_(0) {}

    // If we fail to allocate in the SHM channel, return -1.
    int Consume(swift::Pcb *pcb, FrameBuf *msgbuf, void *app_buf,
                size_t *app_buf_len) {
        const size_t net_hdr_len =
            sizeof(ethhdr) + sizeof(iphdr) + sizeof(udphdr);
        uint8_t *pkt = msgbuf->get_pkt_addr();
        auto frame_len = msgbuf->get_frame_len();
        const auto *ucclh =
            reinterpret_cast<const UcclPktHdr *>(msgbuf + net_hdr_len);
        const auto *payload = reinterpret_cast<const UcclPktHdr *>(
            msgbuf + net_hdr_len + sizeof(UcclPktHdr));
        const auto seqno = ucclh->seqno.value();
        const auto expected_seqno = pcb->rcv_nxt;

        if (swift::seqno_lt(seqno, expected_seqno)) {
            VLOG(2) << "Received old packet: " << seqno << " < "
                    << expected_seqno;
            return 0;
        }

        const size_t distance = seqno - expected_seqno;
        if (distance >= kReassemblyMaxSeqnoDistance) {
            LOG(ERROR)
                << "Packet too far ahead. Dropping as we can't handle SACK. "
                << "seqno: " << seqno << ", expected: " << expected_seqno;
            return 0;
        }

        // Only iterate through the deque if we must, i.e., for ooo packts only
        auto it = reass_q_.begin();
        if (seqno != expected_seqno) {
            it = std::find_if(reass_q_.begin(), reass_q_.end(),
                              [&seqno](const reasm_queue_ent_t &entry) {
                                  return entry.seqno >= seqno;
                              });
            if (it != reass_q_.end() && it->seqno == seqno) {
                return 0;  // Duplicate packet
            }
        }

        // Buffer the packet in the frame pool. It may be out-of-order.
        const size_t payload_len = frame_len - net_hdr_len - sizeof(UcclPktHdr);
        // This records the incoming network packet UcclPktHdr.msg_flags in
        // FrameBuf.
        msgbuf->set_msg_flags(ucclh->msg_flags);

        if (seqno == expected_seqno) {
            reass_q_.emplace_front(msgbuf, seqno);
        } else {
            reass_q_.insert(it, reasm_queue_ent_t(msgbuf, seqno));
        }

        // Update the SACK bitmap for the newly received packet.
        pcb->sack_bitmap_bit_set(distance);

        PushInOrderMsgbufsToApp(pcb, app_buf, app_buf_len);
        return 0;
    }

   private:
    void PushInOrderMsgbufsToApp(swift::Pcb *pcb, void *app_buf,
                                 size_t *app_buf_len) {
        while (!reass_q_.empty() && reass_q_.front().seqno == pcb->rcv_nxt) {
            auto &front = reass_q_.front();
            auto *msgbuf = front.msgbuf;
            reass_q_.pop_front();

            if (cur_msg_train_head_ == nullptr) {
                DCHECK(msgbuf->is_first());
                cur_msg_train_head_ = msgbuf;
                cur_msg_train_tail_ = msgbuf;
            } else {
                cur_msg_train_tail_->set_next(msgbuf);
                cur_msg_train_tail_ = msgbuf;
            }

            if (cur_msg_train_tail_->is_last()) {
                // We have a complete message. Let's deliver it to the
                // application.
                auto *msgbuf_to_deliver = cur_msg_train_head_;
                while (msgbuf_to_deliver != nullptr) {
                    auto *pkt_addr = msgbuf_to_deliver->get_pkt_addr();
                    auto pkt_payload_len = msgbuf_to_deliver->get_frame_len() -
                                           sizeof(ethhdr) - sizeof(iphdr) -
                                           sizeof(udphdr) - sizeof(UcclPktHdr);

                    memcpy((uint8_t *)app_buf + app_buf_pos_, pkt_addr,
                           pkt_payload_len);
                    app_buf_pos_ += pkt_payload_len;

                    socket_->frame_pool_->push(
                        msgbuf_to_deliver->get_frame_offset());

                    msgbuf_to_deliver = msgbuf_to_deliver->next();
                }

                *app_buf_len = app_buf_pos_;

                LOG(WARNING) << "Received a complete message!";
                cur_msg_train_head_ = nullptr;
                cur_msg_train_tail_ = nullptr;
            }

            pcb->advance_rcv_nxt();

            pcb->sack_bitmap_shift_right_one();
        }
    }

    const uint32_t local_ip_;
    const uint16_t local_port_;
    const uint32_t remote_ip_;
    const uint16_t remote_port_;
    AFXDPSocket *socket_;
    std::deque<reasm_queue_ent_t> reass_q_;
    FrameBuf *cur_msg_train_head_;
    FrameBuf *cur_msg_train_tail_;
    // TODO: maintain connectionID to app_buf_pos mappings
    size_t app_buf_pos_;
};

/**
 * @class UcclFlow A flow is a connection between a local and a remote endpoint.
 * @brief Class to abstract the components and functionality of a single flow.
 * A flow is a bidirectional connection between two hosts, uniquely identified
 * by the 5-tuple: {SrcIP, DstIP, SrcPort, DstPort, Protocol}, Protocol is
 * always UDP.
 *
 * A flow is always associated with a single `Channel' object which serves as
 * the communication interface with the application to which the flow belongs.
 *
 * On normal operation, a flow is:
 *    - Receiving network packets from the NIC, which then converts to messages
 *      and enqueues to the `Channel', so that they reach the application.
 *    - Receiving messages from the application (via the `Channel'), which then
 *      converts to network packets and sends them out to the remote recipient.
 */
class UcclFlow {
   public:
    enum class State {
        kClosed,
        kSynSent,
        kSynReceived,
        kEstablished,
    };

    static constexpr char const *StateToString(State state) {
        switch (state) {
            case State::kClosed:
                return "CLOSED";
            case State::kSynSent:
                return "SYN_SENT";
            case State::kSynReceived:
                return "SYN_RECEIVED";
            case State::kEstablished:
                return "ESTABLISHED";
            default:
                LOG(FATAL) << "Unknown state";
                return "UNKNOWN";
        }
    }

    /**
     * @brief Construct a new flow.
     *
     * @param local_addr Local IP address.
     * @param local_port Local UDP port.
     * @param remote_addr Remote IP address.
     * @param remote_port Remote UDP port.
     * @param local_l2_addr Local L2 address.
     * @param remote_l2_addr Remote L2 address.
     * @param AFXDPSocket object for packet IOs.
     */
    UcclFlow(const uint32_t local_addr, const uint16_t local_port,
             const uint32_t remote_addr, const uint16_t remote_port,
             const uint8_t *local_l2_addr, const uint8_t *remote_l2_addr,
             AFXDPSocket *socket)
        : key_(local_addr, local_port, remote_addr, remote_port),
          socket_(CHECK_NOTNULL(socket)),
          state_(State::kEstablished),
          pcb_(),
          tx_tracking_(socket),
          rx_tracking_(local_addr, local_port, remote_addr, remote_port,
                       socket) {
        memcpy(local_l2_addr_, local_l2_addr, ETH_ALEN);
        memcpy(remote_l2_addr_, remote_l2_addr, ETH_ALEN);
    }
    ~UcclFlow() {}
    /**
     * @brief Operator to compare if two flows are equal.
     * @param other Other flow to compare to.
     * @return true if the flows are equal, false otherwise.
     */
    bool operator==(const UcclFlow &other) const { return key_ == other.key(); }

    /**
     * @brief Get the flow key.
     */
    const Key &key() const { return key_; }

    /**
     * @brief Get the current state of the flow.
     */
    State state() const { return state_; }

    std::string ToString() const {
        return Format(
            "%s [%s] <-> [%d]\n\t\t\t%s\n\t\t\t[TX Queue] Pending "
            "MsgBufs: "
            "%u",
            key_.ToString().c_str(), StateToString(state_), socket_->queue_id_,
            pcb_.ToString().c_str(), tx_tracking_.NumUnsentMsgbufs());
    }

    bool Match(const uint8_t *pkt_addr) const {
        const auto *ih =
            reinterpret_cast<const iphdr *>(pkt_addr + sizeof(ethhdr));
        const auto *udph = reinterpret_cast<const udphdr *>(
            pkt_addr + sizeof(ethhdr) + sizeof(iphdr));

        return (ih->saddr == key_.remote_addr && ih->daddr == key_.local_addr &&
                udph->source == key_.remote_port &&
                udph->dest == key_.local_port);
    }

    bool Match(const FrameBuf *tx_msgbuf) const {
        const auto flow_key = tx_msgbuf->get_flow();
        return (flow_key.local_addr == key_.local_addr &&
                flow_key.remote_addr == key_.remote_addr &&
                flow_key.local_port == key_.local_port &&
                flow_key.remote_port == key_.remote_port);
    }

    void InitiateHandshake() {
        DCHECK(state_ == State::kClosed);
        SendSyn(pcb_.get_snd_nxt());
        pcb_.rto_reset();
        state_ = State::kSynSent;
    }

    void ShutDown() {
        switch (state_) {
            case State::kClosed:
                break;
            case State::kSynSent:
                [[fallthrough]];
            case State::kSynReceived:
                [[fallthrough]];
            case State::kEstablished:
                pcb_.rto_disable();
                SendRst();
                state_ = State::kClosed;
                break;
            default:
                LOG(FATAL) << "Unknown state";
        }
    }

    /**
     * @brief Push the received packet onto the ingress queue of the flow.
     * Decrypts packet if required, stores the payload in the relevant channel
     * shared memory space, and if the message is ready for delivery notifies
     * the application.
     *
     * If this is a transport control packet (e.g., ACK) it only updates
     * transport-related parameters for the flow.
     *
     * @param packet Pointer to the allocated packet on the rx ring of the
     * driver
     */
    void InputPacket(FrameBuf *msgbuf, void *app_buf, size_t *app_buf_len) {
        // Parse the Uccl header of the packet.
        const size_t net_hdr_len =
            sizeof(ethhdr) + sizeof(iphdr) + sizeof(udphdr);
        uint8_t *pkt = msgbuf->get_pkt_addr();
        const auto *ucclh =
            reinterpret_cast<const UcclPktHdr *>(msgbuf + net_hdr_len);

        if (ucclh->magic.value() != UcclPktHdr::kMagic) {
            LOG(ERROR) << "Invalid Uccl header magic: " << ucclh->magic;
            return;
        }

        switch (ucclh->net_flags) {
            case UcclPktHdr::UcclFlags::kSyn:
                // SYN packet received. For this to be valid it has to be an
                // already established flow with this SYN being a
                // retransmission.
                if (state_ != State::kSynReceived && state_ != State::kClosed) {
                    LOG(ERROR) << "SYN packet received for flow in state: "
                               << static_cast<int>(state_);
                    return;
                }

                if (state_ == State::kClosed) {
                    // If the flow is in closed state, we need to send a SYN-ACK
                    // packetj and mark the flow as established.
                    pcb_.rcv_nxt = ucclh->seqno.value();
                    pcb_.advance_rcv_nxt();
                    SendSynAck(pcb_.get_snd_nxt());
                    state_ = State::kSynReceived;
                } else if (state_ == State::kSynReceived) {
                    // If the flow is in SYN-RECEIVED state, our SYN-ACK packet
                    // was lost. We need to retransmit it.
                    SendSynAck(pcb_.snd_una);
                }
                break;
            case UcclPktHdr::UcclFlags::kSynAck:
                // SYN-ACK packet received. For this to be valid it has to be an
                // already established flow with this SYN-ACK being a
                // retransmission.
                if (state_ != State::kSynSent &&
                    state_ != State::kEstablished) {
                    LOG(ERROR) << "SYN-ACK packet received for flow in state: "
                               << static_cast<int>(state_);
                    return;
                }

                if (ucclh->ackno.value() != pcb_.snd_nxt) {
                    LOG(ERROR) << "SYN-ACK packet received with invalid ackno: "
                               << ucclh->ackno << " snd_una: " << pcb_.snd_una
                               << " snd_nxt: " << pcb_.snd_nxt;
                    return;
                }

                if (state_ == State::kSynSent) {
                    pcb_.snd_una++;
                    pcb_.rcv_nxt = ucclh->seqno.value();
                    pcb_.advance_rcv_nxt();
                    pcb_.rto_maybe_reset();
                    // Mark the flow as established.
                    state_ = State::kEstablished;
                }
                // Send an ACK packet.
                SendAck();
                break;
            case UcclPktHdr::UcclFlags::kRst: {
                const auto seqno = ucclh->seqno.value();
                const auto expected_seqno = pcb_.rcv_nxt;
                if (swift::seqno_eq(seqno, expected_seqno)) {
                    // If the RST packet is in sequence, we can reset the flow.
                    state_ = State::kClosed;
                }
            } break;
            case UcclPktHdr::UcclFlags::kAck:
                // ACK packet, update the flow.
                // update_flow(ucclh);
                process_ack(ucclh);
                break;
            case UcclPktHdr::UcclFlags::kData:
                if (state_ != State::kEstablished) {
                    LOG(ERROR) << "Data packet received for flow in state: "
                               << static_cast<int>(state_);
                    return;
                }
                // Data packet, process the payload.
                const int consume_returncode =
                    rx_tracking_.Consume(&pcb_, msgbuf, app_buf, app_buf_len);
                if (consume_returncode == 0) SendAck();
                break;
        }
    }

    /**
     * @brief Push a Message from the application onto the egress queue of
     * the flow. Segments the message, and encrypts the packets, and adds all
     * packets onto the egress queue.
     * Caller is responsible for freeing the MsgBuf object.
     *
     * @param msg Pointer to the first message buffer on a train of buffers,
     * aggregating to a partial or a full Message.
     */
    void OutputMessage(FrameBuf *msg) {
        tx_tracking_.Append(msg);

        // TODO(ilias): We first need to check whether the cwnd is < 1, so that
        // we fallback to rate-based CC.

        // Calculate the effective window (in # of packets) to check whether we
        // can send more packets.
        TransmitPackets();
    }

    /**
     * @brief Periodically checks the state of the flow and performs necessary
     * actions.
     *
     * This method is called periodically to check the state of the flow, update
     * the RTO timer, retransmit unacknowledged messages, and potentially remove
     * the flow or notify the application about the connection state.
     *
     * @return Returns true if the flow should continue to be checked
     * periodically, false if the flow should be removed or closed.
     */
    bool PeriodicCheck() {
        // CLOSED state is terminal; the engine might remove the flow.
        if (state_ == State::kClosed) return false;

        if (pcb_.rto_disabled()) return true;

        pcb_.rto_advance();
        if (pcb_.max_rexmits_reached()) {
            if (state_ == State::kSynSent) {
                // Notify the application that the flow has not been
                // established.
                LOG(INFO) << "UcclFlow " << this << " failed to establish";
            }
            // TODO(ilias): Send RST packet.

            // Indicate removal of the flow.
            return false;
        }

        if (pcb_.rto_expired()) {
            // Retransmit the oldest unacknowledged message buffer.
            RTORetransmit();
        }

        return true;
    }

   private:
    void PrepareL2Header(uint8_t *pkt_addr) const {
        auto *eh = (ethhdr *)pkt_addr;
        memcpy(eh->h_source, local_l2_addr_, ETH_ALEN);
        memcpy(eh->h_dest, remote_l2_addr_, ETH_ALEN);
        eh->h_proto = htons(ETH_P_IP);
    }

    void PrepareL3Header(uint8_t *pkt_addr, uint32_t payload_bytes) const {
        auto *ipv4h = (iphdr *)(pkt_addr + sizeof(ethhdr));
        ipv4h->ihl = 5;
        ipv4h->version = 4;
        ipv4h->tos = 0x0;
        ipv4h->id = htons(0x1513);
        ipv4h->frag_off = htons(0);
        ipv4h->ttl = 64;
        ipv4h->protocol = IPPROTO_UDP;
        ipv4h->tot_len = htons(sizeof(iphdr) + sizeof(udphdr) + payload_bytes);
        ipv4h->saddr = htonl(key_.local_addr);
        ipv4h->daddr = htonl(key_.remote_addr);
        ipv4h->check = 0;
    }

    void PrepareL4Header(uint8_t *pkt_addr, uint32_t payload_bytes) const {
        auto *udph = (udphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
        udph->source = htons(key_.local_port);
        udph->dest = htons(key_.remote_port);
        udph->len = htons(sizeof(udphdr) + payload_bytes);
        udph->check = htons(0);
        // TODO: Calculate the UDP checksum.
    }

    void PrepareUcclHdr(uint8_t *pkt_addr, uint32_t seqno,
                        const UcclPktHdr::UcclFlags &net_flags,
                        uint8_t msg_flags = 0) const {
        auto *ucclh = (UcclPktHdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr) +
                                     sizeof(udphdr));
        ucclh->magic = be16_t(UcclPktHdr::kMagic);
        ucclh->net_flags = net_flags;
        ucclh->msg_flags = msg_flags;
        ucclh->seqno = be32_t(seqno);
        ucclh->ackno = be32_t(pcb_.ackno());

        for (size_t i = 0; i < sizeof(UcclPktHdr::sack_bitmap) /
                                   sizeof(UcclPktHdr::sack_bitmap[0]);
             ++i) {
            ucclh->sack_bitmap[i] = be64_t(pcb_.sack_bitmap[i]);
        }
        ucclh->sack_bitmap_count = be16_t(pcb_.sack_bitmap_count);

        ucclh->timestamp1 = be64_t(0);
    }

    void SendControlPacket(uint32_t seqno,
                           const UcclPktHdr::UcclFlags &flags) const {
        auto frame_offset = socket_->frame_pool_->pop();
        uint8_t *pkt_addr = (uint8_t *)socket_->umem_buffer_ + frame_offset;

        const size_t kControlPayloadBytes = sizeof(UcclPktHdr);
        PrepareL2Header(pkt_addr);
        PrepareL3Header(pkt_addr, kControlPayloadBytes);
        PrepareL4Header(pkt_addr, kControlPayloadBytes);
        PrepareUcclHdr(pkt_addr, seqno, flags);

        // Send the packet.
        socket_->send_packet(
            {frame_offset, sizeof(ethhdr) + sizeof(iphdr) + sizeof(ethhdr) +
                               kControlPayloadBytes},
            /*free_frame=*/false);
    }

    void SendSyn(uint32_t seqno) const {
        SendControlPacket(seqno, UcclPktHdr::UcclFlags::kSyn);
    }

    void SendSynAck(uint32_t seqno) const {
        SendControlPacket(
            seqno, UcclPktHdr::UcclFlags::kSyn | UcclPktHdr::UcclFlags::kAck);
    }

    void SendAck() const {
        SendControlPacket(pcb_.seqno(), UcclPktHdr::UcclFlags::kAck);
    }

    void SendRst() const {
        SendControlPacket(pcb_.seqno(), UcclPktHdr::UcclFlags::kRst);
    }

    /**
     * @brief This helper method prepares a network packet that carries the data
     * of a particular `FrameBuf'.
     *
     * @tparam copy_mode Copy mode of the packet. Either kMemCopy or kZeroCopy.
     * @param buf Pointer to the message buffer to be sent.
     * @param packet Pointer to an allocated packet.
     * @param seqno Sequence number of the packet.
     */
    void PrepareDataPacket(FrameBuf *msg_buf, uint32_t seqno) const {
        // Header length after before the payload.
        const size_t hdr_length = sizeof(ethhdr) + sizeof(iphdr) +
                                  sizeof(ethhdr) + sizeof(UcclPktHdr);
        uint32_t frame_len = msg_buf->get_frame_len();
        CHECK_LE(frame_len, AFXDP_MTU);
        uint8_t *pkt_addr = msg_buf->get_pkt_addr();

        // Prepare network headers.
        PrepareL2Header(pkt_addr);
        PrepareL3Header(pkt_addr, frame_len - hdr_length);
        PrepareL4Header(pkt_addr, frame_len - hdr_length);

        // Prepare the Uccl-specific header.
        auto *ucclh = reinterpret_cast<UcclPktHdr *>(
            pkt_addr + sizeof(ethhdr) + sizeof(iphdr) + sizeof(ethhdr));
        ucclh->magic = be16_t(UcclPktHdr::kMagic);
        ucclh->net_flags = UcclPktHdr::UcclFlags::kData;
        ucclh->ackno = be32_t(UINT32_MAX);
        // This fills the FrameBuf.flags into the outgoing packet
        // UcclPktHdr.msg_flags.
        ucclh->msg_flags = msg_buf->msg_flags();

        ucclh->seqno = be32_t(seqno);
        ucclh->timestamp1 = be64_t(0);
    }

    void FastRetransmit() {
        // Retransmit the oldest unacknowledged message buffer.
        auto *msg_buf = tx_tracking_.GetOldestUnackedMsgBuf();
        PrepareDataPacket(msg_buf, pcb_.snd_una);
        socket_->send_packet(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()},
            /*free_frame=*/false);
        pcb_.rto_reset();
        pcb_.fast_rexmits++;
        LOG(INFO) << "Fast retransmitting packet " << pcb_.snd_una;
    }

    void RTORetransmit() {
        if (state_ == State::kEstablished) {
            LOG(INFO) << "RTO retransmitting data packet " << pcb_.snd_una;
            auto *msg_buf = tx_tracking_.GetOldestUnackedMsgBuf();
            PrepareDataPacket(msg_buf, pcb_.snd_una);
            socket_->send_packet(
                {msg_buf->get_frame_offset(), msg_buf->get_frame_len()},
                /*free_frame=*/false);
        } else if (state_ == State::kSynReceived) {
            SendSynAck(pcb_.snd_una);
        } else if (state_ == State::kSynSent) {
            LOG(INFO) << "RTO retransmitting SYN packet " << pcb_.snd_una;
            // Retransmit the SYN packet.
            SendSyn(pcb_.snd_una);
        }
        pcb_.rto_reset();
        pcb_.rto_rexmits++;
    }

    /**
     * @brief Helper function to transmit a number of packets from the queue of
     * pending TX data.
     */
    void TransmitPackets() {
        auto remaining_packets =
            std::min(pcb_.effective_wnd(), tx_tracking_.NumUnsentMsgbufs());
        if (remaining_packets == 0) return;

        VLOG(3) << "TransmitPackets: remaining_packets " << remaining_packets;

        std::vector<AFXDPSocket::frame_desc> frames;

        // Prepare the packets.
        for (uint16_t i = 0; i < remaining_packets; i++) {
            auto msg_buf_opt = tx_tracking_.GetAndUpdateOldestUnsent();
            if (!msg_buf_opt.has_value()) break;
            auto *msg_buf = msg_buf_opt.value();
            PrepareDataPacket(msg_buf, pcb_.get_snd_nxt());
            frames.emplace_back(AFXDPSocket::frame_desc{
                msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
        }

        // TX.
        socket_->send_packets(frames, /*free_frames=*/false);

        if (pcb_.rto_disabled()) pcb_.rto_enable();
    }

    void process_ack(const UcclPktHdr *ucclh) {
        auto ackno = ucclh->ackno.value();
        if (swift::seqno_lt(ackno, pcb_.snd_una)) {
            return;
        } else if (swift::seqno_eq(ackno, pcb_.snd_una)) {
            // Duplicate ACK.
            pcb_.duplicate_acks++;
            // Update the number of out-of-order acknowledgements.
            pcb_.snd_ooo_acks = ucclh->sack_bitmap_count.value();

            if (pcb_.duplicate_acks < swift::Pcb::kRexmitThreshold) {
                // We have not reached the threshold yet, so we do not do
                // anything.
            } else if (pcb_.duplicate_acks == swift::Pcb::kRexmitThreshold) {
                // Fast retransmit.
                FastRetransmit();
            } else {
                // We have already done the fast retransmit, so we are now in
                // the fast recovery phase. We need to send a new packet for
                // every ACK we get.
                auto sack_bitmap_count = ucclh->sack_bitmap_count.value();
                // First we check the SACK bitmap to see if there are more
                // undelivered packets. In fast recovery mode we get after a
                // fast retransmit, and for every new ACKnowledgement we get, we
                // send a new packet. Up until we get the first new
                // acknowledgement, for the next in-order packet, the SACK
                // bitmap will likely keep expanding. In order to avoid
                // retransmitting multiple times other missing packets in the
                // bitmap, we skip holes: we use the number of duplicate ACKs to
                // skip previous holes.
                auto *msgbuf = tx_tracking_.GetOldestUnackedMsgBuf();
                size_t holes_to_skip =
                    pcb_.duplicate_acks - swift::Pcb::kRexmitThreshold;
                size_t index = 0;
                while (sack_bitmap_count) {
                    constexpr size_t sack_bitmap_bucket_size =
                        sizeof(ucclh->sack_bitmap[0]);
                    constexpr size_t sack_bitmap_max_bucket_idx =
                        sizeof(ucclh->sack_bitmap) /
                            sizeof(ucclh->sack_bitmap[0]) -
                        1;
                    const size_t sack_bitmap_bucket_idx =
                        sack_bitmap_max_bucket_idx -
                        index / sack_bitmap_bucket_size;
                    const size_t sack_bitmap_idx_in_bucket =
                        index % sack_bitmap_bucket_size;
                    auto sack_bitmap =
                        ucclh->sack_bitmap[sack_bitmap_bucket_idx].value();
                    if ((sack_bitmap & (1ULL << sack_bitmap_idx_in_bucket)) ==
                        0) {
                        // We found a missing packet.
                        // We skip holes in the SACK bitmap that have already
                        // been retransmitted.
                        if (holes_to_skip-- == 0) {
                            auto seqno = pcb_.snd_una + index;
                            PrepareDataPacket(msgbuf, seqno);
                            socket_->send_packet({msgbuf->get_frame_offset(),
                                                  msgbuf->get_frame_len()},
                                                 /*free_frame=*/false);
                            pcb_.rto_reset();
                            return;
                        }
                    } else {
                        sack_bitmap_count--;
                    }
                    index++;
                    msgbuf = msgbuf->next();
                }
                // There is no other missing segment to retransmit, so we could
                // send new packets.
            }
        } else if (swift::seqno_gt(ackno, pcb_.snd_nxt)) {
            LOG(ERROR) << "ACK received for untransmitted data.";
        } else {
            // This is a valid ACK, acknowledging new data.
            size_t num_acked_packets = ackno - pcb_.snd_una;
            if (state_ == State::kSynReceived) {
                state_ = State::kEstablished;
                num_acked_packets--;
            }

            tx_tracking_.ReceiveAcks(num_acked_packets);

            pcb_.snd_una = ackno;
            pcb_.duplicate_acks = 0;
            pcb_.snd_ooo_acks = 0;
            pcb_.rto_rexmits = 0;
            pcb_.rto_maybe_reset();
        }

        TransmitPackets();
    }

    const Key key_;

    // A flow is identified by the 5-tuple (Proto is always UDP).
    uint8_t local_l2_addr_[ETH_ALEN];
    uint8_t remote_l2_addr_[ETH_ALEN];

    AFXDPSocket *socket_;
    // UcclFlow state.
    State state_;
    // Swift CC protocol control block.
    swift::Pcb pcb_;
    TXTracking tx_tracking_;
    RXTracking rx_tracking_;
};

/**
 * @brief Class `UcclEngine' abstracts the main Uccl engine. This engine
 * contains all the functionality need to be run by the stack's threads.
 */
class UcclEngine {
   public:
    // Slow timer (periodic processing) interval in microseconds.
    const size_t kSlowTimerIntervalUs = 2000;  // 2ms
    const uint32_t RECV_BATCH_SIZE = 32;
    UcclEngine() = delete;
    UcclEngine(UcclEngine const &) = delete;

    /**
     * @brief Construct a new UcclEngine object.
     *
     * @param pmd_port      Pointer to the PMD port to be used by the engine.
     * The PMD port must be initialized (i.e., call InitDriver()).
     * @param rx_queue_id   RX queue index to be used by the engine.
     * @param tx_queue_id   TX queue index to be used by the engine. The TXRing
     *                      associated should be initialized with a packet pool.
     * @param channels      (optional) Uccl channels the engine will be
     *                      responsible for (if any).
     */
    UcclEngine(int queue_id, int num_frames, Channel *channel,
               const uint32_t local_addr, const uint16_t local_port,
               const uint32_t remote_addr, const uint16_t remote_port,
               const uint8_t *local_l2_addr, const uint8_t *remote_l2_addr)
        : socket_(AFXDPFactory::CreateSocket(queue_id, num_frames)),
          channel_(channel),
          last_periodic_timestamp_(std::chrono::high_resolution_clock::now()),
          periodic_ticks_(0) {
        flow_ = std::make_unique<UcclFlow>(local_addr, local_port, remote_addr,
                                           remote_port, local_l2_addr,
                                           remote_l2_addr, socket_);
    }

    /**
     * @brief This is the main event cycle of the Uccl engine.
     * It is called repeatedly by the main thread of the Uccl engine.
     * On each cycle, the engine processes incoming packets in the RX queue and
     * enqueued messages in all channels that it is responsible for.
     * This method is not thread-safe.
     *
     * @param now The current TSC.
     */
    void Run() {
        bool has_rx_work = false;
        ChannelMsg rx_work;
        bool has_tx_work = false;
        ChannelMsg tx_work;
        FrameBuf *tx_msgbuf_start = nullptr;
        FrameBuf *tx_msgbuf_cur = nullptr;
        while (true) {
            // Calculate the time elapsed since the last periodic processing.
            auto now = std::chrono::high_resolution_clock::now();
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    now - last_periodic_timestamp_)
                    .count();

            if (elapsed >= kSlowTimerIntervalUs) {
                // Perform periodic processing.
                PeriodicProcess();
                last_periodic_timestamp_ = now;
            }

            // TODO: where to run process_rx_pkt for ACK packets even without
            // user-supplied buffer?
            if (has_rx_work) {
                auto frames = socket_->recv_packets(RECV_BATCH_SIZE);
                VLOG(3) << "Rx recv_packets" << frames.size();
                for (auto &frame : frames) {
                    auto msgbuf = FrameBuf::Create(frame.frame_offset,
                                                   socket_->umem_buffer_,
                                                   frame.frame_len);
                    // TODO: how to guarantee before receiving packets, the
                    // application already post the application buffer to the
                    // engine?
                    process_rx_pkt(msgbuf, rx_work.data, rx_work.len_ptr);
                }
            } else {
                if (jring_sc_dequeue_bulk(channel_->rx_ring_, &rx_work, 1,
                                          nullptr) == 1) {
                    VLOG(3) << "Rx jring dequeue";
                    has_rx_work = true;
                }
            }

            if (has_tx_work) {
                VLOG(3) << "Tx process_tx_pkt";
                if (tx_msgbuf_start != nullptr) {
                    process_tx_pkt(tx_msgbuf_start);
                    tx_msgbuf_start = tx_msgbuf_start->next();
                } else {
                    has_tx_work = false;
                }
            } else {
                if (jring_sc_dequeue_bulk(channel_->tx_ring_, &tx_work, 1,
                                          nullptr) == 1) {
                    VLOG(3) << "Tx jring dequeue";
                    has_tx_work = true;
                    auto *app_buf = tx_work.data;
                    auto remaining_bytes = *tx_work.len_ptr;
                    const auto net_hdr_len = sizeof(ethhdr) + sizeof(iphdr) +
                                             sizeof(udphdr) +
                                             sizeof(UcclPktHdr);
                    //  Deserializing the message into MTU-sized frames.
                    while (remaining_bytes > 0) {
                        auto payload_len =
                            std::min(remaining_bytes, (size_t)AFXDP_MTU);
                        auto frame_offset = socket_->frame_pool_->pop();
                        auto *msgbuf = FrameBuf::Create(
                            frame_offset, socket_->umem_buffer_,
                            payload_len + net_hdr_len);
                        auto pkt_payload_addr =
                            msgbuf->get_pkt_addr() + net_hdr_len;
                        memcpy(pkt_payload_addr, app_buf, payload_len);
                        if (tx_msgbuf_start == nullptr) {
                            msgbuf->mark_first();
                            tx_msgbuf_start = msgbuf;
                            tx_msgbuf_cur = msgbuf;
                        } else {
                            tx_msgbuf_cur->set_next(msgbuf);
                            tx_msgbuf_cur = msgbuf;
                        }
                        remaining_bytes -= payload_len;
                        app_buf += payload_len;
                        if (remaining_bytes == 0) {
                            msgbuf->mark_last();
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Method to perform periodic processing. This is called by the main
     * engine cycle (see method `Run`).
     *
     * @param now The current TSC.
     */
    void PeriodicProcess() {
        // Advance the periodic ticks counter.
        ++periodic_ticks_;
        HandleRTO();
        DumpStatus();
        ProcessControlRequests();
    }

   protected:
    void DumpStatus() {
        std::string s;
        s += "[Uccl Engine Status]";
        // TODO: Add more status information.
        s += "\n";
        LOG(INFO) << s;
    }

    /**
     * @brief This method polls active channels for all control plane requests
     * and processes them. It is called periodically.
     */
    void ProcessControlRequests() {
        // TODO: maintain pending_requests_; right now, we have done it in main
        // run()
    }

    /**
     * @brief Iterate throught the list of flows, check and handle RTOs.
     */
    void HandleRTO() {
        // TODO: maintain active_flows_map_
        auto is_active_flow = flow_->PeriodicCheck();
        DCHECK(is_active_flow);
    }

    /**
     * @brief Process an incoming packet.
     *
     * @param pkt Pointer to the packet.
     * @param now TSC timestamp.
     */
    void process_rx_pkt(FrameBuf *msgbuf, void *app_buf, size_t *app_buf_len) {
        // Sanity ethernet header check.
        if (msgbuf->get_frame_len() < sizeof(ethhdr)) [[unlikely]]
            return;
        auto *pkt_addr = msgbuf->get_pkt_addr();
        auto *eh = reinterpret_cast<ethhdr *>(pkt_addr);
        switch (ntohs(eh->h_proto)) {
            [[likely]] case ETH_P_IP:
                process_rx_ipv4(msgbuf, app_buf, app_buf_len);
                break;
            case ETH_P_IPV6:
                LOG(ERROR) << "IPv6 not supported yet.";
                break;
            case ETH_P_ARP:
                LOG(ERROR) << "ARP not supported yet.";
                break;
            default:
                break;
        }
    }

    void process_rx_ipv4(FrameBuf *msgbuf, void *app_buf, size_t *app_buf_len) {
        auto frame_len = msgbuf->get_frame_len();
        // Sanity ipv4 header check.
        if (frame_len < sizeof(ethhdr) + sizeof(iphdr)) [[unlikely]]
            return;

        auto pkt_addr = msgbuf->get_pkt_addr();
        auto *eh = reinterpret_cast<ethhdr *>(pkt_addr);
        auto *ipv4h = reinterpret_cast<iphdr *>(pkt_addr + sizeof(ethhdr));
        auto *udph = reinterpret_cast<udphdr *>(pkt_addr + sizeof(ethhdr) +
                                                sizeof(iphdr));

        const Key pkt_key(ipv4h->daddr, udph->dest, ipv4h->saddr, udph->source);
        // Check ivp4 header length.
        if (frame_len != sizeof(ethhdr) + ntohs(ipv4h->tot_len)) [[unlikely]] {
            LOG(WARNING) << "IPv4 packet length mismatch (expected: "
                         << ntohs(ipv4h->tot_len) << ", actual: " << frame_len
                         << ")";
            return;
        }

        switch (ipv4h->protocol) {
            [[likely]] case IPPROTO_UDP:
                flow_->InputPacket(msgbuf, app_buf, app_buf_len);
                break;
            case IPPROTO_ICMP:
                LOG(WARNING) << "ICMP not supported yet.";
                break;
            default:
                LOG(WARNING) << "Unsupported IP protocol: "
                             << static_cast<uint32_t>(ipv4h->protocol);
                break;
        }
    }

    /**
     * Process a message enqueued from an application to a channel.
     * @param channel A pointer to the channel that the message was enqueued to.
     * @param msg     A pointer to the `MsgBuf` containing the first buffer of
     * the message.
     */
    void process_tx_pkt(FrameBuf *msg) {
        // TODO: lookup the msg five-tuple in an active_flows_map_
        flow_->OutputMessage(msg);
    }

   private:
    // AFXDP socket used for send/recv packets.
    AFXDPSocket *socket_;
    // For now, we just assume a single flow.
    std::unique_ptr<UcclFlow> flow_;
    // Control plan channel with Endpoint.
    Channel *channel_;
    // Timestamp of last periodic process execution.
    std::chrono::time_point<std::chrono::high_resolution_clock>
        last_periodic_timestamp_;
    // Clock ticks for the slow timer.
    uint64_t periodic_ticks_{0};
};

}  // namespace uccl

namespace std {

template <>
struct hash<uccl::UcclFlow> {
    size_t operator()(const uccl::UcclFlow &flow) const {
        const auto &key = flow.key();
        return std::hash<std::string_view>{}(
            {reinterpret_cast<const char *>(&key), sizeof(key)});
    }
};

}  // namespace std