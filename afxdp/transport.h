#pragma once

#include <glog/logging.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>

#include <bitset>
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

class Channel {
    constexpr static uint32_t kChannelSize = 1024;

   public:
    struct Msg {
        enum Op : uint8_t {
            kTx = 0,
            kTxComp = 1,
            kRx = 2,
            kRxComp = 3,
        };
        Op opcode;
        void *data;
        size_t *len_ptr;
        ConnectionID connection_id;
    };
    static_assert(sizeof(Msg) % 4 == 0, "channelMsg must be 32-bit aligned");

    Channel() {
        tx_ring_ = create_ring(sizeof(Msg), kChannelSize);
        tx_comp_ring_ = create_ring(sizeof(Msg), kChannelSize);
        rx_ring_ = create_ring(sizeof(Msg), kChannelSize);
        rx_comp_ring_ = create_ring(sizeof(Msg), kChannelSize);
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

/**
 * @class Endpoint
 * @brief application-facing interface, communicating with `UcclEngine' through
 * `Channel'. Each connection is identified by a unique connection_id, and uses
 * multiple src+dst port combinations to leverage multiple paths. Under the
 * hood, we leverage TCP to boostrap our connections. We do not consider
 * multi-tenancy for now, assuming this endpoint exclusively uses the NIC and
 * its all queues.
 */
class Endpoint {
    constexpr static uint16_t kBootstrapPort = 40000;
    Channel *channel_;

   public:
    Endpoint(Channel *channel) : channel_(channel) {}
    ~Endpoint() {}

    // Connecting to a remote address.
    ConnectionID Connect(uint32_t remote_ip) {
        // TODO(yang): Using TCP to negotiate a ConnectionID.
        return 0xdeadbeaf;
    }

    ConnectionID Accept() {
        // TODO(yang): Using TCP to negotiate a ConnectionID.
        return 0xdeadbeaf;
    }

    // Sending the data by leveraging multiple port combinations.
    bool Send(ConnectionID connection_id, const void *data, size_t len) {
        Channel::Msg msg = {
            .opcode = Channel::Msg::Op::kTx,
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
        Channel::Msg msg = {
            .opcode = Channel::Msg::Op::kRx,
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
 * Uccl Packet Header just after UDP header.
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

static const size_t kNetHdrLen =
    sizeof(ethhdr) + sizeof(iphdr) + sizeof(udphdr);
static const size_t kUcclHdrLen = sizeof(UcclPktHdr);

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

class TXTracking {
   public:
    TXTracking() = delete;
    TXTracking(AFXDPSocket *socket, Channel *channel)
        : socket_(socket),
          channel_(channel),
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
            // Free transmitted frames that are acked
            socket_->frame_pool_->push(msgbuf->get_frame_offset());
            num_tracked_msgbufs_--;
            num_acked_pkts--;

            if (msgbuf->is_last()) {
                // Tx a full message; wakeup app thread waiting on endpoint.
                LOG(WARNING) << "Transmitted a complete message";
                Channel::Msg tx_work;
                while (jring_sp_enqueue_bulk(channel_->tx_comp_ring_, &tx_work,
                                             1, nullptr) != 1) {
                    // do nothing
                }
            }
        }
    }

    void Append(FrameBuf *msgbuf_head, FrameBuf *msgbuf_tail,
                uint32_t num_frames) {
        DCHECK(msgbuf_head->is_first());
        // Append the message at the end of the chain of buffers, if any.
        if (last_msgbuf_ == nullptr) {
            // This is the first pending message buffer in the flow.
            DCHECK(oldest_unsent_msgbuf_ == nullptr);
            last_msgbuf_ = msgbuf_tail;
            oldest_unsent_msgbuf_ = msgbuf_head;
            oldest_unacked_msgbuf_ = msgbuf_head;
        } else {
            // This is not the first message buffer in the flow.
            DCHECK(oldest_unacked_msgbuf_ != nullptr);
            // Let's enqueue the new message buffer at the end of the chain.
            last_msgbuf_->set_next(msgbuf_head);
            // Update the last buffer pointer to point to the current buffer.
            last_msgbuf_ = msgbuf_tail;
            if (oldest_unsent_msgbuf_ == nullptr)
                oldest_unsent_msgbuf_ = msgbuf_head;
        }

        num_unsent_msgbufs_ += num_frames;
        num_tracked_msgbufs_ += num_frames;
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
    Channel *channel_;

    /**
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
               uint16_t remote_port, AFXDPSocket *socket, Channel *channel)
        : local_ip_(local_ip),
          local_port_(local_port),
          remote_ip_(remote_ip),
          remote_port_(remote_port),
          socket_(socket),
          channel_(channel),
          cur_msg_train_head_(nullptr),
          cur_msg_train_tail_(nullptr),
          msg_ready_(false),
          app_buf_(nullptr),
          app_buf_len_(nullptr) {}

    // If we fail to allocate in the SHM channel, return -1.
    int Consume(swift::Pcb *pcb, FrameBuf *msgbuf) {
        uint8_t *pkt_addr = msgbuf->get_pkt_addr();
        auto frame_len = msgbuf->get_frame_len();
        const auto *ucclh =
            reinterpret_cast<const UcclPktHdr *>(pkt_addr + kNetHdrLen);
        const auto *payload = reinterpret_cast<const UcclPktHdr *>(
            pkt_addr + kNetHdrLen + kUcclHdrLen);
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
        const size_t payload_len = frame_len - kNetHdrLen - kUcclHdrLen;
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

        PushInOrderMsgbufsToApp(pcb);
        return 0;
    }

   private:
    void PushInOrderMsgbufsToApp(swift::Pcb *pcb) {
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
                msg_ready_ = true;
                TryCopyMsgbufToAppBuf(nullptr, nullptr);
            }

            pcb->advance_rcv_nxt();

            pcb->sack_bitmap_shift_right_one();
        }
    }

   public:
    /**
     * Either the app supplies the app buffer or the engine receives a full msg.
     * It returns true if successfully copying the msgbuf to the app buffer;
     * otherwise false.
     */
    void TryCopyMsgbufToAppBuf(void *app_buf, size_t *app_buf_len) {
        // Either both app_buf and app_buf_len are nullptr or both are not.
        if (app_buf_ == nullptr && app_buf_len_ == nullptr) {
            app_buf_ = app_buf;
            app_buf_len_ = app_buf_len;
        } else {
            DCHECK(app_buf_ && app_buf_len_);
        }

        if (!(msg_ready_ && app_buf_ && app_buf_len_)) return;

        // We have a complete message. Let's deliver it to the app.
        auto *msgbuf_to_deliver = cur_msg_train_head_;
        size_t pos = 0;
        while (msgbuf_to_deliver != nullptr) {
            auto *pkt_addr = msgbuf_to_deliver->get_pkt_addr();
            auto *payload_addr = pkt_addr + kNetHdrLen + kUcclHdrLen;
            auto payload_len =
                msgbuf_to_deliver->get_frame_len() - kNetHdrLen - kUcclHdrLen;

            memcpy((uint8_t *)app_buf_ + pos, payload_addr, payload_len);
            pos += payload_len;

            // Free received frames that have been copied to app buf.
            socket_->frame_pool_->push(msgbuf_to_deliver->get_frame_offset());

            msgbuf_to_deliver = msgbuf_to_deliver->next();
        }
        *app_buf_len_ = pos;

        // Wakeup app thread waiting on endpoint.
        Channel::Msg rx_work;
        while (jring_sp_enqueue_bulk(channel_->rx_comp_ring_, &rx_work, 1,
                                     nullptr) != 1) {
            // do nothing
        }

        LOG(WARNING) << "Received a complete message " << pos << " bytes";

        cur_msg_train_head_ = nullptr;
        cur_msg_train_tail_ = nullptr;

        // Reset the message ready flag for the next message.
        msg_ready_ = false;
        app_buf_ = nullptr;
        app_buf_len_ = nullptr;
    }

   private:
    const uint32_t local_ip_;
    const uint16_t local_port_;
    const uint32_t remote_ip_;
    const uint16_t remote_port_;
    AFXDPSocket *socket_;
    Channel *channel_;

    std::deque<reasm_queue_ent_t> reass_q_;
    FrameBuf *cur_msg_train_head_;
    FrameBuf *cur_msg_train_tail_;
    bool msg_ready_;
    void *app_buf_;
    size_t *app_buf_len_;
};

/**
 * @class UcclFlow, a connection between a local and a remote endpoint.
 * @brief Class to abstract the components and functionality of a single flow.
 * A flow is a bidirectional connection between two hosts, uniquely identified
 * by a TCP-negotiated `ConnectionID', Protocol is always UDP.
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
     * @param ConnectionID Connection ID for the flow.
     */
    UcclFlow(const uint32_t local_addr, const uint16_t local_port,
             const uint32_t remote_addr, const uint16_t remote_port,
             const uint8_t *local_l2_addr, const uint8_t *remote_l2_addr,
             AFXDPSocket *socket, Channel *channel, ConnectionID connection_id)
        : local_addr_(local_addr),
          local_port_(local_port),
          remote_addr_(remote_addr),
          remote_port_(remote_port),
          socket_(CHECK_NOTNULL(socket)),
          channel_(channel),
          connection_id_(connection_id),
          pcb_(),
          tx_tracking_(socket, channel),
          rx_tracking_(local_addr, local_port, remote_addr, remote_port, socket,
                       channel) {
        memcpy(local_l2_addr_, local_l2_addr, ETH_ALEN);
        memcpy(remote_l2_addr_, remote_l2_addr, ETH_ALEN);
    }
    ~UcclFlow() {}

    std::string ToString() const {
        return Format(
            "%x [queue %d] <-> %x\n\t\t\t%s\n\t\t\t[TX Queue] Pending "
            "MsgBufs: "
            "%u",
            local_addr_, socket_->queue_id_, remote_addr_,
            pcb_.ToString().c_str(), tx_tracking_.NumUnsentMsgbufs());
    }

    void Shutdown() { pcb_.rto_disable(); }

    /**
     * @brief Push the received packet onto the ingress queue of the flow.
     * Decrypts packet if required, stores the payload in the relevant channel
     * shared memory space, and if the message is ready for delivery notifies
     * the application.
     *
     * If this is a transport control packet (e.g., ACK) it only updates
     * transport-related parameters for the flow.
     *
     * @param msgbuf Pointer to the allocated packet
     * @param app_buf Pointer to the application receiving buffer
     * @param app_buf_len Pointer to the application buffer length
     */
    void RxMsg(FrameBuf *msgbuf) {
        // Parse the Uccl header of the packet.
        uint8_t *pkt_addr = msgbuf->get_pkt_addr();
        const auto *ucclh =
            reinterpret_cast<const UcclPktHdr *>(pkt_addr + kNetHdrLen);

        if (ucclh->magic.value() != UcclPktHdr::kMagic) {
            LOG(ERROR) << "Invalid Uccl header magic: " << ucclh->magic;
            return;
        }

        switch (ucclh->net_flags) {
            case UcclPktHdr::UcclFlags::kSyn:
            case UcclPktHdr::UcclFlags::kSynAck:
            case UcclPktHdr::UcclFlags::kRst:
                LOG(ERROR) << "Unsupported UcclFlags: "
                           << std::bitset<8>((uint8_t)ucclh->net_flags);
                break;
            case UcclPktHdr::UcclFlags::kAck:
                // ACK packet, update the flow.
                // update_flow(ucclh);
                process_ack(ucclh);
                break;
            case UcclPktHdr::UcclFlags::kData:
                // Data packet, process the payload.
                const int consume_returncode =
                    rx_tracking_.Consume(&pcb_, msgbuf);
                VLOG(3) << "Consumed packet with return code: "
                        << consume_returncode;
                if (consume_returncode == 0) SendAck();
                break;
        }
    }

    void rx_supply_app_buf(void *app_buf, size_t *app_buf_len) {
        rx_tracking_.TryCopyMsgbufToAppBuf(app_buf, app_buf_len);
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
    void TxMsg(FrameBuf *msg_head, FrameBuf *msg_tail, uint32_t num_frames) {
        tx_tracking_.Append(msg_head, msg_tail, num_frames);

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
        if (pcb_.rto_disabled()) return true;

        pcb_.rto_advance();
        if (pcb_.max_rexmits_reached()) {
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
        ipv4h->saddr = htonl(local_addr_);
        ipv4h->daddr = htonl(remote_addr_);
        ipv4h->check = 0;
        // AWS would block traffic if ipv4 checksum is not calculated.
        ipv4h->check = ipv4_checksum(ipv4h, sizeof(iphdr));
    }

    void PrepareL4Header(uint8_t *pkt_addr, uint32_t payload_bytes) const {
        auto *udph = (udphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
        udph->source = htons(local_port_);
        udph->dest = htons(remote_port_);
        udph->len = htons(sizeof(udphdr) + payload_bytes);
        udph->check = htons(0);
        // TODO(yang): Calculate the UDP checksum.
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

        const size_t kControlPayloadBytes = kUcclHdrLen;
        PrepareL2Header(pkt_addr);
        PrepareL3Header(pkt_addr, kControlPayloadBytes);
        PrepareL4Header(pkt_addr, kControlPayloadBytes);
        PrepareUcclHdr(pkt_addr, seqno, flags);

        // Let AFXDPSocket::pull_complete_queue() free control frames.
        FrameBuf::mark_txpulltime_free(frame_offset, socket_->umem_buffer_);
        // Send the packet.
        socket_->send_packet({frame_offset, sizeof(ethhdr) + sizeof(iphdr) +
                                                sizeof(ethhdr) +
                                                kControlPayloadBytes});
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
        uint32_t frame_len = msg_buf->get_frame_len();
        CHECK_LE(frame_len, AFXDP_MTU);
        uint8_t *pkt_addr = msg_buf->get_pkt_addr();

        // Prepare network headers.
        PrepareL2Header(pkt_addr);
        PrepareL3Header(pkt_addr, frame_len - kNetHdrLen);
        PrepareL4Header(pkt_addr, frame_len - kNetHdrLen);

        // Prepare the Uccl-specific header.
        auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);
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
        msg_buf->mark_not_txpulltime_free();
        socket_->send_packet(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
        pcb_.rto_reset();
        pcb_.fast_rexmits++;
        LOG(INFO) << "Fast retransmitting packet " << pcb_.snd_una;
    }

    void RTORetransmit() {
        LOG(INFO) << "RTO retransmitting data packet " << pcb_.snd_una;
        auto *msg_buf = tx_tracking_.GetOldestUnackedMsgBuf();
        PrepareDataPacket(msg_buf, pcb_.snd_una);
        msg_buf->mark_not_txpulltime_free();
        socket_->send_packet(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
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
            auto seqno = pcb_.get_snd_nxt();
            PrepareDataPacket(msg_buf, seqno);
            VLOG(3) << "TransmitPackets: transmit idx " << i << " seq "
                    << seqno;
            msg_buf->mark_not_txpulltime_free();
            frames.emplace_back(AFXDPSocket::frame_desc{
                msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
        }

        // TX.
        socket_->send_packets(frames);

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
                            msgbuf->mark_not_txpulltime_free();
                            socket_->send_packet({msgbuf->get_frame_offset(),
                                                  msgbuf->get_frame_len()});
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
            tx_tracking_.ReceiveAcks(num_acked_packets);

            pcb_.snd_una = ackno;
            pcb_.duplicate_acks = 0;
            pcb_.snd_ooo_acks = 0;
            pcb_.rto_rexmits = 0;
            pcb_.rto_maybe_reset();
        }

        TransmitPackets();
    }

    // The following is used to fill packet headers.
    uint32_t local_addr_;
    uint16_t local_port_;
    uint32_t remote_addr_;
    uint16_t remote_port_;
    uint8_t local_l2_addr_[ETH_ALEN];
    uint8_t remote_l2_addr_[ETH_ALEN];

    // The underlying AFXDPSocket.
    AFXDPSocket *socket_;
    // The channel this flow belongs to.
    Channel *channel_;
    // ConnectionID of this flow.
    ConnectionID connection_id_;

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
    const uint32_t SEND_BATCH_SIZE = 32;
    UcclEngine() = delete;
    UcclEngine(UcclEngine const &) = delete;

    /**
     * @brief Construct a new UcclEngine object.
     *
     * @param queue_id      RX/TX queue index to be used by the engine.
     * @param num_frames    Number of frames to be allocated for the queue.
     * @param channel       Uccl channel the engine will be responsible for. For
     *                      now, we assume an engine is responsible for a single
     *                      channel, but future it may be responsible for
     *                      multiple channels.
     */
    UcclEngine(int queue_id, int num_frames, Channel *channel,
               const uint32_t local_addr, const uint16_t local_port,
               const uint32_t remote_addr, const uint16_t remote_port,
               const uint8_t *local_l2_addr, const uint8_t *remote_l2_addr)
        : socket_(AFXDPFactory::CreateSocket(queue_id, num_frames)),
          channel_(channel),
          last_periodic_timestamp_(std::chrono::high_resolution_clock::now()),
          periodic_ticks_(0) {
        // TODO(yang): using TCP-negotiated ConnectionID.
        flow_ = new UcclFlow(local_addr, local_port, remote_addr, remote_port,
                             local_l2_addr, remote_l2_addr, socket_, channel,
                             0xdeadbeaf);
    }

    /**
     * @brief This is the main event cycle of the Uccl engine.
     * It is called by a separate thread running the Uccl engine.
     * On each iteration, the engine processes incoming packets in the RX queue
     * and enqueued messages in all channels that it is responsible for. This
     * method is not thread-safe.
     */
    void Run() {
        // TODO(yang): maintain a queue of rx_work and tx_work
        Channel::Msg rx_work;
        Channel::Msg tx_work;
        FrameBuf *tx_msgbuf_head = nullptr;
        FrameBuf *tx_msgbuf_tail = nullptr;
        uint32_t num_tx_frames = 0;

        while (!shutdown_) {
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

            if (jring_sc_dequeue_bulk(channel_->rx_ring_, &rx_work, 1,
                                      nullptr) == 1) {
                VLOG(3) << "Rx jring dequeue";
                rx_supply_app_buf(rx_work.data, rx_work.len_ptr);
            }

            auto frames = socket_->recv_packets(RECV_BATCH_SIZE);
            if (frames.size()) {
                VLOG(3) << "Rx recv_packets " << frames.size();
                for (auto &frame : frames) {
                    auto msgbuf = FrameBuf::Create(frame.frame_offset,
                                                   socket_->umem_buffer_,
                                                   frame.frame_len);
                    process_rx_msg(msgbuf);
                }
            }

            if (jring_sc_dequeue_bulk(channel_->tx_ring_, &tx_work, 1,
                                      nullptr) == 1) {
                VLOG(3) << "Tx jring dequeue";
                auto *app_buf = tx_work.data;
                auto remaining_bytes = *tx_work.len_ptr;

                //  Deserializing the message into MTU-sized frames.
                FrameBuf *tx_msgbuf_iter = nullptr;
                while (remaining_bytes > 0) {
                    auto payload_len =
                        std::min(remaining_bytes,
                                 (size_t)AFXDP_MTU - kNetHdrLen - kUcclHdrLen);
                    auto frame_offset = socket_->frame_pool_->pop();
                    auto *msgbuf = FrameBuf::Create(
                        frame_offset, socket_->umem_buffer_,
                        payload_len + kNetHdrLen + kUcclHdrLen);
                    //  The transport engine will free these Tx frames when
                    //  receiving ACKs from receivers.
                    msgbuf->mark_not_txpulltime_free();
                    auto pkt_payload_addr =
                        msgbuf->get_pkt_addr() + kNetHdrLen + kUcclHdrLen;
                    memcpy(pkt_payload_addr, app_buf, payload_len);
                    remaining_bytes -= payload_len;
                    app_buf += payload_len;

                    if (tx_msgbuf_head == nullptr) {
                        msgbuf->mark_first();
                        tx_msgbuf_head = msgbuf;
                    } else {
                        tx_msgbuf_iter->set_next(msgbuf);
                    }

                    if (remaining_bytes == 0) {
                        msgbuf->mark_last();
                        tx_msgbuf_tail = msgbuf;
                    }

                    tx_msgbuf_iter = msgbuf;
                    num_tx_frames++;
                }
            }

            if (tx_msgbuf_head && tx_msgbuf_tail) {
                VLOG(3) << "Tx process_tx_msg";
                process_tx_msg(tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames);

                tx_msgbuf_head = nullptr;
                tx_msgbuf_tail = nullptr;
                num_tx_frames = 0;
            }
        }

        // This will reset flow pcb state.
        flow_->Shutdown();
        // This will flush all unpolled tx frames.
        socket_->Shutdown();

        delete flow_;
        delete socket_;
    }

    // Called by application to shutdown the engine. App will need to join the
    // engine thread.
    void Shutdown() { shutdown_ = true; }

    /**
     * @brief Method to perform periodic processing. This is called by the main
     * engine cycle (see method `Run`).
     */
    void PeriodicProcess() {
        // Advance the periodic ticks counter.
        ++periodic_ticks_;
        HandleRTO();
        // DumpStatus();
        ProcessControlRequests();
    }

   protected:
    void DumpStatus() {
        std::string s;
        s += "[Uccl Engine Status]";
        // TODO(yang): Add more status information.
        s += "\n";
        LOG(INFO) << s;
    }

    /**
     * @brief This method polls active channels for all control plane requests
     * and processes them. It is called periodically.
     */
    void ProcessControlRequests() {
        // TODO(yang): maintain pending_requests?
    }

    /**
     * @brief Iterate throught the list of flows, check and handle RTOs.
     */
    void HandleRTO() {
        // TODO(yang): maintain active_flows_map_
        auto is_active_flow = flow_->PeriodicCheck();
        DCHECK(is_active_flow);
    }

    void rx_supply_app_buf(void *app_buf, size_t *app_buf_len) {
        flow_->rx_supply_app_buf(app_buf, app_buf_len);
    }

    /**
     * @brief Process an incoming packet.
     *
     * @param msgbuf Pointer to the packet.
     * @param app_buf Pointer to the application receiving buffer.
     * @param app_buf_len Pointer to the length of the application buffer.
     */
    void process_rx_msg(FrameBuf *msgbuf) {
        // ebpf_transport has filtered out invalid pkts, therefore, we do not
        // need to care about the memory management of these frames.

        auto frame_len = msgbuf->get_frame_len();
        // Sanity ethernet header check.
        DCHECK_GE(frame_len, kNetHdrLen + kUcclHdrLen)
            << "Invalid frame length";

        auto *pkt_addr = msgbuf->get_pkt_addr();
        auto *eh = reinterpret_cast<ethhdr *>(pkt_addr);
        auto *ipv4h = reinterpret_cast<iphdr *>(pkt_addr + sizeof(ethhdr));
        auto *udph = reinterpret_cast<udphdr *>(pkt_addr + sizeof(ethhdr) +
                                                sizeof(iphdr));

        DCHECK_EQ(eh->h_proto, ntohs(ETH_P_IP))
            << "Unsupported ethertype: " << ntohs(eh->h_proto);
        DCHECK_EQ(ipv4h->protocol, IPPROTO_UDP)
            << "Unsupported IP protocol: "
            << static_cast<uint32_t>(ipv4h->protocol);
        DCHECK_EQ(frame_len, sizeof(ethhdr) + ntohs(ipv4h->tot_len))
            << "IPv4 packet length mismatch";

        flow_->RxMsg(msgbuf);
    }

    /**
     * Process a message enqueued from an application to a channel.
     * @param msg     A pointer to the `MsgBuf` containing the first buffer of
     * the message.
     */
    void process_tx_msg(FrameBuf *msg_head, FrameBuf *msg_tail,
                        uint32_t num_frames) {
        // TODO(yang): lookup the msg five-tuple in an active_flows_map
        flow_->TxMsg(msg_head, msg_tail, num_frames);
    }

   private:
    // AFXDP socket used for send/recv packets.
    AFXDPSocket *socket_;
    // For now, we just assume a single flow.
    UcclFlow *flow_;
    // Control plan channel with Endpoint.
    Channel *channel_;
    // Timestamp of last periodic process execution.
    std::chrono::time_point<std::chrono::high_resolution_clock>
        last_periodic_timestamp_;
    // Clock ticks for the slow timer.
    uint64_t periodic_ticks_{0};
    // Whether shutdown is requested.
    std::atomic<bool> shutdown_{false};
};

}  // namespace uccl
