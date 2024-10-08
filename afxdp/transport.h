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
#include "transport_config.h"
#include "util.h"
#include "util_afxdp.h"
#include "util_endian.h"

namespace uccl {

typedef uint64_t ConnectionID;

enum ChannelOp : uint8_t {
    kTx = 0,
    kTxComp = 1,
    kRx = 2,
    kRxComp = 3,
};

struct ChannelMsg {
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
    constexpr static uint16_t bootstrap_port = 40000;
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
    ConnectionID Connect(const char *remote_ip) {
        // TODO: Using TCP to negotiate a ConnectionID.
    }

    // Sending the data by leveraging multiple port combinations.
    bool Send(ConnectionID connection_id, const void *data, size_t len) {
        ChannelMsg msg = {
            .opcode = ChannelOp::kTx,
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
    }

    // Receiving the data by leveraging multiple port combinations.
    bool Recv(ConnectionID connection_id, void *data, size_t *len) {
        ChannelMsg msg = {
            .opcode = ChannelOp::kRx,
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
    }
};

/**
 * @struct Key
 * @brief Flow key: corresponds to the 5-tuple (UDP is always the protocol).
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
static_assert(sizeof(Key) == 12, "Flow key size is not 12 bytes.");

/**
 * Machnet Packet Header.
 */
struct __attribute__((packed)) MachnetPktHdr {
    static constexpr uint16_t kMagic = 0x4e53;
    be16_t magic;  // Magic value tagged after initialization for the flow.
    enum class MachnetFlags : uint8_t {
        kData = 0b0,
        kSyn = 0b1,         // SYN packet.
        kAck = 0b10,        // ACK packet.
        kSynAck = 0b11,     // SYN-ACK packet.
        kRst = 0b10000000,  // RST packet.
    };
    MachnetFlags net_flags;  // Network flags.
    uint8_t msg_flags;       // Field to reflect the `MachnetMsgBuf_t' flags.
    be32_t seqno;  // Sequence number to denote the packet counter in the flow.
    be32_t ackno;  // Sequence number to denote the packet counter in the flow.
    be64_t sack_bitmap[4];     // Bitmap of the SACKs received.
    be16_t sack_bitmap_count;  // Length of the SACK bitmap [0-256].
    be64_t timestamp1;         // Timestamp of the packet before sending.
};
static_assert(sizeof(MachnetPktHdr) == 54, "MachnetPktHdr size mismatch");

inline MachnetPktHdr::MachnetFlags operator|(MachnetPktHdr::MachnetFlags lhs,
                                             MachnetPktHdr::MachnetFlags rhs) {
    using MachnetFlagsType =
        std::underlying_type<MachnetPktHdr::MachnetFlags>::type;
    return MachnetPktHdr::MachnetFlags(static_cast<MachnetFlagsType>(lhs) |
                                       static_cast<MachnetFlagsType>(rhs));
}

inline MachnetPktHdr::MachnetFlags operator&(MachnetPktHdr::MachnetFlags lhs,
                                             MachnetPktHdr::MachnetFlags rhs) {
    using MachnetFlagsType =
        std::underlying_type<MachnetPktHdr::MachnetFlags>::type;
    return MachnetPktHdr::MachnetFlags(static_cast<MachnetFlagsType>(lhs) &
                                       static_cast<MachnetFlagsType>(rhs));
}

class FrameBuf {
    // If multi-buffer message (SG), next points to next buffer index.
    FrameBuf *next_;
    // If multi-buffer message (SG), last points to the last buffer index.
    // This is only set in the first buffer of the message.
    FrameBuf *last_;

    // Describing the packet frame.
    uint64_t frame_offset_;
    void *umem_buffer_;
    uint32_t frame_len_;

    // Flags to denote the message buffer state.
    uint8_t flags_;

#define MACHNET_MSGBUF_FLAGS_SYN (1 << 0)
#define MACHNET_MSGBUF_FLAGS_SG (1 << 1)
#define MACHNET_MSGBUF_FLAGS_FIN (1 << 2)
#define MACHNET_MSGBUF_FLAGS_CHAIN (1 << 3)
#define MACHNET_MSGBUF_NOTIFY_DELIVERY (1 << 7)

    FrameBuf(uint64_t frame_offset, void *umem_buffer, uint32_t frame_len)
        : frame_offset_(frame_offset),
          umem_buffer_(umem_buffer),
          frame_len_(frame_len) {
        next_ = nullptr;
        last_ = nullptr;
        flags_ = 0;
    }

   public:
    static FrameBuf *Create(uint64_t frame_offset, void *umem_buffer,
                            uint32_t frame_len) {
        // the XDP_PACKET_HEADROOM bytes before frame_offset is xdp metedata,
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

    uint16_t flags() const { return flags_; }

    // Returns true if the `MachnetMsgBuf_t' is the first in a message.
    bool is_first() const { return (flags_ & MACHNET_MSGBUF_FLAGS_SYN) != 0; }
    // Returns true if the `MachnetMsgBuf_t' is the last in a message.
    bool is_last() const { return (flags_ & MACHNET_MSGBUF_FLAGS_FIN) != 0; }
    // Returns true if the `MachnetMsgBuf_t' is the last in a message.
    bool is_sg() const { return (flags_ & MACHNET_MSGBUF_FLAGS_SG) != 0; }

    // Returns true if there is another message buffer in the chain.
    bool has_next() const { return flags_ & (MACHNET_MSGBUF_FLAGS_SG); }
    // Returns the next message buffer in the chain.
    bool has_chain() const { return flags_ & (MACHNET_MSGBUF_FLAGS_CHAIN); }
    // Returns the next message buffer index in the chain.
    FrameBuf *next() const { return next_; }
    // Returns the last message buffer index in the chain.
    FrameBuf *last() const { return last_; }

    void set_next(FrameBuf *buf) {
        next_ = buf;
        add_flags(MACHNET_MSGBUF_FLAGS_SG);
    }
    void set_last(FrameBuf *buf) { last_ = buf; }
    void mark_first() { add_flags(MACHNET_MSGBUF_FLAGS_SYN); }
    void mark_last() { add_flags(MACHNET_MSGBUF_FLAGS_FIN); }

    void link(FrameBuf *next) {
        DCHECK(is_last()) << "This is not the last buffer of a message!";
        DCHECK(next->is_first())
            << "The next buffer is not the first of a message!";
        next_ = next;
        add_flags(MACHNET_MSGBUF_FLAGS_CHAIN);
    }

    void set_flags(uint16_t flags) { flags_ = flags; }
    void add_flags(uint16_t flags) { flags_ |= flags; }

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
            last_msgbuf_->link(msgbuf);
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
        sizeof(sizeof(MachnetPktHdr::sack_bitmap)) * 8;

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
          cur_msg_train_tail_(nullptr) {}

    // If we fail to allocate in the SHM channel, return -1.
    int Consume(swift::Pcb *pcb, uint64_t frame_offset, void *umem_buffer,
                uint32_t frame_len) {
        const size_t net_hdr_len =
            sizeof(ethhdr) + sizeof(iphdr) + sizeof(udphdr);
        uint8_t *pkt = (uint8_t *)umem_buffer + frame_offset;
        const auto *machneth =
            reinterpret_cast<const MachnetPktHdr *>(pkt + net_hdr_len);
        const auto *payload = reinterpret_cast<const MachnetPktHdr *>(
            pkt + net_hdr_len + sizeof(MachnetPktHdr));
        const auto seqno = machneth->seqno.value();
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
        auto *msgbuf = FrameBuf::Create(frame_offset, umem_buffer, frame_len);
        if (msgbuf == nullptr) {
            VLOG(1) << "Failed to allocate a message buffer. Dropping packet.";
            return -1;
        }

        const size_t payload_len =
            frame_len - net_hdr_len - sizeof(MachnetPktHdr);
        DCHECK(!(msgbuf->is_last() && msgbuf->is_sg()));

        if (seqno == expected_seqno) {
            reass_q_.emplace_front(msgbuf, seqno);
        } else {
            reass_q_.insert(it, reasm_queue_ent_t(msgbuf, seqno));
        }

        // Update the SACK bitmap for the newly received packet.
        pcb->sack_bitmap_bit_set(distance);

        PushInOrderMsgbufsToShmTrain(pcb);
        return 0;
    }

   private:
    void PushInOrderMsgbufsToShmTrain(swift::Pcb *pcb) {
        while (!reass_q_.empty() && reass_q_.front().seqno == pcb->rcv_nxt) {
            auto &front = reass_q_.front();
            auto *msgbuf = front.msgbuf;
            reass_q_.pop_front();

            if (cur_msg_train_head_ == nullptr) {
                DCHECK(msgbuf->is_first());
                cur_msg_train_head_ = msgbuf;
                cur_msg_train_tail_ = msgbuf;
            } else {
                cur_msg_train_tail_->link(msgbuf);
                cur_msg_train_tail_ = msgbuf;
            }

            if (cur_msg_train_tail_->is_last()) {
                // We have a complete message. Let's deliver it to the
                // application.
                DCHECK(!cur_msg_train_tail_->is_sg());
                auto *msgbuf_to_deliver = cur_msg_train_head_;
                // TODO: deliver the message to the application.
                LOG(WARNING) << "Received a complete message!";
                // auto nr_delivered =
                //     channel_->EnqueueMessages(&msgbuf_to_deliver, 1);
                // if (nr_delivered != 1) {
                //     LOG(FATAL) << "SHM channel full, failed to deliver
                //     message";
                // }

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
};

/**
 * @class Flow A flow is a connection between a local and a remote endpoint.
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
class Flow {
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
    Flow(const uint32_t local_addr, const uint16_t local_port,
         const uint32_t remote_addr, const uint16_t remote_port,
         const uint8_t *local_l2_addr, const uint8_t *remote_l2_addr,
         AFXDPSocket *socket)
        : key_(local_addr, local_port, remote_addr, remote_port),
          socket_(CHECK_NOTNULL(socket)),
          state_(State::kClosed),
          pcb_(),
          tx_tracking_(socket),
          rx_tracking_(local_addr, local_port, remote_addr, remote_port,
                       socket) {
        memcpy(local_l2_addr_, local_l2_addr, ETH_ALEN);
        memcpy(remote_l2_addr_, remote_l2_addr, ETH_ALEN);
    }
    ~Flow() {}
    /**
     * @brief Operator to compare if two flows are equal.
     * @param other Other flow to compare to.
     * @return true if the flows are equal, false otherwise.
     */
    bool operator==(const Flow &other) const { return key_ == other.key(); }

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
        CHECK(state_ == State::kClosed);
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
    void InputPacket(uint64_t frame_offset, void *umem_buffer,
                     uint32_t frame_len) {
        // Parse the Machnet header of the packet.
        const size_t net_hdr_len =
            sizeof(ethhdr) + sizeof(iphdr) + sizeof(udphdr);
        uint8_t *pkt = (uint8_t *)umem_buffer + frame_offset;
        const auto *machneth =
            reinterpret_cast<const MachnetPktHdr *>(pkt + net_hdr_len);

        if (machneth->magic.value() != MachnetPktHdr::kMagic) {
            LOG(ERROR) << "Invalid Machnet header magic: " << machneth->magic;
            return;
        }

        switch (machneth->net_flags) {
            case MachnetPktHdr::MachnetFlags::kSyn:
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
                    pcb_.rcv_nxt = machneth->seqno.value();
                    pcb_.advance_rcv_nxt();
                    SendSynAck(pcb_.get_snd_nxt());
                    state_ = State::kSynReceived;
                } else if (state_ == State::kSynReceived) {
                    // If the flow is in SYN-RECEIVED state, our SYN-ACK packet
                    // was lost. We need to retransmit it.
                    SendSynAck(pcb_.snd_una);
                }
                break;
            case MachnetPktHdr::MachnetFlags::kSynAck:
                // SYN-ACK packet received. For this to be valid it has to be an
                // already established flow with this SYN-ACK being a
                // retransmission.
                if (state_ != State::kSynSent &&
                    state_ != State::kEstablished) {
                    LOG(ERROR) << "SYN-ACK packet received for flow in state: "
                               << static_cast<int>(state_);
                    return;
                }

                if (machneth->ackno.value() != pcb_.snd_nxt) {
                    LOG(ERROR)
                        << "SYN-ACK packet received with invalid ackno: "
                        << machneth->ackno << " snd_una: " << pcb_.snd_una
                        << " snd_nxt: " << pcb_.snd_nxt;
                    return;
                }

                if (state_ == State::kSynSent) {
                    pcb_.snd_una++;
                    pcb_.rcv_nxt = machneth->seqno.value();
                    pcb_.advance_rcv_nxt();
                    pcb_.rto_maybe_reset();
                    // Mark the flow as established.
                    state_ = State::kEstablished;
                }
                // Send an ACK packet.
                SendAck();
                break;
            case MachnetPktHdr::MachnetFlags::kRst: {
                const auto seqno = machneth->seqno.value();
                const auto expected_seqno = pcb_.rcv_nxt;
                if (swift::seqno_eq(seqno, expected_seqno)) {
                    // If the RST packet is in sequence, we can reset the flow.
                    state_ = State::kClosed;
                }
            } break;
            case MachnetPktHdr::MachnetFlags::kAck:
                // ACK packet, update the flow.
                // update_flow(machneth);
                process_ack(machneth);
                break;
            case MachnetPktHdr::MachnetFlags::kData:
                if (state_ != State::kEstablished) {
                    LOG(ERROR) << "Data packet received for flow in state: "
                               << static_cast<int>(state_);
                    return;
                }
                // Data packet, process the payload.
                const int consume_returncode = rx_tracking_.Consume(
                    &pcb_, frame_offset, umem_buffer, frame_len);
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
                LOG(INFO) << "Flow " << this << " failed to establish";
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

    void PrepareMachnetHdr(uint8_t *pkt_addr, uint32_t seqno,
                           const MachnetPktHdr::MachnetFlags &net_flags,
                           uint8_t msg_flags = 0) const {
        auto *machneth = (MachnetPktHdr *)(pkt_addr + sizeof(ethhdr) +
                                           sizeof(iphdr) + sizeof(udphdr));
        machneth->magic = be16_t(MachnetPktHdr::kMagic);
        machneth->net_flags = net_flags;
        machneth->msg_flags = msg_flags;
        machneth->seqno = be32_t(seqno);
        machneth->ackno = be32_t(pcb_.ackno());

        for (size_t i = 0; i < sizeof(MachnetPktHdr::sack_bitmap) /
                                   sizeof(MachnetPktHdr::sack_bitmap[0]);
             ++i) {
            machneth->sack_bitmap[i] = be64_t(pcb_.sack_bitmap[i]);
        }
        machneth->sack_bitmap_count = be16_t(pcb_.sack_bitmap_count);

        machneth->timestamp1 = be64_t(0);
    }

    void SendControlPacket(uint32_t seqno,
                           const MachnetPktHdr::MachnetFlags &flags) const {
        auto frame_offset = socket_->frame_pool_->pop();
        uint8_t *pkt_addr = (uint8_t *)socket_->umem_buffer_ + frame_offset;

        const size_t kControlPayloadBytes = sizeof(MachnetPktHdr);
        PrepareL2Header(pkt_addr);
        PrepareL3Header(pkt_addr, kControlPayloadBytes);
        PrepareL4Header(pkt_addr, kControlPayloadBytes);
        PrepareMachnetHdr(pkt_addr, seqno, flags);

        // Send the packet.
        socket_->send_packet({frame_offset, sizeof(ethhdr) + sizeof(iphdr) +
                                                sizeof(ethhdr) +
                                                kControlPayloadBytes});
    }

    void SendSyn(uint32_t seqno) const {
        SendControlPacket(seqno, MachnetPktHdr::MachnetFlags::kSyn);
    }

    void SendSynAck(uint32_t seqno) const {
        SendControlPacket(seqno, MachnetPktHdr::MachnetFlags::kSyn |
                                     MachnetPktHdr::MachnetFlags::kAck);
    }

    void SendAck() const {
        SendControlPacket(pcb_.seqno(), MachnetPktHdr::MachnetFlags::kAck);
    }

    void SendRst() const {
        SendControlPacket(pcb_.seqno(), MachnetPktHdr::MachnetFlags::kRst);
    }

    /**
     * @brief This helper method prepares a network packet that carries the data
     * of a particular `MachnetMsgBuf_t'.
     *
     * @tparam copy_mode Copy mode of the packet. Either kMemCopy or kZeroCopy.
     * @param buf Pointer to the message buffer to be sent.
     * @param packet Pointer to an allocated packet.
     * @param seqno Sequence number of the packet.
     */
    void PrepareDataPacket(FrameBuf *msg_buf, uint32_t seqno) const {
        DCHECK(!(msg_buf->is_last() && msg_buf->is_sg()));
        // Header length after before the payload.
        const size_t hdr_length = sizeof(ethhdr) + sizeof(iphdr) +
                                  sizeof(ethhdr) + sizeof(MachnetPktHdr);
        uint32_t frame_len = msg_buf->get_frame_len();
        CHECK_LE(frame_len, AFXDP_MTU);
        uint8_t *pkt_addr = msg_buf->get_pkt_addr();

        // Prepare network headers.
        PrepareL2Header(pkt_addr);
        PrepareL3Header(pkt_addr, frame_len - hdr_length);
        PrepareL4Header(pkt_addr, frame_len - hdr_length);

        // Prepare the Machnet-specific header.
        auto *machneth = reinterpret_cast<MachnetPktHdr *>(
            pkt_addr + sizeof(ethhdr) + sizeof(iphdr) + sizeof(ethhdr));
        machneth->magic = be16_t(MachnetPktHdr::kMagic);
        machneth->net_flags = MachnetPktHdr::MachnetFlags::kData;
        machneth->ackno = be32_t(UINT32_MAX);
        machneth->msg_flags = msg_buf->flags();
        DCHECK(!(msg_buf->is_last() && msg_buf->is_sg()));

        machneth->seqno = be32_t(seqno);
        machneth->timestamp1 = be64_t(0);
    }

    void FastRetransmit() {
        // Retransmit the oldest unacknowledged message buffer.
        auto *msg_buf = tx_tracking_.GetOldestUnackedMsgBuf();
        PrepareDataPacket(msg_buf, pcb_.snd_una);
        socket_->send_packet(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
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
                {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
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

        std::vector<AFXDPSocket::frame_desc> frames;

        // Prepare the packets.
        for (uint16_t i = 0; i < remaining_packets; i++) {
            auto msg_buf_opt = tx_tracking_.GetAndUpdateOldestUnsent();
            if (!msg_buf_opt.has_value()) break;
            auto *msg_buf = msg_buf_opt.value();
            PrepareDataPacket(msg_buf, pcb_.get_snd_nxt());
            frames.emplace_back(msg_buf->get_frame_offset(),
                                msg_buf->get_frame_len());
        }

        // TX.
        socket_->send_packets(frames);

        if (pcb_.rto_disabled()) pcb_.rto_enable();
    }

    void process_ack(const MachnetPktHdr *machneth) {
        auto ackno = machneth->ackno.value();
        if (swift::seqno_lt(ackno, pcb_.snd_una)) {
            return;
        } else if (swift::seqno_eq(ackno, pcb_.snd_una)) {
            // Duplicate ACK.
            pcb_.duplicate_acks++;
            // Update the number of out-of-order acknowledgements.
            pcb_.snd_ooo_acks = machneth->sack_bitmap_count.value();

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
                auto sack_bitmap_count = machneth->sack_bitmap_count.value();
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
                        sizeof(machneth->sack_bitmap[0]);
                    constexpr size_t sack_bitmap_max_bucket_idx =
                        sizeof(machneth->sack_bitmap) /
                            sizeof(machneth->sack_bitmap[0]) -
                        1;
                    const size_t sack_bitmap_bucket_idx =
                        sack_bitmap_max_bucket_idx -
                        index / sack_bitmap_bucket_size;
                    const size_t sack_bitmap_idx_in_bucket =
                        index % sack_bitmap_bucket_size;
                    auto sack_bitmap =
                        machneth->sack_bitmap[sack_bitmap_bucket_idx].value();
                    if ((sack_bitmap & (1ULL << sack_bitmap_idx_in_bucket)) ==
                        0) {
                        // We found a missing packet.
                        // We skip holes in the SACK bitmap that have already
                        // been retransmitted.
                        if (holes_to_skip-- == 0) {
                            auto seqno = pcb_.snd_una + index;
                            PrepareDataPacket(msgbuf, seqno);
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
    // Flow state.
    State state_;
    // Swift CC protocol control block.
    swift::Pcb pcb_;
    TXTracking tx_tracking_;
    RXTracking rx_tracking_;
};

/**
 * @brief Class `MachnetEngineSharedState' contains any state that might need to
 * be shared among different engines. This might be required when multiple
 * engine threads operate on a single PMD, sharing one or more IP addresses.
 */
class MachnetEngineSharedState {
   public:
    static const size_t kSrcPortMin = (1 << 10);      // 1024
    static const size_t kSrcPortMax = (1 << 16) - 1;  // 65535
    static constexpr size_t kSrcPortBitmapSize =
        (kSrcPortMax + 1) / sizeof(uint64_t) / 8;
    explicit MachnetEngineSharedState(std::vector<uint8_t> rss_key,
                                      uint8_t *l2addr,
                                      std::vector<uint32_t> ipv4_addrs)
        : rss_key_(rss_key) {
        for (const auto &addr : ipv4_addrs) {
            CHECK(ipv4_port_bitmap_.find(addr) == ipv4_port_bitmap_.end());
            ipv4_port_bitmap_.insert(
                {addr, std::vector<uint64_t>(1, UINT64_MAX)});
        }
    }

    const std::unordered_map<uint32_t, std::vector<uint64_t>> &
    GetIpv4PortBitmap() const {
        return ipv4_port_bitmap_;
    }

    bool IsLocalIpv4Address(const uint32_t ipv4_addr) const {
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
     * @param ipv4_addr The net::Ipv4::Address for which the source port is
     * being allocated.
     * @param lambda A predicate (lambda function) taking a uint16_t as input
     * and returning a bool. The function should return true if the input port
     * number is suitable for allocation. Default behavior is to accept any
     * port.
     * @return std::optional<net::Udp::Port> containing the allocated source
     * port if found, or std::nullopt otherwise.
     *
     * Example usage:
     * @code
     * net::Ipv4::Address ipv4_addr = ...;
     * auto port = SrcPortAlloc(ipv4_addr, [](uint16_t port) { return port % 2
     * == 0; }); // Allocate an even port if (port.has_value()) {
     *   // Port successfully allocated, proceed with usage
     * } else {
     *   // No suitable port found
     * }
     * @endcode
     */
    std::optional<uint16_t> SrcPortAlloc(
        const uint32_t &ipv4_addr,
        std::function<bool(uint16_t)> auto &&lambda) {
        constexpr size_t bits_per_slot = sizeof(uint64_t) * 8;
        auto it = ipv4_port_bitmap_.find(ipv4_addr);
        if (it == ipv4_port_bitmap_.end()) {
            return std::nullopt;
        }

        // Helper lambda to find a free port.
        // Given a 64-bit wide slot in a bitmap (vector of uint64_t), find the
        // first available port that satisfies the lambda condition.
        auto find_free_port = [&lambda, &bits_per_slot](
                                  auto &bitmap,
                                  size_t index) -> std::optional<uint16_t> {
            auto mask = ~0ULL;
            do {
                auto pos = __builtin_ffsll(bitmap[index] & mask);
                if (pos == 0) break;  // This slot is fully used.
                const size_t candidate_port = index * bits_per_slot + pos - 1;
                if (candidate_port > kSrcPortMax) break;  // Illegal port.
                if (lambda(candidate_port)) {
                    bitmap[index] &= ~(1ULL << (pos - 1));
                    return candidate_port;
                }
                // If we reached the end of the slot and the port is not
                // suitable, abort.
                if (pos == sizeof(uint64_t) * 8) break;
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
            if (bitmap[i] == 0) continue;  // This slot is fully used.
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
            if (port.has_value()) return port;
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
     * @param ipv4_addr The IPv4 address for which the source port was
     * allocated.
     * @param port The net::Udp::Port instance representing the allocated source
     * port to be released.
     *
     * @note Thread-safe, as it uses a lock_guard to protect concurrent access
     * to the shared data.
     *
     * Example usage:
     * @code
     * net::Ipv4::Address ipv4_addr = ...;
     * net::Udp::Port port = ...; // Previously allocated port
     * SrcPortRelease(ipv4_addr, port); // Release the allocated port
     * @endcode
     */
    void SrcPortRelease(const uint32_t ipv4_addr, const uint16_t port) {
        const std::lock_guard<std::mutex> lock(mtx_);
        SrcPortReleaseLocked(ipv4_addr, port);
    }

    /**
     * @brief Registers a listener on a specific IPv4 address and UDP port,
     * associating the port with a receive queue.
     *
     * This function attempts to register a listener on the provided IPv4
     * address and UDP port. If the port is available and not already in use, it
     * will be associated with the specified receive queue, and the function
     * will return true. If the port is already in use or the provided address
     * and port are not valid, the function returns false.
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
     * @note Thread-safe, as it uses a lock_guard to protect concurrent access
     * to the shared data.
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
    bool RegisterListener(const uint32_t ipv4_addr, const uint16_t port,
                          size_t rx_queue_id) {
        const std::lock_guard<std::mutex> lock(mtx_);
        if (ipv4_port_bitmap_.find(ipv4_addr) == ipv4_port_bitmap_.end()) {
            return false;
        }

        constexpr size_t bits_per_slot = sizeof(uint64_t) * 8;
        auto p = port;
        const auto slot = p / bits_per_slot;
        const auto bit = p % bits_per_slot;
        auto &bitmap = ipv4_port_bitmap_[ipv4_addr];
        if (slot >= bitmap.size()) {
            // Allocate more elements in the bitmap.
            bitmap.resize(slot + 1, ~0ULL);
        }
        // Check if the port is already in use (i.e, bit is unset).
        if (!(bitmap[slot] & (1ULL << bit))) return false;
        bitmap[slot] &= ~(1ULL << bit);

        // Add the port and engine to the listeners.
        DCHECK(listeners_to_rxq.find({ipv4_addr, port}) ==
               listeners_to_rxq.end());
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
     * @note Thread-safe, as it uses a lock_guard to protect concurrent access
     * to the shared data.
     *
     * Example usage:
     * @code
     * net::Ipv4::Address ipv4_addr = ...;
     * net::Udp::Port port = ...; // Previously registered port
     * UnregisterListener(ipv4_addr, port); // Unregister the listener and
     * release the associated port
     * @endcode
     */
    void UnregisterListener(const uint32_t ipv4_addr, const uint16_t port) {
        const std::lock_guard<std::mutex> lock(mtx_);
        auto it = listeners_to_rxq.find({ipv4_addr, port});
        if (it == listeners_to_rxq.end()) {
            return;
        }

        listeners_to_rxq.erase(it);
        SrcPortReleaseLocked(ipv4_addr, port);
    }

    // TODO: Query kernel FIB table and cache in user space
    // std::optional<net::Ethernet::Address> GetL2Addr(
    //     const dpdk::TxRing *txring, const net::Ipv4::Address &local_ip,
    //     const net::Ipv4::Address &target_ip) {
    //     const std::lock_guard<std::mutex> lock(mtx_);
    //     return arp_handler_.GetL2Addr(txring, local_ip, target_ip);
    // }

    // void ProcessArpPacket(dpdk::TxRing *txring, net::Arp *arph) {
    //     const std::lock_guard<std::mutex> lock(mtx_);
    //     arp_handler_.ProcessArpPacket(txring, arph);
    // }

    // std::vector<std::tuple<std::string, std::string>> GetArpTableEntries() {
    //     const std::lock_guard<std::mutex> lock(mtx_);
    //     return arp_handler_.GetArpTableEntries();
    // }

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
    void SrcPortReleaseLocked(const uint32_t ipv4_addr, const uint16_t port) {
        auto it = ipv4_port_bitmap_.find(ipv4_addr);
        if (it == ipv4_port_bitmap_.end()) {
            return;
        }

        constexpr size_t bits_per_slot = sizeof(uint64_t) * 8;
        auto p = port;
        const auto slot = p / bits_per_slot;
        const auto bit = p % bits_per_slot;
        auto &bitmap = ipv4_port_bitmap_[ipv4_addr];
        bitmap[slot] |= (1ULL << bit);
    }

    const std::vector<uint8_t> rss_key_;
    std::mutex mtx_{};
    std::unordered_map<uint32_t, std::vector<uint64_t>> ipv4_port_bitmap_{};
    std::unordered_map<std::pair<uint32_t, uint16_t>, size_t, hash_ip_port_pair>
        listeners_to_rxq{};
};

/**
 * @brief Class `MachnetEngine' abstracts the main Machnet engine. This engine
 * contains all the functionality need to be run by the stack's threads.
 */
class MachnetEngine {
   public:
    // Slow timer (periodic processing) interval in microseconds.
    const size_t kSlowTimerIntervalUs = 2000;  // 2ms
    const size_t kPendingRequestTimeoutSlowTicks = 3;
    // Flow creation timeout in slow ticks (# of periodic executions since
    // flow creation request).
    const size_t kFlowCreationTimeoutSlowTicks = 3;
    MachnetEngine() = delete;
    MachnetEngine(MachnetEngine const &) = delete;

    /**
     * @brief Construct a new MachnetEngine object.
     *
     * @param pmd_port      Pointer to the PMD port to be used by the engine.
     * The PMD port must be initialized (i.e., call InitDriver()).
     * @param rx_queue_id   RX queue index to be used by the engine.
     * @param tx_queue_id   TX queue index to be used by the engine. The TXRing
     *                      associated should be initialized with a packet pool.
     * @param channels      (optional) Machnet channels the engine will be
     *                      responsible for (if any).
     */
    MachnetEngine(int queue_id, int num_frames, Channel *channel,
                  std::shared_ptr<MachnetEngineSharedState> shared_state,
                  const uint32_t local_addr, const uint16_t local_port,
                  const uint32_t remote_addr, const uint16_t remote_port,
                  const uint8_t *local_l2_addr, const uint8_t *remote_l2_addr)
        : socket_(queue_id, num_frames),
          channel_(channel),
          shared_state_(CHECK_NOTNULL(shared_state)),
          last_periodic_timestamp_(std::chrono::high_resolution_clock::now()),
          periodic_ticks_(0) {
        flow_ = std::make_unique<Flow>(local_addr, local_port, remote_addr,
                                       remote_port, local_l2_addr,
                                       remote_l2_addr, &socket_);
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
    void Run() {
        // Calculate the time elapsed since the last periodic processing.
        auto now = std::chrono::high_resolution_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                now - last_periodic_timestamp_)
                .count();

        // time::cycles_to_us(now - last_periodic_timestamp_);
        if (elapsed >= kSlowTimerIntervalUs) {
            // Perform periodic processing.
            PeriodicProcess();
            last_periodic_timestamp_ = now;
        }

        auto frames = socket_.recv_packets(RECV_BATCH_SIZE);
        for (auto &frame : frames) {
            auto pkt = FrameBuf::Create(frame.frame_offset,
                                        socket_.umem_buffer_, frame.frame_len);
            process_rx_pkt(pkt);
        }

        // We have processed the RX batch; release it.
        for (auto &frame : frames) {
            socket_.frame_pool_->push(frame.frame_offset);
        }

        // Process messages from channels.
        FrameBufBatch msg_buf_batch;
        for (auto &channel : channels_) {
            // TODO(ilias): Revisit the number of messages to dequeue.
            const auto nb_msg_dequeued =
                channel->DequeueMessages(&msg_buf_batch);
            for (uint32_t i = 0; i < nb_msg_dequeued; i++) {
                auto *msg = msg_buf_batch.bufs()[i];
                process_tx_pkt(msg);
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
    void PeriodicProcess() {
        // Advance the periodic ticks counter.
        ++periodic_ticks_;
        HandleRTO();
        DumpStatus();
        ProcessControlRequests();
        // Continue the rest of management tasks locked to avoid race conditions
        // with the control plane.
        const std::lock_guard<std::mutex> lock(mtx_);
    }

   protected:
    void DumpStatus() {
        std::string s;
        s += "[Machnet Engine Status]";
        // TODO: Add more status information.
        s += "\n";
        LOG(INFO) << s;
    }

    /**
     * @brief This method polls active channels for all control plane requests
     * and processes them. It is called periodically.
     */
    void ProcessControlRequests() {
        MachnetCtrlQueueEntry_t reqs[MACHNET_CHANNEL_CTRL_SQ_SLOT_NR];
        for (const auto &channel : channels_) {
            // Peek the control SQ.
            const auto nreqs = channel->DequeueCtrlRequests(
                reqs, MACHNET_CHANNEL_CTRL_SQ_SLOT_NR);
            for (auto i = 0u; i < nreqs; i++) {
                const auto &req = reqs[i];
                auto emit_completion = [&req, &channel](bool success) {
                    MachnetCtrlQueueEntry_t resp;
                    resp.id = req.id;
                    resp.opcode = MACHNET_CTRL_OP_STATUS;
                    resp.status = success ? MACHNET_CTRL_STATUS_OK
                                          : MACHNET_CTRL_STATUS_ERROR;
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
                        LOG(ERROR) << "Unknown control plane request opcode: "
                                   << req.opcode;
                }
            }
        }

        for (auto it = pending_requests_.begin();
             it != pending_requests_.end();) {
            const auto &[timestamp_, req, channel] = *it;
            if (periodic_ticks_ - timestamp_ >
                kPendingRequestTimeoutSlowTicks) {
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
            auto rss_lambda =
                [src_addr, dst_addr, dst_port, rss_key = pmd_port_->GetRSSKey(),
                 pmd_port = pmd_port_,
                 rx_queue_id = rxring_->GetRingId()](uint16_t port) -> bool {
                rte_thash_tuple ipv4_l3_l4_tuple;
                ipv4_l3_l4_tuple.v4.src_addr = src_addr.address.value();
                ipv4_l3_l4_tuple.v4.dst_addr = dst_addr.address.value();
                ipv4_l3_l4_tuple.v4.sport = port;
                ipv4_l3_l4_tuple.v4.dport = dst_port.port.value();

                rte_thash_tuple reversed_ipv4_l3_l4_tuple;
                reversed_ipv4_l3_l4_tuple.v4.src_addr =
                    dst_addr.address.value();
                reversed_ipv4_l3_l4_tuple.v4.dst_addr =
                    src_addr.address.value();
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

                if (pmd_port->GetRSSRxQueue(
                        __builtin_bswap32(reversed_rss_hash)) != rx_queue_id) {
                    return false;
                }

                LOG(INFO) << "RSS hash for " << src_addr.ToString() << ":"
                          << port << " -> " << dst_addr.ToString() << ":"
                          << dst_port.port.value() << " is " << rss_hash
                          << " and reversed " << reversed_rss_hash
                          << " (queue: " << rx_queue_id << ")";

                return true;
            };

            auto src_port = shared_state_->SrcPortAlloc(src_addr, rss_lambda);
            if (!src_port.has_value()) {
                LOG(ERROR) << "Cannot allocate source port for "
                           << src_addr.ToString();
                it = pending_requests_.erase(it);
                continue;
            }

            auto application_callback =
                [req_id = req.id](shm::Channel *channel, bool success,
                                  const juggler::net::flow::Key &flow_key) {
                    MachnetCtrlQueueEntry_t resp;
                    resp.id = req_id;
                    resp.opcode = MACHNET_CTRL_OP_STATUS;
                    resp.status = success ? MACHNET_CTRL_STATUS_OK
                                          : MACHNET_CTRL_STATUS_ERROR;
                    resp.flow_info.src_ip = flow_key.local_addr.address.value();
                    resp.flow_info.src_port = flow_key.local_port.port.value();
                    resp.flow_info.dst_ip =
                        flow_key.remote_addr.address.value();
                    resp.flow_info.dst_port = flow_key.remote_port.port.value();
                    channel->EnqueueCtrlCompletions(&resp, 1);
                };
            const auto &flow_it = channel->CreateFlow(
                src_addr, src_port.value(), dst_addr, dst_port,
                pmd_port_->GetL2Addr(), remote_l2_addr.value(), txring_,
                application_callback);
            (*flow_it)->InitiateHandshake();
            active_flows_map_.emplace((*flow_it)->key(), flow_it);
            it = pending_requests_.erase(it);
        }
    }

    /**
     * @brief Iterate throught the list of flows, check and handle RTOs.
     */
    void HandleRTO() {
        // TODO: maintain active_flows_map_
        auto is_active_flow = flow_->PeriodicCheck();
        CHECK(is_active_flow);
    }

    /**
     * @brief Process an incoming packet.
     *
     * @param pkt Pointer to the packet.
     * @param now TSC timestamp.
     */
    void process_rx_pkt(FrameBuf *pkt) {
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
          process_rx_ipv4(pkt);
        break;
            // clang-format on
            case Ethernet::kIpv6:
                // We do not support IPv6 yet.
                break;
            default:
                break;
        }
    }

    void process_rx_ipv4(FrameBuf *pkt) {
        // Sanity ipv4 header check.
        if (pkt->length() < sizeof(Ethernet) + sizeof(Ipv4)) [[unlikely]]
            return;

        const auto *eh = pkt->head_data<Ethernet *>();
        const auto *ipv4h = pkt->head_data<Ipv4 *>(sizeof(Ethernet));
        const auto *udph =
            pkt->head_data<Udp *>(sizeof(Ethernet) + sizeof(Ipv4));

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
                if (active_flows_map_.find(pkt_key) !=
                    active_flows_map_.end()) {
                    const auto &flow_it = active_flows_map_[pkt_key];
                    (*flow_it)->InputPacket(pkt);
                    return;
                }

                {
                    // If we reach here, it means that the packet does not
                    // belong to any active flow. Check if there is a listener
                    // on this port.
                    const auto &local_ipv4_addr = ipv4h->dst_addr;
                    const auto &local_udp_port = udph->dst_port;
                    if (listeners_.find(local_ipv4_addr) != listeners_.end()) {
                        // We have a listener on this port.
                        const auto &listeners_on_ip =
                            listeners_[local_ipv4_addr];
                        if (listeners_on_ip.find(local_udp_port) ==
                            listeners_on_ip.end()) {
                            LOG(INFO)
                                << "Dropping packet with RSS hash: "
                                << pkt->rss_hash() << " (be: "
                                << __builtin_bswap32(pkt->rss_hash()) << ")"
                                << " because there is no listener on port "
                                << local_udp_port.port.value()
                                << " (engine @rx_q_id: " << rxring_->GetRingId()
                                << ")";
                            return;
                        }

                        // Create a new flow.
                        const auto &channel =
                            listeners_on_ip.at(local_udp_port);
                        const auto &remote_ipv4_addr = ipv4h->src_addr;
                        const auto &remote_udp_port = udph->src_port;

                        // Check if it is a SYN packet.
                        const auto *machneth =
                            pkt->head_data<net::MachnetPktHdr *>(
                                sizeof(Ethernet) + sizeof(Ipv4) + sizeof(Udp));
                        if (machneth->net_flags !=
                            net::MachnetPktHdr::MachnetFlags::kSyn) {
                            LOG(WARNING) << "Received a non-SYN packet on a "
                                            "listening port";
                            break;
                        }

                        auto empty_callback = [](shm::Channel *, bool,
                                                 const net::flow::Key &) {};
                        const auto &flow_it = channel->CreateFlow(
                            local_ipv4_addr, local_udp_port, remote_ipv4_addr,
                            remote_udp_port, pmd_port_->GetL2Addr(),
                            eh->src_addr, txring_, empty_callback);
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
     * @param msg     A pointer to the `MsgBuf` containing the first buffer of
     * the message.
     */
    void process_tx_pkt(FrameBuf *msg) {
        // TODO: lookup the msg five-tuple in an active_flows_map_
        flow_->OutputMessage(msg);
    }

   private:
    static const size_t kSrcPortMin = (1 << 10);      // 1024
    static const size_t kSrcPortMax = (1 << 16) - 1;  // 65535
    static constexpr size_t kSrcPortBitmapSize =
        ((kSrcPortMax - kSrcPortMin + 1) + sizeof(uint64_t) - 1) /
        sizeof(uint64_t);

    // AFXDP socket used for send/recv packets.
    AFXDPSocket socket_;
    // For now, we just assume a single flow.
    std::unique_ptr<Flow> flow_;
    // Control plan channel with Endpoint.
    Channel *channel_;
    // A mutex to synchronize control plane operations.
    std::mutex mtx_;
    // Shared State instance for this engine.
    std::shared_ptr<MachnetEngineSharedState> shared_state_;
    // Timestamp of last periodic process execution.
    std::chrono::time_point<std::chrono::high_resolution_clock>
        last_periodic_timestamp_;
    // Clock ticks for the slow timer.
    uint64_t periodic_ticks_{0};
    // List of pending control plane requests.
    std::list<std::tuple<uint64_t, ChannelMsg>> pending_requests_{};
};

}  // namespace uccl

namespace std {

template <>
struct hash<uccl::Flow> {
    size_t operator()(const uccl::Flow &flow) const {
        const auto &key = flow.key();
        return std::hash<std::string_view>{}(
            {reinterpret_cast<const char *>(&key), sizeof(key)});
    }
};

}  // namespace std