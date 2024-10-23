#pragma once

#include <glog/logging.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <netdb.h>

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
#include "transport_config.h"
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
    constexpr static uint16_t kBootstrapPort = 30000;
    Channel *channel_;
    int listen_fd_;
    int next_avail_conn_id_;
    std::unordered_map<ConnectionID, int> bootstrap_fd_map_;

   public:
    Endpoint(Channel *channel) : channel_(channel), next_avail_conn_id_(0x1) {
        // Create listening socket
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        DCHECK(listen_fd_ >= 0) << "ERROR: opening socket";

        int flag = 1;
        DCHECK(setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &flag,
                          sizeof(int)) >= 0)
            << "ERROR: setsockopt SO_REUSEADDR fails";

        struct sockaddr_in serv_addr;
        bzero((char *)&serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = INADDR_ANY;
        serv_addr.sin_port = htons(kBootstrapPort);
        DCHECK(bind(listen_fd_, (struct sockaddr *)&serv_addr,
                    sizeof(serv_addr)) >= 0)
            << "ERROR: binding";

        DCHECK(!listen(listen_fd_, 5)) << "ERROR: listen";
        LOG(INFO) << "Server ready, listening on port " << kBootstrapPort;
    }
    ~Endpoint() { close(listen_fd_); }

    // Connecting to a remote address.
    ConnectionID uccl_connect(std::string remote_ip) {
        struct sockaddr_in serv_addr;
        struct hostent *server;
        int bootstrap_fd;

        bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
        DCHECK(bootstrap_fd >= 0) << "ERROR: opening socket";

        server = gethostbyname(remote_ip.c_str());
        DCHECK(server) << "ERROR: no such host";

        bzero((char *)&serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr,
              server->h_length);
        serv_addr.sin_port = htons(kBootstrapPort);

        LOG(INFO) << "Connecting to " << remote_ip << " (0x" << std::hex
                  << serv_addr.sin_addr.s_addr << std::dec << ":"
                  << kBootstrapPort << ")";

        // Connect and set nonblocking and nodelay
        while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                       sizeof(serv_addr))) {
            LOG(INFO) << "Connecting... Make sure the server is up.";
            sleep(1);
        }

        int flag = 1;
        setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
                   sizeof(int));

        ConnectionID conn_id;
        int ret = read(bootstrap_fd, &conn_id, sizeof(ConnectionID));
        DCHECK(ret == sizeof(ConnectionID)) << "ERROR: reading connection_id";

        DCHECK(bootstrap_fd_map_.find(conn_id) == bootstrap_fd_map_.end())
            << "Dup ConnectionID";
        bootstrap_fd_map_[conn_id] = bootstrap_fd;

        return conn_id;
    }

    std::tuple<ConnectionID, std::string> uccl_accept() {
        struct sockaddr_in cli_addr;
        socklen_t clilen = sizeof(cli_addr);
        int bootstrap_fd;

        // Accept connection and set nonblocking and nodelay
        bootstrap_fd =
            accept(listen_fd_, (struct sockaddr *)&cli_addr, &clilen);
        DCHECK(bootstrap_fd >= 0) << "ERROR: accept";
        auto ip_str = ip_to_str(cli_addr.sin_addr.s_addr);

        LOG(INFO) << "Accepting from " << ip_str << " (0x" << std::hex
                  << cli_addr.sin_addr.s_addr << std::dec << ":"
                  << cli_addr.sin_port << ")";

        int flag = 1;
        setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
                   sizeof(int));

        // TODO(yang): making it unique across all servers.
        ConnectionID conn_id = next_avail_conn_id_++;

        DCHECK(bootstrap_fd_map_.find(conn_id) == bootstrap_fd_map_.end())
            << "Dup ConnectionID";
        bootstrap_fd_map_[conn_id] = bootstrap_fd;

        int ret = write(bootstrap_fd, &conn_id, sizeof(ConnectionID));
        DCHECK(ret == sizeof(ConnectionID)) << "ERROR: writing connection_id";

        return std::make_tuple(conn_id, ip_str);
    }

    // Sending the data by leveraging multiple port combinations.
    bool uccl_send(ConnectionID connection_id, const void *data,
                   const size_t &len) {
        Channel::Msg msg = {
            .opcode = Channel::Msg::Op::kTx,
            .data = const_cast<void *>(data),
            .len_ptr = const_cast<size_t *>(&len),
            .connection_id = connection_id,
        };
        while (jring_mp_enqueue_bulk(channel_->tx_ring_, &msg, 1, nullptr) !=
               1) {
            // do nothing
        }
        // Wait for the completion.
        while (jring_mc_dequeue_bulk(channel_->tx_comp_ring_, &msg, 1,
                                     nullptr) != 1) {
            // do nothing
            // usleep(5);
        }
        return true;
    }

    // Sending the data by leveraging multiple port combinations.
    bool uccl_send_async(ConnectionID connection_id, const void *data,
                         const size_t &len) {
        Channel::Msg msg = {
            .opcode = Channel::Msg::Op::kTx,
            .data = const_cast<void *>(data),
            .len_ptr = const_cast<size_t *>(&len),
            .connection_id = connection_id,
        };
        while (jring_mp_enqueue_bulk(channel_->tx_ring_, &msg, 1, nullptr) !=
               1) {
            // do nothing
        }
        return true;
    }

    bool uccl_send_poll() {
        Channel::Msg msg;
        // Wait for the completion.
        while (jring_mc_dequeue_bulk(channel_->tx_comp_ring_, &msg, 1,
                                     nullptr) != 1) {
            // do nothing
            // usleep(5);
        }
        return true;
    }

    bool uccl_send_poll_once() {
        Channel::Msg msg;
        // Check for the completion.
        if (jring_mc_dequeue_bulk(channel_->tx_comp_ring_, &msg, 1, nullptr) ==
            1) {
            return true;
        }
        return false;
    }

    // Receiving the data by leveraging multiple port combinations.
    bool uccl_recv(ConnectionID connection_id, void *data, size_t *len) {
        Channel::Msg msg = {
            .opcode = Channel::Msg::Op::kRx,
            .data = data,
            .len_ptr = len,
            .connection_id = connection_id,
        };
        while (jring_mp_enqueue_bulk(channel_->rx_ring_, &msg, 1, nullptr) !=
               1) {
            // do nothing
        }
        // Wait for the completion.
        while (jring_mc_dequeue_bulk(channel_->rx_comp_ring_, &msg, 1,
                                     nullptr) != 1) {
            // do nothing
            // usleep(5);
        }
        return true;
    }

    // Receiving the data by leveraging multiple port combinations.
    bool uccl_recv_async(ConnectionID connection_id, void *data, size_t *len) {
        Channel::Msg msg = {
            .opcode = Channel::Msg::Op::kRx,
            .data = data,
            .len_ptr = len,
            .connection_id = connection_id,
        };
        while (jring_mp_enqueue_bulk(channel_->rx_ring_, &msg, 1, nullptr) !=
               1) {
            // do nothing
        }
        return true;
    }

    bool uccl_recv_poll() {
        Channel::Msg msg;
        // Wait for the completion.
        while (jring_mc_dequeue_bulk(channel_->rx_comp_ring_, &msg, 1,
                                     nullptr) != 1) {
            // do nothing
            // usleep(5);
        }
        return true;
    }

    bool uccl_recv_poll_once() {
        Channel::Msg msg;
        // Check for the completion.
        if (jring_mc_dequeue_bulk(channel_->rx_comp_ring_, &msg, 1, nullptr) ==
            1) {
            return true;
        }
        return false;
    }
};

/**
 * Uccl Packet Header just after UDP header.
 */
struct __attribute__((packed)) UcclPktHdr {
    static constexpr uint16_t kMagic = 0x4e53;
    be16_t magic;  // Magic value tagged after initialization for the flow.
    enum class UcclFlags : uint8_t {
        kData = 0b0,      // Data packet.
        kAck = 0b10,      // ACK packet.
        kAckEcn = 0b100,  // ACK-ECN packet.
    };
    UcclFlags net_flags;  // Network flags.
    uint8_t msg_flags;    // Field to reflect the `FrameBuf' flags.
    be32_t seqno;  // Sequence number to denote the packet counter in the flow.
    be32_t ackno;  // Sequence number to denote the packet counter in the flow.
    be64_t sack_bitmap[4];     // Bitmap of the SACKs received.
    be16_t sack_bitmap_count;  // Length of the SACK bitmap [0-256].
};
static_assert(sizeof(UcclPktHdr) == 46, "UcclPktHdr size mismatch");

#ifdef USING_TCP
static const size_t kNetHdrLen =
    sizeof(ethhdr) + sizeof(iphdr) + sizeof(tcphdr);
#else
static const size_t kNetHdrLen =
    sizeof(ethhdr) + sizeof(iphdr) + sizeof(udphdr);
#endif
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

    const uint32_t num_unsent_msgbufs() const { return num_unsent_msgbufs_; }
    FrameBuf *get_oldest_unacked_msgbuf() const {
        return oldest_unacked_msgbuf_;
    }

    void receive_acks(uint32_t num_acked_pkts) {
        VLOG(3) << "Received " << num_acked_pkts << " acks "
                << "num_tracked_msgbufs " << num_tracked_msgbufs_;
        while (num_acked_pkts) {
            auto msgbuf = oldest_unacked_msgbuf_;
            DCHECK(msgbuf != nullptr);
            // if (msgbuf != last_msgbuf_) {
            if (num_tracked_msgbufs_ != 1) {
                DCHECK_NE(oldest_unacked_msgbuf_, oldest_unsent_msgbuf_)
                    << "Releasing an unsent msgbuf!";
                oldest_unacked_msgbuf_ = msgbuf->next();
                DCHECK(oldest_unacked_msgbuf_ != nullptr);
            } else {
                oldest_unacked_msgbuf_ = nullptr;
                oldest_unsent_msgbuf_ = nullptr;
                last_msgbuf_ = nullptr;
                // CHECK_EQ(num_tracked_msgbufs_, 1);
            }

            if (msgbuf->is_last()) {
                // Tx a full message; wakeup app thread waiting on endpoint.
                VLOG(3) << "Transmitted a complete message";
                Channel::Msg tx_work;
                while (jring_sp_enqueue_bulk(channel_->tx_comp_ring_, &tx_work,
                                             1, nullptr) != 1) {
                    // do nothing
                }
            }
            // Free transmitted frames that are acked
            socket_->push_frame(msgbuf->get_frame_offset());
            num_tracked_msgbufs_--;
            num_acked_pkts--;
        }
    }

    void append(FrameBuf *msgbuf_head, FrameBuf *msgbuf_tail,
                uint32_t num_frames) {
        VLOG(3) << "Appending " << num_frames << " frames "
                << " num_unsent_msgbufs_ " << num_unsent_msgbufs_
                << " last_msgbuf_ " << last_msgbuf_;
        DCHECK(msgbuf_head->is_first());
        DCHECK(msgbuf_tail->is_last());
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

    std::optional<FrameBuf *> get_and_update_oldest_unsent() {
        VLOG(3) << "Get: unsent messages " << num_unsent_msgbufs_
                << " oldest_unsent_msgbuf " << oldest_unsent_msgbuf_;
        if (oldest_unsent_msgbuf_ == nullptr) {
            DCHECK_EQ(num_unsent_msgbufs(), 0);
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
    const uint32_t num_tracked_msgbufs() const { return num_tracked_msgbufs_; }
    const FrameBuf *get_last_msgbuf() const { return last_msgbuf_; }
    const FrameBuf *get_oldest_unsent_msgbuf() const {
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

class UcclFlow;
class UcclEngine;
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
        sizeof(UcclPktHdr::sack_bitmap) * 8;

    static_assert((kReassemblyMaxSeqnoDistance &
                   (kReassemblyMaxSeqnoDistance - 1)) == 0,
                  "kReassemblyMaxSeqnoDistance must be a power of two");

    RXTracking(const RXTracking &) = delete;
    RXTracking(AFXDPSocket *socket, Channel *channel)
        : socket_(socket),
          channel_(channel),
          cur_msg_train_head_(nullptr),
          cur_msg_train_tail_(nullptr) {}

    friend class UcclFlow;
    friend class UcclEngine;

    enum ConsumeRet : int {
        kOldPkt = 0,
        kOOOUntrackable = 1,
        kOOOTrackableDup = 2,
        kOOOTrackableExpectedOrInOrder = 3,
    };

    ConsumeRet consume(swift::Pcb *pcb, FrameBuf *msgbuf) {
        uint8_t *pkt_addr = msgbuf->get_pkt_addr();
        auto frame_len = msgbuf->get_frame_len();
        const auto *ucclh =
            reinterpret_cast<const UcclPktHdr *>(pkt_addr + kNetHdrLen);
        const auto *payload = reinterpret_cast<const UcclPktHdr *>(
            pkt_addr + kNetHdrLen + kUcclHdrLen);
        const auto seqno = ucclh->seqno.value();
        const auto expected_seqno = pcb->rcv_nxt;

        if (swift::seqno_lt(seqno, expected_seqno)) {
            VLOG(3) << "Received old packet: " << seqno << " < "
                    << expected_seqno;
            socket_->push_frame(msgbuf->get_frame_offset());
            return kOldPkt;
        }

        const size_t distance = seqno - expected_seqno;
        if (distance >= kReassemblyMaxSeqnoDistance) {
            VLOG(3)
                << "Packet too far ahead. Dropping as we can't handle SACK. "
                << "seqno: " << seqno << ", expected: " << expected_seqno;
            socket_->push_frame(msgbuf->get_frame_offset());
            return kOOOUntrackable;
        }

        // Only iterate through the deque if we must, i.e., for ooo packts only
        auto it = reass_q_.begin();
        if (seqno != expected_seqno) {
            it = std::find_if(reass_q_.begin(), reass_q_.end(),
                              [&seqno](const reasm_queue_ent_t &entry) {
                                  return entry.seqno >= seqno;
                              });
            VLOG(3) << "Received OOO packet: reass_q size " << reass_q_.size();
            if (it != reass_q_.end() && it->seqno == seqno) {
                VLOG(3) << "Received duplicate packet: " << seqno;
                // Duplicate packet. Drop it.
                socket_->push_frame(msgbuf->get_frame_offset());
                return kOOOTrackableDup;
            }
        }

        // Buffer the packet in the frame pool. It may be out-of-order.
        const size_t payload_len = frame_len - kNetHdrLen - kUcclHdrLen;
        // This records the incoming network packet UcclPktHdr.msg_flags in
        // FrameBuf.
        msgbuf->set_msg_flags(ucclh->msg_flags);

        if (seqno == expected_seqno) {
            VLOG(3) << "Received expected packet: " << seqno;
            reass_q_.emplace_front(msgbuf, seqno);
        } else {
            VLOG(3) << "Received OOO trackable packet: " << seqno;
            reass_q_.insert(it, reasm_queue_ent_t(msgbuf, seqno));
        }

        // Update the SACK bitmap for the newly received packet.
        pcb->sack_bitmap_bit_set(distance);

        // These frames will be freed when the message is delivered to the app.
        push_inorder_msgbuf_to_app(pcb);

        return kOOOTrackableExpectedOrInOrder;
    }

   private:
    void push_inorder_msgbuf_to_app(swift::Pcb *pcb) {
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
                // Stash cur_msg_train_head/tail_ in case application threads
                // have not supplied the app buffer while the engine is keeping
                // receiving messages? Stash this ready message
                ready_msg_stash_.push_back(
                    {cur_msg_train_head_, cur_msg_train_tail_});
                try_copy_msgbuf_to_appbuf(nullptr, nullptr);

                // Reset the message train for the next message.
                cur_msg_train_head_ = nullptr;
                cur_msg_train_tail_ = nullptr;
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
    void try_copy_msgbuf_to_appbuf(void *app_buf, size_t *app_buf_len) {
        if (app_buf && app_buf_len)
            app_buf_stash_.push_back({app_buf, app_buf_len});

        VLOG(2) << "ready_msg_stash_ size: " << ready_msg_stash_.size()
                << " app_buf_stash_ size: " << app_buf_stash_.size();

        while (!ready_msg_stash_.empty() && !app_buf_stash_.empty()) {
            ready_msg_t ready_msg = ready_msg_stash_.front();
            ready_msg_stash_.pop_front();
            app_buf_t app_buf_desc = app_buf_stash_.front();
            app_buf_stash_.pop_front();

            // We have a complete message. Let's deliver it to the app.
            auto *msgbuf_iter = ready_msg.msg_head;
            size_t app_buf_pos = 0;
            while (true) {
                auto *pkt_addr = msgbuf_iter->get_pkt_addr();
                auto *payload_addr = pkt_addr + kNetHdrLen + kUcclHdrLen;
                auto payload_len =
                    msgbuf_iter->get_frame_len() - kNetHdrLen - kUcclHdrLen;

                memcpy((uint8_t *)app_buf_desc.buf + app_buf_pos, payload_addr,
                       payload_len);
                app_buf_pos += payload_len;

                // Free received frames that have been copied to app buf.
                socket_->push_frame(msgbuf_iter->get_frame_offset());
                if (msgbuf_iter->is_last()) break;
                msgbuf_iter = msgbuf_iter->next();

                DCHECK(msgbuf_iter);
            }

            *app_buf_desc.buf_len = app_buf_pos;

            // Wakeup app thread waiting on endpoint.
            Channel::Msg rx_work;
            while (jring_sp_enqueue_bulk(channel_->rx_comp_ring_, &rx_work, 1,
                                         nullptr) != 1) {
                // do nothing
            }

            VLOG(3) << "Received a complete message " << app_buf_pos
                    << " bytes";
        }
    }

   private:
    AFXDPSocket *socket_;
    Channel *channel_;

    struct reasm_queue_ent_t {
        FrameBuf *msgbuf;
        uint64_t seqno;

        reasm_queue_ent_t(FrameBuf *m, uint64_t s) : msgbuf(m), seqno(s) {}
    };
    std::deque<reasm_queue_ent_t> reass_q_;
    FrameBuf *cur_msg_train_head_;
    FrameBuf *cur_msg_train_tail_;
    struct ready_msg_t {
        FrameBuf *msg_head;
        FrameBuf *msg_tail;
    };
    // FIFO queue for ready messages that wait for app to claim.
    std::deque<ready_msg_t> ready_msg_stash_;
    struct app_buf_t {
        void *buf;
        size_t *buf_len;
    };
    std::deque<app_buf_t> app_buf_stash_;
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
    const static uint32_t kReadyMsgThresholdForEcn = 32;

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
    UcclFlow(const std::string local_addr, const uint16_t local_port,
             const std::string remote_addr, const uint16_t remote_port,
             const std::string local_l2_addr, const std::string remote_l2_addr,
             AFXDPSocket *socket, Channel *channel, ConnectionID connection_id)
        : local_addr_(htonl(str_to_ip(local_addr))),
          local_port_(local_port),
          remote_addr_(htonl(str_to_ip(remote_addr))),
          remote_port_(remote_port),
          socket_(CHECK_NOTNULL(socket)),
          channel_(channel),
          connection_id_(connection_id),
          pcb_(),
          tx_tracking_(socket, channel),
          rx_tracking_(socket, channel) {
        DCHECK(str_to_mac(local_l2_addr, local_l2_addr_));
        DCHECK(str_to_mac(remote_l2_addr, remote_l2_addr_));
    }
    ~UcclFlow() {}

    friend class UcclEngine;

    std::string to_string() const {
        std::string s;
        s += "\t\t" + pcb_.to_string() + "\n\t\t[TX] pending msgbufs unsent: " +
             std::to_string(tx_tracking_.num_unsent_msgbufs()) +
             "\n\t\t[RX] ready msgs unconsumed: " +
             std::to_string(rx_tracking_.ready_msg_stash_.size());
        return s;
    }

    void shutdown() { pcb_.rto_disable(); }

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
    void rx_messages(std::vector<FrameBuf *> msgbufs) {
        VLOG(3) << "Received " << msgbufs.size() << " packets";
        uint32_t num_data_frames_recvd = 0;
        bool ecn_recvd = false;
        RXTracking::ConsumeRet consume_ret;
        for (auto msgbuf : msgbufs) {
            // ebpf_transport has filtered out invalid pkts.
            auto *pkt_addr = msgbuf->get_pkt_addr();
            auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);

            switch (ucclh->net_flags) {
                case UcclPktHdr::UcclFlags::kAck:
                    // ACK packet, update the flow.
                    process_ack(ucclh);
                    // Free the received frame.
                    socket_->push_frame(msgbuf->get_frame_offset());
                    break;
                case UcclPktHdr::UcclFlags::kAckEcn:
                    process_ack(ucclh);
                    socket_->push_frame(msgbuf->get_frame_offset());
                    // Need to slowdown the sender.
                    ecn_recvd = true;
                    break;
                case UcclPktHdr::UcclFlags::kData:
                    // Data packet, process the payload. The frame will be freed
                    // once the engine copies the payload into app buffer
                    consume_ret = rx_tracking_.consume(&pcb_, msgbuf);
                    num_data_frames_recvd++;
                    break;
                default:
                    VLOG(3) << "Unsupported UcclFlags: "
                            << std::bitset<8>((uint8_t)ucclh->net_flags);
            }
        }
        // Send one ack for a bunch of received packets.
        if (num_data_frames_recvd) {
            if (rx_tracking_.ready_msg_stash_.size() <=
                kReadyMsgThresholdForEcn) {
                socket_->send_packet(craft_ack(pcb_.seqno(), pcb_.ackno()));
            } else {
                socket_->send_packet(
                    craft_ack_with_ecn(pcb_.seqno(), pcb_.ackno()));
            }
        }

        if (ecn_recvd) {
            // update the cwnd and rate.
            pcb_.mutliplicative_decrease();
        } else {
            pcb_.additive_increase();
        }

        // Sending data frames that can be send per cwnd.
        transmit_pending_packets();
    }

    void supply_rx_app_buf(void *app_buf, size_t *app_buf_len) {
        VLOG(3) << "Supplying app buffer";
        rx_tracking_.try_copy_msgbuf_to_appbuf(app_buf, app_buf_len);
    }

    /**
     * @brief Push a Message from the application onto the egress queue of
     * the flow. Segments the message, and encrypts the packets, and adds
     * all packets onto the egress queue. Caller is responsible for freeing
     * the MsgBuf object.
     *
     * @param msg Pointer to the first message buffer on a train of buffers,
     * aggregating to a partial or a full Message.
     */
    void tx_messages(FrameBuf *msg_head, FrameBuf *msg_tail,
                     uint32_t num_frames) {
        if (num_frames) tx_tracking_.append(msg_head, msg_tail, num_frames);

        // TODO(ilias): We first need to check whether the cwnd is < 1, so
        // that we fallback to rate-based CC.

        // Calculate the effective window (in # of packets) to check whether
        // we can send more packets.
        transmit_pending_packets();
    }

    /**
     * @brief Periodically checks the state of the flow and performs
     * necessary actions.
     *
     * This method is called periodically to check the state of the flow,
     * update the RTO timer, retransmit unacknowledged messages, and
     * potentially remove the flow or notify the application about the
     * connection state.
     *
     * @return Returns true if the flow should continue to be checked
     * periodically, false if the flow should be removed or closed.
     */
    bool periodic_check() {
        if (pcb_.rto_disabled()) return true;

        pcb_.rto_advance();

        // TODO(ilias): send RST packet, indicating removal of the flow.
        if (pcb_.max_rto_rexmits_consectutive_reached()) {
            return false;
        }

        if (pcb_.rto_expired()) {
            // Retransmit the oldest unacknowledged message buffer.
            rto_retransmit();
        }

        return true;
    }

   private:
    void process_ack(const UcclPktHdr *ucclh) {
        auto ackno = ucclh->ackno.value();
        if (swift::seqno_lt(ackno, pcb_.snd_una)) {
            VLOG(3) << "Received old ACK " << ackno;
            return;
        } else if (swift::seqno_eq(ackno, pcb_.snd_una)) {
            VLOG(3) << "Received duplicate ACK " << ackno;
            // Duplicate ACK.
            pcb_.duplicate_acks++;
            // Update the number of out-of-order acknowledgements.
            pcb_.snd_ooo_acks = ucclh->sack_bitmap_count.value();

            if (pcb_.duplicate_acks < swift::Pcb::kFastRexmitDupAckThres) {
                // We have not reached the threshold yet, so we do not do
                // anything.
            } else if (pcb_.duplicate_acks ==
                       swift::Pcb::kFastRexmitDupAckThres) {
                // Fast retransmit.
                VLOG(3) << "Fast retransmit " << ackno;
                fast_retransmit();
            } else {
                // We have already done the fast retransmit, so we are now
                // in the fast recovery phase.
                auto sack_bitmap_count = ucclh->sack_bitmap_count.value();
                // We check the SACK bitmap to see if there are more undelivered
                // packets. In fast recovery mode we get after a fast
                // retransmit, we will retransmit all missing packets that we
                // find from the SACK bitmap, when enumerating the SACK bitmap
                // for up to sack_bitmap_count ACKs.
                auto *msgbuf = tx_tracking_.get_oldest_unacked_msgbuf();
                VLOG(3) << "Fast recovery " << ackno << " sack_bitmap_count "
                        << sack_bitmap_count;
                size_t index = 0;
                while (sack_bitmap_count && msgbuf) {
                    constexpr size_t sack_bitmap_bucket_size =
                        sizeof(ucclh->sack_bitmap[0]) * 8;
                    const size_t sack_bitmap_bucket_idx =
                        index / sack_bitmap_bucket_size;
                    const size_t sack_bitmap_idx_in_bucket =
                        index % sack_bitmap_bucket_size;
                    auto sack_bitmap =
                        ucclh->sack_bitmap[sack_bitmap_bucket_idx].value();
                    if ((sack_bitmap & (1ULL << sack_bitmap_idx_in_bucket)) ==
                        0) {
                        // We found a missing packet.
                        VLOG(3) << "Fast recovery sack_bitmap_count "
                                << sack_bitmap_count;
                        auto seqno = pcb_.snd_una + index;
                        prepare_datapacket(msgbuf, seqno);
                        msgbuf->mark_not_txpulltime_free();
                        missing_frames_.push_back({msgbuf->get_frame_offset(),
                                                   msgbuf->get_frame_len()});
                        pcb_.rto_reset();
                    } else {
                        sack_bitmap_count--;
                    }
                    index++;
                    msgbuf = msgbuf->next();
                }
                if (!missing_frames_.empty()) {
                    socket_->send_packets(missing_frames_);
                    missing_frames_.clear();
                }
            }
        } else if (swift::seqno_gt(ackno, pcb_.snd_nxt)) {
            VLOG(3) << "Received ACK for untransmitted data.";
        } else {
            VLOG(3) << "Received valid ACK " << ackno;
            // This is a valid ACK, acknowledging new data.
            size_t num_acked_packets = ackno - pcb_.snd_una;
            tx_tracking_.receive_acks(num_acked_packets);

            pcb_.snd_una = ackno;
            pcb_.duplicate_acks = 0;
            pcb_.snd_ooo_acks = 0;
            pcb_.rto_rexmits_consectutive = 0;
            pcb_.rto_maybe_reset();
        }
    }

    void prepare_l2header(uint8_t *pkt_addr) const {
        auto *eh = (ethhdr *)pkt_addr;
        memcpy(eh->h_source, local_l2_addr_, ETH_ALEN);
        memcpy(eh->h_dest, remote_l2_addr_, ETH_ALEN);
        eh->h_proto = htons(ETH_P_IP);
    }

    void prepare_l3header(uint8_t *pkt_addr, uint32_t payload_bytes) const {
        auto *ipv4h = (iphdr *)(pkt_addr + sizeof(ethhdr));
        ipv4h->ihl = 5;
        ipv4h->version = 4;
        ipv4h->tos = 0x0;
        ipv4h->id = htons(0x1513);
        ipv4h->frag_off = htons(0);
        ipv4h->ttl = 64;
#ifdef USING_TCP
        ipv4h->protocol = IPPROTO_TCP;
        ipv4h->tot_len = htons(sizeof(iphdr) + sizeof(tcphdr) + payload_bytes);
#else
        ipv4h->protocol = IPPROTO_UDP;
        ipv4h->tot_len = htons(sizeof(iphdr) + sizeof(udphdr) + payload_bytes);
#endif
        ipv4h->saddr = htonl(local_addr_);
        ipv4h->daddr = htonl(remote_addr_);
        ipv4h->check = 0;
        // AWS would block traffic if ipv4 checksum is not calculated.
        ipv4h->check = ipv4_checksum(ipv4h, sizeof(iphdr));
    }

    void prepare_l4header(uint8_t *pkt_addr, uint32_t payload_bytes) const {
#ifdef USING_TCP
        auto *tcph = (tcphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
        memset(tcph, 0, sizeof(tcphdr));
#ifdef USING_MULTIPATH
        static uint16_t rand_port = 0;
        tcph->source = htons(local_port_ + (rand_port++) % 8);
        tcph->dest = htons(remote_port_ + (rand_port++) % 8);
#else
        tcph->source = htons(local_port_);
        tcph->dest = htons(remote_port_);
#endif
        tcph->doff = 5;
        // TODO(yang): tcpdump shows wrong checksum. Need to fix it.
        // tcph->check = tcp_hdr_chksum(htonl(local_addr_), htonl(remote_addr_),
        //                              5 * sizeof(uint32_t) + payload_bytes);
#else
        auto *udph = (udphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
#ifdef USING_MULTIPATH
        static uint16_t rand_port = 0;
        udph->source = htons(local_port_ + (rand_port++) % 8);
        udph->dest = htons(remote_port_ + (rand_port++) % 8);
#else
        udph->source = htons(local_port_);
        udph->dest = htons(remote_port_);
#endif
        udph->len = htons(sizeof(udphdr) + payload_bytes);
        udph->check = htons(0);
        // TODO(yang): Calculate the UDP checksum.
#endif
    }

    void prepare_ucclhdr(uint8_t *pkt_addr, uint32_t seqno, uint32_t ackno,
                         const UcclPktHdr::UcclFlags &net_flags,
                         uint8_t msg_flags = 0) const {
        auto *ucclh = (UcclPktHdr *)(pkt_addr + kNetHdrLen);
        ucclh->magic = be16_t(UcclPktHdr::kMagic);
        ucclh->net_flags = net_flags;
        ucclh->msg_flags = msg_flags;
        ucclh->seqno = be32_t(seqno);
        ucclh->ackno = be32_t(ackno);

        for (size_t i = 0; i < sizeof(UcclPktHdr::sack_bitmap) /
                                   sizeof(UcclPktHdr::sack_bitmap[0]);
             ++i) {
            ucclh->sack_bitmap[i] = be64_t(pcb_.sack_bitmap[i]);
        }
        ucclh->sack_bitmap_count = be16_t(pcb_.sack_bitmap_count);
    }

    AFXDPSocket::frame_desc craft_ctlpacket(
        uint32_t seqno, uint32_t ackno,
        const UcclPktHdr::UcclFlags &flags) const {
        const size_t kControlPayloadBytes = kUcclHdrLen;
        auto frame_offset = socket_->pop_frame();
        auto msgbuf = FrameBuf::Create(frame_offset, socket_->umem_buffer_,
                                       kNetHdrLen + kControlPayloadBytes);
        // Let AFXDPSocket::pull_complete_queue() free control frames.
        msgbuf->mark_txpulltime_free();

        uint8_t *pkt_addr = (uint8_t *)socket_->umem_buffer_ + frame_offset;
        prepare_l2header(pkt_addr);
        prepare_l3header(pkt_addr, kControlPayloadBytes);
        prepare_l4header(pkt_addr, kControlPayloadBytes);
        prepare_ucclhdr(pkt_addr, seqno, ackno, flags);

        return {frame_offset, kNetHdrLen + kControlPayloadBytes};
    }

    AFXDPSocket::frame_desc craft_ack(uint32_t seqno, uint32_t ackno) const {
        VLOG(3) << "Sending ACK for seqno " << seqno << " ackno " << ackno;
        return craft_ctlpacket(seqno, ackno, UcclPktHdr::UcclFlags::kAck);
    }

    AFXDPSocket::frame_desc craft_ack_with_ecn(uint32_t seqno,
                                               uint32_t ackno) const {
        VLOG(3) << "Sending ACK-ECN for seqno " << seqno << " ackno " << ackno;
        return craft_ctlpacket(seqno, ackno, UcclPktHdr::UcclFlags::kAckEcn);
    }

    /**
     * @brief This helper method prepares a network packet that carries the
     * data of a particular `FrameBuf'.
     *
     * @tparam copy_mode Copy mode of the packet. Either kMemCopy or
     * kZeroCopy.
     * @param buf Pointer to the message buffer to be sent.
     * @param packet Pointer to an allocated packet.
     * @param seqno Sequence number of the packet.
     */
    void prepare_datapacket(FrameBuf *msg_buf, uint32_t seqno) const {
        // Header length after before the payload.
        uint32_t frame_len = msg_buf->get_frame_len();
        CHECK_LE(frame_len, AFXDP_MTU);
        uint8_t *pkt_addr = msg_buf->get_pkt_addr();

        // Prepare network headers.
        prepare_l2header(pkt_addr);
        prepare_l3header(pkt_addr, frame_len - kNetHdrLen);
        prepare_l4header(pkt_addr, frame_len - kNetHdrLen);

        // Prepare the Uccl-specific header.
        auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);
        ucclh->magic = be16_t(UcclPktHdr::kMagic);
        ucclh->net_flags = UcclPktHdr::UcclFlags::kData;
        ucclh->ackno = be32_t(UINT32_MAX);
        // This fills the FrameBuf.flags into the outgoing packet
        // UcclPktHdr.msg_flags.
        ucclh->msg_flags = msg_buf->msg_flags();

        ucclh->seqno = be32_t(seqno);
    }

    void fast_retransmit() {
        VLOG(3) << "Fast retransmitting oldest unacked packet " << pcb_.snd_una;
        // Retransmit the oldest unacknowledged message buffer.
        auto *msg_buf = tx_tracking_.get_oldest_unacked_msgbuf();
        if (msg_buf) {
            prepare_datapacket(msg_buf, pcb_.snd_una);
            msg_buf->mark_not_txpulltime_free();
            socket_->send_packet(
                {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
        }
        pcb_.rto_reset();
        pcb_.fast_rexmits++;
    }

    void rto_retransmit() {
        VLOG(3) << "RTO retransmitting oldest unacked packet " << pcb_.snd_una;
        auto *msg_buf = tx_tracking_.get_oldest_unacked_msgbuf();
        if (msg_buf) {
            prepare_datapacket(msg_buf, pcb_.snd_una);
            msg_buf->mark_not_txpulltime_free();
            socket_->send_packet(
                {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
        }
        pcb_.rto_reset();
        pcb_.rto_rexmits++;
        pcb_.rto_rexmits_consectutive++;
    }

    /**
     * @brief Helper function to transmit a number of packets from the queue
     * of pending TX data.
     */
    void transmit_pending_packets() {
        auto remaining_packets =
            std::min(pcb_.effective_wnd(), tx_tracking_.num_unsent_msgbufs());
        if (remaining_packets == 0) return;

        // Prepare the packets.
        for (uint16_t i = 0; i < remaining_packets; i++) {
            auto msg_buf_opt = tx_tracking_.get_and_update_oldest_unsent();
            if (!msg_buf_opt.has_value()) break;

            auto *msg_buf = msg_buf_opt.value();
            auto seqno = pcb_.get_snd_nxt();
            prepare_datapacket(msg_buf, seqno);
            msg_buf->mark_not_txpulltime_free();
            pending_tx_frames_.push_back(
                {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
        }

        // TX both data and ack frames.
        if (pending_tx_frames_.empty()) return;
        VLOG(3) << "transmit_pending_packets " << pending_tx_frames_.size();
        socket_->send_packets(pending_tx_frames_);
        pending_tx_frames_.clear();

        if (pcb_.rto_disabled()) pcb_.rto_enable();
    }

    // The following is used to fill packet headers.
    uint32_t local_addr_;
    uint16_t local_port_;
    uint32_t remote_addr_;
    uint16_t remote_port_;
    char local_l2_addr_[ETH_ALEN];
    char remote_l2_addr_[ETH_ALEN];

    // The underlying AFXDPSocket.
    AFXDPSocket *socket_;
    // The channel this flow belongs to.
    Channel *channel_;
    // ConnectionID of this flow.
    ConnectionID connection_id_;
    // Accumulated data frames to be sent.
    std::vector<AFXDPSocket::frame_desc> pending_tx_frames_;
    // Missing data frames to be sent.
    std::vector<AFXDPSocket::frame_desc> missing_frames_;

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
    const size_t kDumpStatusTicks = 1000;      // 2s
    UcclEngine() = delete;
    UcclEngine(UcclEngine const &) = delete;

    /**
     * @brief Construct a new UcclEngine object.
     *
     * @param queue_id      RX/TX queue index to be used by the engine.
     * @param num_frames    Number of frames to be allocated for the queue.
     * @param channel       Uccl channel the engine will be responsible for.
     * For now, we assume an engine is responsible for a single channel, but
     * future it may be responsible for multiple channels.
     */
    UcclEngine(int queue_id, int num_frames, Channel *channel,
               const std::string local_addr, const uint16_t local_port,
               const std::string remote_addr, const uint16_t remote_port,
               const std::string local_l2_addr,
               const std::string remote_l2_addr)
        : socket_(AFXDPFactory::CreateSocket(queue_id, num_frames)),
          channel_(channel),
          last_periodic_timestamp_(rdtsc_to_us(rdtsc())),
          periodic_ticks_(0) {
        // TODO(yang): using TCP-negotiated ConnectionID.
        flow_ = new UcclFlow(local_addr, local_port, remote_addr, remote_port,
                             local_l2_addr, remote_l2_addr, socket_, channel,
                             0xdeadbeaf);
    }

    /**
     * @brief This is the main event cycle of the Uccl engine.
     * It is called by a separate thread running the Uccl engine.
     * On each iteration, the engine processes incoming packets in the RX
     * queue and enqueued messages in all channels that it is responsible
     * for. This method is not thread-safe.
     */
    void run() {
        // TODO(yang): maintain a queue of rx_work and tx_work
        Channel::Msg rx_work;
        Channel::Msg tx_work;

        while (!shutdown_) {
            // Calculate the time elapsed since the last periodic
            // processing.
            auto now = rdtsc_to_us(rdtsc());
            const auto elapsed = now - last_periodic_timestamp_;

            if (elapsed >= kSlowTimerIntervalUs) {
                // Perform periodic processing.
                periodic_process();
                last_periodic_timestamp_ = now;
            }

            if (jring_sc_dequeue_bulk(channel_->rx_ring_, &rx_work, 1,
                                      nullptr) == 1) {
                VLOG(3) << "Rx jring dequeue";
                supply_rx_app_buf(rx_work.data, rx_work.len_ptr);
            }

            auto frames = socket_->recv_packets(RECV_BATCH_SIZE);
            if (frames.size()) {
                VLOG(3) << "Rx recv_packets " << frames.size();
                std::vector<FrameBuf *> msgbufs;
                msgbufs.reserve(frames.size());
                for (auto &frame : frames) {
                    auto *msgbuf = FrameBuf::Create(frame.frame_offset,
                                                    socket_->umem_buffer_,
                                                    frame.frame_len);
                    msgbufs.push_back(msgbuf);
                }
                process_rx_msg(msgbufs);
            }

            if (jring_sc_dequeue_bulk(channel_->tx_ring_, &tx_work, 1,
                                      nullptr) == 1) {
                VLOG(3) << "Tx jring dequeue";
                auto [tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames] =
                    deserialize_msg(tx_work.data, *tx_work.len_ptr);
                VLOG(3) << "Tx process_tx_msg";
                // Append these tx frames to the flow's tx queue, and trigger
                // intial tx. Future received ACKs will trigger more tx.
                process_tx_msg(tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames);
            }

            // process_tx_msg(nullptr, nullptr, 0);
        }

        // This will reset flow pcb state.
        flow_->shutdown();
        // This will flush all unpolled tx frames.
        socket_->shutdown();

        delete flow_;
        delete socket_;
    }

    /**
     * @brief Method to perform periodic processing. This is called by the
     * main engine cycle (see method `Run`).
     */
    void periodic_process() {
        // Advance the periodic ticks counter.
        periodic_ticks_++;
        if (periodic_ticks_ % kDumpStatusTicks == 0) dump_status();
        handle_rto();
        process_ctl_reqs();
    }

    // Called by application to shutdown the engine. App will need to join
    // the engine thread.
    void shutdown() { shutdown_ = true; }

   protected:
    /**
     * @brief Supply the application with a buffer to receive the incoming
     * message.
     *
     * @param app_buf Pointer to the application buffer.
     * @param app_buf_len Pointer to the length of the application buffer.
     */
    void supply_rx_app_buf(void *app_buf, size_t *app_buf_len) {
        flow_->supply_rx_app_buf(app_buf, app_buf_len);
    }

    /**
     * @brief Process an incoming packet.
     *
     * @param msgbuf Pointer to the packet.
     * @param app_buf Pointer to the application receiving buffer.
     * @param app_buf_len Pointer to the length of the application buffer.
     */
    void process_rx_msg(std::vector<FrameBuf *> msgbufs) {
        flow_->rx_messages(msgbufs);
    }

    /**
     * Process a message enqueued from an application to a channel.
     * @param msg     A pointer to the `MsgBuf` containing the first buffer
     * of the message.
     */
    void process_tx_msg(FrameBuf *msg_head, FrameBuf *msg_tail,
                        uint32_t num_frames) {
        // TODO(yang): lookup the msg five-tuple in an active_flows_map
        flow_->tx_messages(msg_head, msg_tail, num_frames);
    }

    /**
     * @brief Iterate throught the list of flows, check and handle RTOs.
     */
    void handle_rto() {
        // TODO(yang): maintain active_flows_map_
        auto is_active_flow = flow_->periodic_check();
        DCHECK(is_active_flow);
    }

    /**
     * @brief This method polls active channels for all control plane
     * requests and processes them. It is called periodically.
     */
    void process_ctl_reqs() {
        // TODO(yang): maintain pending_requests?
    }

    void dump_status() {
        std::string s;
        s += "\n\t\t[Uccl Engine] " +
             Format("%x [queue %d] <-> %x [queue 0]\n", flow_->local_addr_,
                    socket_->queue_id_, flow_->remote_addr_);
        s += flow_->to_string();
        s += socket_->to_string();
        // TODO(yang): Add more status information.
        s += "\n";
        LOG(INFO) << s;
    }

    std::tuple<FrameBuf *, FrameBuf *, uint32_t> deserialize_msg(
        void *app_buf, size_t app_buf_len) {
        FrameBuf *tx_msgbuf_head = nullptr;
        FrameBuf *tx_msgbuf_tail = nullptr;
        uint32_t num_tx_frames = 0;

        auto remaining_bytes = app_buf_len;

        //  Deserializing the message into MTU-sized frames.
        FrameBuf *last_msgbuf = nullptr;
        while (remaining_bytes > 0) {
            auto payload_len = std::min(
                remaining_bytes, (size_t)AFXDP_MTU - kNetHdrLen - kUcclHdrLen);
            auto frame_offset = socket_->pop_frame();
            auto *msgbuf =
                FrameBuf::Create(frame_offset, socket_->umem_buffer_,
                                 kNetHdrLen + kUcclHdrLen + payload_len);
            // The engine will free these Tx frames when receiving ACKs.
            msgbuf->mark_not_txpulltime_free();

            VLOG(3) << "Deser msgbuf " << msgbuf << " " << num_tx_frames;
            auto pkt_payload_addr =
                msgbuf->get_pkt_addr() + kNetHdrLen + kUcclHdrLen;
            memcpy(pkt_payload_addr, app_buf, payload_len);
            remaining_bytes -= payload_len;
            app_buf += payload_len;

            if (tx_msgbuf_head == nullptr) {
                msgbuf->mark_first();
                tx_msgbuf_head = msgbuf;
            } else {
                last_msgbuf->set_next(msgbuf);
            }

            if (remaining_bytes == 0) {
                msgbuf->mark_last();
                msgbuf->set_next(nullptr);
                tx_msgbuf_tail = msgbuf;
            }

            last_msgbuf = msgbuf;
            num_tx_frames++;
        }
        CHECK(tx_msgbuf_head->is_first());
        CHECK(tx_msgbuf_tail->is_last());
        return std::make_tuple(tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames);
    }

   private:
    // AFXDP socket used for send/recv packets.
    AFXDPSocket *socket_;
    // For now, we just assume a single flow.
    UcclFlow *flow_;
    // Control plan channel with Endpoint.
    Channel *channel_;
    // Timestamp of last periodic process execution.
    uint64_t last_periodic_timestamp_;
    // Clock ticks for the slow timer.
    uint64_t periodic_ticks_;
    // Whether shutdown is requested.
    std::atomic<bool> shutdown_{false};
};

}  // namespace uccl
