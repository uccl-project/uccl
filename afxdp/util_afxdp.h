#pragma once

#include <arpa/inet.h>
#include <assert.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <glog/logging.h>
#include <ifaddrs.h>
#include <inttypes.h>
#include <linux/if_ether.h>
#include <linux/if_link.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <memory.h>
#include <net/if.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>
#include <xdp/libxdp.h>
#include <xdp/xsk.h>

#include <deque>
#include <memory>
#include <mutex>

#include "util_umem.h"

namespace uccl {

constexpr static uint32_t AFXDP_MTU = 3498;

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
#define UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE (1 << 2)
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
        /*
         * The XDP_PACKET_HEADROOM bytes before frame_offset is xdp metedata,
         * and we reuse it to chain Framebufs.
         */
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
        DCHECK(is_last()) << "This is not the last buffer of a message!";
        DCHECK(next->is_first())
            << "The next buffer is not the first of a message!";
        next_ = next;
    }

    void mark_first() { add_msg_flags(UCCL_MSGBUF_FLAGS_SYN); }
    void mark_last() { add_msg_flags(UCCL_MSGBUF_FLAGS_FIN); }

    void mark_txpulltime_free() {
        add_msg_flags(UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE);
    }
    void mark_not_txpulltime_free() {
        msg_flags_ &= ~UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE;
    }
    bool is_txpulltime_free() {
        return (msg_flags_ & UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE) != 0;
    }

#define GET_FRAMEBUF_PTR(frame_offset, umem_buffer)                     \
    reinterpret_cast<FrameBuf *>(frame_offset + (uint64_t)umem_buffer - \
                                 XDP_PACKET_HEADROOM)

    static void mark_txpulltime_free(uint64_t frame_offset, void *umem_buffer) {
        auto msgbuf = GET_FRAMEBUF_PTR(frame_offset, umem_buffer);
        msgbuf->add_msg_flags(UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE);
    }
    static void mark_not_txpulltime_free(uint64_t frame_offset,
                                         void *umem_buffer) {
        auto msgbuf = GET_FRAMEBUF_PTR(frame_offset, umem_buffer);
        msgbuf->msg_flags_ &= ~UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE;
    }
    static bool is_txpulltime_free(uint64_t frame_offset, void *umem_buffer) {
        auto msgbuf = GET_FRAMEBUF_PTR(frame_offset, umem_buffer);
        return (msgbuf->msg_flags_ & UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE) != 0;
    }

    void set_msg_flags(uint16_t flags) { msg_flags_ = flags; }
    void add_msg_flags(uint16_t flags) { msg_flags_ |= flags; }
};

class AFXDPSocket;

class AFXDPFactory {
   public:
    int interface_index_;
    char interface_name_[256];
    struct xdp_program *program_;
    bool attached_native_;
    bool attached_skb_;
    std::mutex socket_q_lock_;
    std::deque<AFXDPSocket *> socket_q_;

    static void init(const char *interface_name, const char *ebpf_filename,
                     const char *section_name);
    static AFXDPSocket *CreateSocket(int queue_id, int num_frames);
    static void shutdown();
};

extern AFXDPFactory afxdp_ctl;

class AFXDPSocket {
    constexpr static uint32_t FRAME_SIZE = XSK_UMEM__DEFAULT_FRAME_SIZE;
    constexpr static uint32_t RECV_SPIN_US = 10;

    AFXDPSocket(int queue_id, int num_frames);

   public:
    int queue_id_;
    void *umem_buffer_;
    struct xsk_umem *umem_;
    struct xsk_socket *xsk_;
    struct xsk_ring_cons recv_queue_;
    struct xsk_ring_prod send_queue_;
    struct xsk_ring_cons complete_queue_;
    struct xsk_ring_prod fill_queue_;
    FramePool *frame_pool_;

    struct frame_desc {
        uint64_t frame_offset;
        uint32_t frame_len;
    };

    // The completed packets might come from last send_packet call.
    uint32_t send_packet(frame_desc frame);
    uint32_t send_packets(std::vector<frame_desc> &frames);
    uint32_t pull_complete_queue();

    void populate_fill_queue(uint32_t nb_frames);
    std::vector<frame_desc> recv_packets(uint32_t nb_frames);

    std::string to_string() const;
    void shutdown();
    ~AFXDPSocket();

    friend class AFXDPFactory;

   private:
    uint32_t unpulled_tx_pkts_;
    uint32_t fill_queue_entries_;
};

}  // namespace uccl