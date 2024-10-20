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
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <xdp/libxdp.h>
#include <xdp/xsk.h>

#include <deque>
#include <memory>
#include <mutex>
#include <set>

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

    void mark_first() { add_msg_flags(UCCL_MSGBUF_FLAGS_SYN); }
    void mark_last() { add_msg_flags(UCCL_MSGBUF_FLAGS_FIN); }

#define GET_FRAMEBUF_PTR(frame_offset, umem_buffer)                     \
    reinterpret_cast<FrameBuf *>(frame_offset + (uint64_t)umem_buffer - \
                                 XDP_PACKET_HEADROOM)

    static FrameBuf *get_msgbuf_ptr(uint64_t frame_offset, void *umem_buffer) {
        return GET_FRAMEBUF_PTR(frame_offset, umem_buffer);
    }

    void mark_txpulltime_free() {
        add_msg_flags(UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE);
    }
    void mark_not_txpulltime_free() {
        msg_flags_ &= ~UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE;
    }
    bool is_txpulltime_free() {
        return (msg_flags_ & UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE) != 0;
    }

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

    void clear_fields() {
        next_ = nullptr;
        frame_offset_ = 0;
        umem_buffer_ = nullptr;
        frame_len_ = 0;
        msg_flags_ = 0;
    }
    static void clear_fields(uint64_t frame_offset, void *umem_buffer) {
        auto msgbuf = GET_FRAMEBUF_PTR(frame_offset, umem_buffer);
        msgbuf->clear_fields();
    }

    void set_msg_flags(uint16_t flags) { msg_flags_ = flags; }
    void add_msg_flags(uint16_t flags) { msg_flags_ |= flags; }
};

class AFXDPSocket;

class AFXDPFactory {
#define SHM_NAME "UMEM_SHM"
#define SOCKET_PATH "/tmp/privileged_socket"

   public:
    // UDS socket to connect to the afxdp daemon.
    int client_sock_;
    char interface_name_[256];
    std::mutex socket_q_lock_;
    std::deque<AFXDPSocket *> socket_q_;

    static void init(const char *interface_name, const char *ebpf_filename,
                     const char *section_name);
    static AFXDPSocket *CreateSocket(int queue_id, int num_frames);
    static void shutdown();
};

extern AFXDPFactory afxdp_ctl;

class AFXDPSocket {
#define FILL_RING_SIZE                  \
    (XSK_RING_PROD__DEFAULT_NUM_DESCS * \
     2)  // recommened to be RX_RING_SIZE + NIC RING SIZE
#define COMP_RING_SIZE XSK_RING_CONS__DEFAULT_NUM_DESCS
#define TX_RING_SIZE XSK_RING_PROD__DEFAULT_NUM_DESCS
#define RX_RING_SIZE XSK_RING_CONS__DEFAULT_NUM_DESCS

    constexpr static uint32_t QUEUE_ID = 0;
    constexpr static uint32_t NUM_FRAMES = 64 * 4096;
    constexpr static uint32_t FRAME_SIZE = XSK_UMEM__DEFAULT_FRAME_SIZE;

    AFXDPSocket(int queue_id, int num_frames);

    // For manually mapping umem struct from the afxdp daemon.
    typedef __u64 u64;
    typedef __u32 u32;
    typedef __u16 u16;
    typedef __u8 u8;

    /* Up until and including Linux 5.3 */
    struct xdp_ring_offset_v1 {
        __u64 producer;
        __u64 consumer;
        __u64 desc;
    };

    /* Up until and including Linux 5.3 */
    struct xdp_mmap_offsets_v1 {
        struct xdp_ring_offset_v1 rx;
        struct xdp_ring_offset_v1 tx;
        struct xdp_ring_offset_v1 fr;
        struct xdp_ring_offset_v1 cr;
    };

    void xsk_mmap_offsets_v1(struct xdp_mmap_offsets *off);
    int xsk_get_mmap_offsets(int fd, struct xdp_mmap_offsets *off);
    void destroy_afxdp_socket();
    int create_afxdp_socket();

    int xsk_fd_;
    int umem_fd_;

    void *fill_map_;
    size_t fill_map_size_;
    void *comp_map_;
    size_t comp_map_size_;
    void *rx_map_;
    size_t rx_map_size_;
    void *tx_map_;
    size_t tx_map_size_;

   public:
    uint32_t queue_id_;
    uint32_t num_frames_;
    void *umem_buffer_;
    size_t umem_size_;
    struct xsk_ring_cons recv_queue_;
    struct xsk_ring_prod send_queue_;
    struct xsk_ring_cons complete_queue_;
    struct xsk_ring_prod fill_queue_;
    FramePool</*Sync=*/false> *frame_pool_;
#if 0
    std::set<uint64_t> free_frames_;
#endif

    struct frame_desc {
        uint64_t frame_offset;
        uint32_t frame_len;
    };

    // The completed packets might come from last send_packet call.
    uint32_t send_queue_free_entries(uint32_t nb_frames);
    uint32_t send_packet(frame_desc frame);
    uint32_t send_packets(std::vector<frame_desc> &frames);
    uint32_t pull_complete_queue();

    void populate_fill_queue(uint32_t nb_frames);
    std::vector<frame_desc> recv_packets(uint32_t nb_frames);

    uint64_t pop_frame() {
#if 0
        auto frame_offset = frame_pool_->pop();
        CHECK(free_frames_.erase(frame_offset) == 1);
        FrameBuf::clear_fields(frame_offset, umem_buffer_);
        return frame_offset;
#else
        auto frame_offset = frame_pool_->pop();
        FrameBuf::clear_fields(frame_offset, umem_buffer_);
        return frame_offset;
#endif
    }

    void push_frame(uint64_t frame_offset) {
#if 0
        if (free_frames_.find(frame_offset) == free_frames_.end()) {
            free_frames_.insert(frame_offset);
            frame_pool_->push(frame_offset);
        } else {
            LOG(ERROR) << "Frame offset " << std::hex << frame_offset
                       << " size " << std::dec
                       << FrameBuf::get_msgbuf_ptr(frame_offset, umem_buffer_)
                              ->get_frame_len()
                       << " already in free_frames_";
        }
#else
        frame_pool_->push(frame_offset);
#endif
    }

    int get_xsk_fd() const { return xsk_fd_; }
    int get_umem_fd() const { return umem_fd_; }

    std::string to_string() const;
    void shutdown();
    ~AFXDPSocket();

    friend class AFXDPFactory;

   private:
    uint32_t unpulled_tx_pkts_;
    uint32_t fill_queue_entries_;
};

}  // namespace uccl