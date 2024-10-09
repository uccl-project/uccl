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

class AFXDPSocket;

class AFXDPFactory {
   public:
    int interface_index_;
    char interface_name_[256];
    struct xdp_program* program_;
    bool attached_native_;
    bool attached_skb_;
    std::mutex socket_q_lock_;
    std::deque<AFXDPSocket*> socket_q_;

    static void init(const char* interface_name, const char* ebpf_filename,
                     const char* section_name);
    static AFXDPSocket* CreateSocket(int queue_id, int num_frames);
    static void shutdown();
};

extern AFXDPFactory afxdp_ctl;

class AFXDPSocket {
    constexpr static uint32_t FRAME_SIZE = XSK_UMEM__DEFAULT_FRAME_SIZE;

    AFXDPSocket(int queue_id, int num_frames);

   public:
    int queue_id_;
    void* umem_buffer_;
    struct xsk_umem* umem_;
    struct xsk_socket* xsk_;
    struct xsk_ring_cons recv_queue_;
    struct xsk_ring_prod send_queue_;
    struct xsk_ring_cons complete_queue_;
    struct xsk_ring_prod fill_queue_;
    FramePool* frame_pool_;

    struct frame_desc {
        uint64_t frame_offset;
        uint32_t frame_len;
    };

    // The completed packets might come from last send_packet call.
    uint32_t send_packet(frame_desc frame, bool free_frame);
    uint32_t send_packets(std::vector<frame_desc>& frames, bool free_frame);
    uint32_t pull_complete_queue(bool free_frame);

    void populate_fill_queue(uint32_t nb_frames);
    std::vector<frame_desc> recv_packets(uint32_t nb_frames);

    ~AFXDPSocket();

    friend class AFXDPFactory;
};

}  // namespace uccl