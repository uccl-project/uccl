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
#include <pthread.h>
#include <signal.h>
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

    static void init(const char* interface_name, const char* xdp_program_path);
    static AFXDPSocket* createSocket(int queue_id, int num_frames);
    static void shutdown();
};

extern AFXDPFactory afxdp_ctl;

class AFXDPSocket {
    constexpr static int FRAME_SIZE = XSK_UMEM__DEFAULT_FRAME_SIZE;

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

    AFXDPSocket(int queue_id, int num_frames);
    ~AFXDPSocket();
};

}  // namespace uccl