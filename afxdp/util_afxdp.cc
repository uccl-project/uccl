#include "util_afxdp.h"

namespace uccl {

AFXDPFactory afxdp_ctl;

void AFXDPFactory::init(const char* interface_name, const char* ebpf_filename,
                        const char* section_name) {
    // we can only run xdp programs as root
    CHECK(geteuid() == 0) << "error: this program must be run as root";

    strcpy(afxdp_ctl.interface_name_, interface_name);

    // find the network interface that matches the interface name
    {
        bool found = false;
        struct ifaddrs* addrs;
        CHECK(getifaddrs(&addrs) == 0) << "error: getifaddrs failed";

        for (struct ifaddrs* iap = addrs; iap != NULL; iap = iap->ifa_next) {
            if (iap->ifa_addr && (iap->ifa_flags & IFF_UP) &&
                iap->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in* sa = (struct sockaddr_in*)iap->ifa_addr;
                if (strcmp(interface_name, iap->ifa_name) == 0) {
                    LOG(INFO) << "found network interface: " << iap->ifa_name;
                    afxdp_ctl.interface_index_ = if_nametoindex(iap->ifa_name);
                    CHECK(afxdp_ctl.interface_index_)
                        << "error: if_nametoindex failed";
                    found = true;
                    break;
                }
            }
        }

        freeifaddrs(addrs);

        CHECK(found) << "error: could not find any network interface matching "
                     << interface_name;
    }

    // load the ebpf_client program and attach it to the network interface
    LOG(INFO) << "loading " << section_name << "...";

    afxdp_ctl.program_ =
        xdp_program__open_file(ebpf_filename, section_name, NULL);
    CHECK(!libxdp_get_error(afxdp_ctl.program_))
        << "error: could not load " << ebpf_filename << "program";

    LOG(INFO) << "ebpf_client loaded successfully.";
    LOG(INFO) << "attaching ebpf_client to network interface";

    int ret = xdp_program__attach(
        afxdp_ctl.program_, afxdp_ctl.interface_index_, XDP_MODE_NATIVE, 0);
    if (ret == 0) {
        afxdp_ctl.attached_native_ = true;
    } else {
        LOG(INFO) << "falling back to skb mode...";
        ret = xdp_program__attach(afxdp_ctl.program_,
                                  afxdp_ctl.interface_index_, XDP_MODE_SKB, 0);
        if (ret == 0) {
            afxdp_ctl.attached_skb_ = true;
        } else {
            LOG(ERROR) << "error: failed to attach ebpf_client program to "
                          "interface";
        }
    }

    // allow unlimited locking of memory, so all memory needed for packet
    // buffers can be locked
    struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};
    CHECK(!setrlimit(RLIMIT_MEMLOCK, &rlim)) << "error: could not setrlimit";
}

AFXDPSocket* AFXDPFactory::CreateSocket(int queue_id, int num_frames) {
    auto socket = new AFXDPSocket(queue_id, num_frames);
    std::lock_guard<std::mutex> lock(afxdp_ctl.socket_q_lock_);
    afxdp_ctl.socket_q_.push_back(socket);
    return socket;
}

void AFXDPFactory::shutdown() {
    if (afxdp_ctl.program_ != NULL) {
        if (afxdp_ctl.attached_native_) {
            xdp_program__detach(afxdp_ctl.program_, afxdp_ctl.interface_index_,
                                XDP_MODE_NATIVE, 0);
        }

        if (afxdp_ctl.attached_skb_) {
            xdp_program__detach(afxdp_ctl.program_, afxdp_ctl.interface_index_,
                                XDP_MODE_SKB, 0);
        }

        xdp_program__close(afxdp_ctl.program_);
    }

    std::lock_guard<std::mutex> lock(afxdp_ctl.socket_q_lock_);
    for (auto socket : afxdp_ctl.socket_q_) {
        delete socket;
    }
    afxdp_ctl.socket_q_.clear();
}

AFXDPSocket::AFXDPSocket(int queue_id, int num_frames) : unpulled_tx_pkts_(0) {
    // initialize queues, or misterious queue sync problems will happen
    memset(&recv_queue_, 0, sizeof(recv_queue_));
    memset(&send_queue_, 0, sizeof(send_queue_));
    memset(&complete_queue_, 0, sizeof(complete_queue_));
    memset(&fill_queue_, 0, sizeof(fill_queue_));

    queue_id_ = queue_id;

    // allocate buffer for umem
    const int buffer_size = num_frames * FRAME_SIZE;

    if (posix_memalign(&umem_buffer_, getpagesize(), buffer_size)) {
        printf("\nerror: could not allocate buffer\n\n");
        exit(0);
    }

    // allocate umem
    int ret = xsk_umem__create(&umem_, umem_buffer_, buffer_size, &fill_queue_,
                               &complete_queue_, NULL);
    if (ret) {
        printf("\nerror: could not create umem\n\n");
        exit(0);
    }

    // create xsk socket and assign to network interface queue
    struct xsk_socket_config xsk_config;

    memset(&xsk_config, 0, sizeof(xsk_config));

    xsk_config.rx_size = XSK_RING_CONS__DEFAULT_NUM_DESCS;
    xsk_config.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;
    xsk_config.xdp_flags = XDP_ZEROCOPY;  // force zero copy mode
    xsk_config.bind_flags =
        XDP_USE_NEED_WAKEUP;  // manually wake up the driver when it needs
                              // to do work to send packets
    xsk_config.libbpf_flags = XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD;

    ret = xsk_socket__create(&xsk_, afxdp_ctl.interface_name_, queue_id_, umem_,
                             &recv_queue_, &send_queue_, &xsk_config);
    if (ret) {
        printf("\nerror: could not create xsk socket [%d]\n\n", queue_id);
        exit(0);
    }

    // apply_setsockopt(xsk_socket__fd(socket->xsk));

    // initialize frame allocator
    frame_pool_ = new FramePool(num_frames);
    for (int j = 0; j < num_frames; j++) {
        frame_pool_->push(j * FRAME_SIZE + XDP_PACKET_HEADROOM);
    }

    // We also need to load and update the xsks_map for receiving packets
    struct bpf_map* map = bpf_object__find_map_by_name(
        xdp_program__bpf_obj(afxdp_ctl.program_), "xsks_map");
    int xsk_map_fd = bpf_map__fd(map);
    if (xsk_map_fd < 0) {
        fprintf(stderr, "ERROR: no xsks map found: %s\n", strerror(xsk_map_fd));
        exit(0);
    }
    ret = xsk_socket__update_xskmap(xsk_, xsk_map_fd);
    if (ret) {
        fprintf(stderr, "ERROR: xsks map update fails: %s\n",
                strerror(xsk_map_fd));
        exit(0);
    }

    populate_fill_queue(XSK_RING_PROD__DEFAULT_NUM_DESCS);
}

uint32_t AFXDPSocket::pull_complete_queue() {
    uint32_t complete_index;
    uint32_t completed = xsk_ring_cons__peek(
        &complete_queue_, XSK_RING_CONS__DEFAULT_NUM_DESCS, &complete_index);
    if (completed > 0) {
        for (int i = 0; i < completed; i++) {
            uint64_t frame_offset =
                *xsk_ring_cons__comp_addr(&complete_queue_, complete_index++);
            if (FrameBuf::is_txpulltime_free(frame_offset, umem_buffer_)) {
                frame_pool_->push(frame_offset);
            }
            // Otherwise, the transport layer should handle frame freeing.
        }

        xsk_ring_cons__release(&complete_queue_, completed);
        unpulled_tx_pkts_ -= completed;
    }
    VLOG(2) << "tx complete_queue completed = " << completed
            << " unpulled_tx_pkts = " << unpulled_tx_pkts_;
    return completed;
}

uint32_t AFXDPSocket::send_packet(frame_desc frame) {
    // reserving a slot in the send queue.
    uint32_t send_index;
    // VLOG(3) << "tx send_packets num_frames = " << 1;
    while (xsk_ring_prod__reserve(&send_queue_, 1, &send_index) == 0) {
        LOG(WARNING) << "send_queue_ is full. Busy waiting...";
    }
    struct xdp_desc* desc = xsk_ring_prod__tx_desc(&send_queue_, send_index);
    desc->addr = frame.frame_offset;
    desc->len = frame.frame_len;
    xsk_ring_prod__submit(&send_queue_, 1);
    unpulled_tx_pkts_++;

    if (xsk_ring_prod__needs_wakeup(&send_queue_)) {
        sendto(xsk_socket__fd(xsk_), NULL, 0, MSG_DONTWAIT, NULL, 0);
    }

    return pull_complete_queue();
}

uint32_t AFXDPSocket::send_packets(std::vector<frame_desc>& frames) {
    // reserving slots in the send queue.
    uint32_t send_index;
    auto num_frames = frames.size();
    VLOG(2) << "tx send_packets num_frames = " << num_frames;
    while (xsk_ring_prod__reserve(&send_queue_, num_frames, &send_index) == 0) {
        LOG(WARNING) << "send_queue_ is full. Busy waiting...";
    }
    for (int i = 0; i < num_frames; i++) {
        struct xdp_desc* desc =
            xsk_ring_prod__tx_desc(&send_queue_, send_index++);
        desc->addr = frames[i].frame_offset;
        desc->len = frames[i].frame_len;
    }
    xsk_ring_prod__submit(&send_queue_, num_frames);
    unpulled_tx_pkts_ += num_frames;

    if (xsk_ring_prod__needs_wakeup(&send_queue_)) {
        sendto(xsk_socket__fd(xsk_), NULL, 0, MSG_DONTWAIT, NULL, 0);
    }

    return pull_complete_queue();
}

void AFXDPSocket::populate_fill_queue(uint32_t nb_frames) {
    uint32_t idx_fq;
    int free_slots = xsk_prod_nb_free(&fill_queue_, nb_frames);
    if (free_slots <= 0) return;
    int free_slots2 = xsk_ring_prod__reserve(&fill_queue_, free_slots, &idx_fq);
    for (int i = 0; i < free_slots2; i++) {
        *xsk_ring_prod__fill_addr(&fill_queue_, idx_fq++) = frame_pool_->pop();
    }
    xsk_ring_prod__submit(&fill_queue_, free_slots2);
}

std::vector<AFXDPSocket::frame_desc> AFXDPSocket::recv_packets(
    uint32_t nb_frames) {
    std::vector<AFXDPSocket::frame_desc> frames;

    uint32_t idx_rx, rcvd;
    rcvd = xsk_ring_cons__peek(&recv_queue_, nb_frames, &idx_rx);
    if (!rcvd) return frames;
    VLOG(2) << "rx recv_packets num_frames = " << rcvd;

    populate_fill_queue(rcvd);

    for (int i = 0; i < rcvd; i++) {
        const struct xdp_desc* desc =
            xsk_ring_cons__rx_desc(&recv_queue_, idx_rx++);
        frames.push_back({desc->addr, desc->len});
    }

    xsk_ring_cons__release(&recv_queue_, rcvd);
    return frames;
}

std::string AFXDPSocket::to_string() const {
    std::string s;
    s += Format("\t\t\t[Frame pool] free frames: %u\n", frame_pool_->size());
    return s;
}

void AFXDPSocket::shutdown() {
    // pull_complete_queue to make sure all frames are tx successfully.
    while (unpulled_tx_pkts_) pull_complete_queue();
}

AFXDPSocket::~AFXDPSocket() {
    delete frame_pool_;
    if (xsk_) xsk_socket__delete(xsk_);
    if (umem_) xsk_umem__delete(umem_);
    free(umem_buffer_);
}
}  // namespace uccl
