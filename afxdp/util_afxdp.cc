#include "util_afxdp.h"

#include "transport_config.h"

namespace uccl {

AFXDPFactory afxdp_ctl;

void AFXDPFactory::init(const char *interface_name, uint64_t num_frames,
                        const char *ebpf_filename, const char *section_name) {
    // TODO(yang): negotiate with afxdp daemon to load the specified program
    strcpy(afxdp_ctl.interface_name_, interface_name);

    struct sockaddr_un addr;
    // Create a UNIX domain socket to receive file descriptors
    if ((afxdp_ctl.client_sock_ = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, SOCKET_PATH);

    if (connect(afxdp_ctl.client_sock_, (struct sockaddr *)&addr,
                sizeof(addr)) == -1) {
        perror("connect");
        exit(EXIT_FAILURE);
    }
    // Receive the file descriptor for the UMEM
    DCHECK(receive_fd(afxdp_ctl.client_sock_, &afxdp_ctl.umem_fd_) == 0);

    afxdp_ctl.umem_size_ = num_frames * FRAME_SIZE;
    afxdp_ctl.umem_buffer_ = attach_shm(SHM_NAME, afxdp_ctl.umem_size_);
    if (afxdp_ctl.umem_buffer_ == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    afxdp_ctl.num_frames_ = num_frames;
}

AFXDPSocket *AFXDPFactory::CreateSocket(int queue_id) {
    auto socket = new AFXDPSocket(queue_id);
    std::lock_guard<std::mutex> lock(afxdp_ctl.socket_q_lock_);
    afxdp_ctl.socket_q_.push_back(socket);
    return socket;
}

void AFXDPFactory::shutdown() {
    // eBPF program detaching is done by the afxdp daemon

    std::lock_guard<std::mutex> lock(afxdp_ctl.socket_q_lock_);
    for (auto socket : afxdp_ctl.socket_q_) {
        delete socket;
    }
    afxdp_ctl.socket_q_.clear();
}

AFXDPSocket::AFXDPSocket(int queue_id)
    : unpulled_tx_pkts_(0), fill_queue_entries_(0) {
    // TODO(yang): negotiate with afxdp daemon for queue_id and num_frames.

    queue_id_ = queue_id;

    // initialize queues, or misterious queue sync problems will happen
    memset(&recv_queue_, 0, sizeof(recv_queue_));
    memset(&send_queue_, 0, sizeof(send_queue_));
    memset(&complete_queue_, 0, sizeof(complete_queue_));
    memset(&fill_queue_, 0, sizeof(fill_queue_));

    // Step1: receive the file descriptors for AF_XDP socket
    DCHECK(receive_fd(afxdp_ctl.client_sock_, &xsk_fd_) == 0);

    // Step2: map UMEM and build four rings for the AF_XDP socket
    int ret = create_afxdp_socket();
    CHECK_EQ(ret, 0) << "xsk_socket__create_shared failed, " << ret;

    LOG(INFO) << "AF_XDP socket successfully shared.";

    // apply_setsockopt(xsk_fd_);

    // xsks_map for receiving packets has been updated by afxdp daemon.

    populate_fill_queue(XSK_RING_PROD__DEFAULT_NUM_DESCS);
}

void AFXDPSocket::xsk_mmap_offsets_v1(struct xdp_mmap_offsets *off) {
    struct xdp_mmap_offsets_v1 off_v1;

    /* getsockopt on a kernel <= 5.3 has no flags fields.
     * Copy over the offsets to the correct places in the >=5.4 format
     * and put the flags where they would have been on that kernel.
     */
    memcpy(&off_v1, off, sizeof(off_v1));

    off->rx.producer = off_v1.rx.producer;
    off->rx.consumer = off_v1.rx.consumer;
    off->rx.desc = off_v1.rx.desc;
    off->rx.flags = off_v1.rx.consumer + sizeof(__u32);

    off->tx.producer = off_v1.tx.producer;
    off->tx.consumer = off_v1.tx.consumer;
    off->tx.desc = off_v1.tx.desc;
    off->tx.flags = off_v1.tx.consumer + sizeof(__u32);

    off->fr.producer = off_v1.fr.producer;
    off->fr.consumer = off_v1.fr.consumer;
    off->fr.desc = off_v1.fr.desc;
    off->fr.flags = off_v1.fr.consumer + sizeof(__u32);

    off->cr.producer = off_v1.cr.producer;
    off->cr.consumer = off_v1.cr.consumer;
    off->cr.desc = off_v1.cr.desc;
    off->cr.flags = off_v1.cr.consumer + sizeof(__u32);
}

int AFXDPSocket::xsk_get_mmap_offsets(int fd, struct xdp_mmap_offsets *off) {
    socklen_t optlen;
    int err;

    optlen = sizeof(*off);
    err = getsockopt(fd, SOL_XDP, XDP_MMAP_OFFSETS, off, &optlen);
    if (err) return err;

    if (optlen == sizeof(*off)) return 0;

    if (optlen == sizeof(struct xdp_mmap_offsets_v1)) {
        xsk_mmap_offsets_v1(off);
        return 0;
    }

    return -1;
}

void AFXDPSocket::destroy_afxdp_socket() {
    if (rx_map_ && rx_map_ != MAP_FAILED) munmap(rx_map_, rx_map_size_);
    if (tx_map_ && tx_map_ != MAP_FAILED) munmap(tx_map_, tx_map_size_);
    if (fill_map_ && fill_map_ != MAP_FAILED) munmap(fill_map_, fill_map_size_);
    if (comp_map_ && comp_map_ != MAP_FAILED) munmap(comp_map_, comp_map_size_);
    if (umem_buffer_ && umem_buffer_ != MAP_FAILED) {
        detach_shm(umem_buffer_, umem_size_);
    }
}

/**
 * @brief: Manually map UMEM and build four rings for a AF_XDP socket
 * @note: (RX/TX/FILL/COMP_RING_SIZE, NUM_FRAMES, FRAME_SIZE) need negotiating
 * with privileged processes
 */
int AFXDPSocket::create_afxdp_socket() {
    struct xsk_ring_cons *rx = &recv_queue_;
    struct xsk_ring_prod *tx = &send_queue_;
    struct xsk_ring_prod *fill = &fill_queue_;
    struct xsk_ring_cons *comp = &complete_queue_;
    struct xdp_mmap_offsets off;

    /* Map UMEM */
    umem_fd_ = afxdp_ctl.umem_fd_;
    umem_size_ = afxdp_ctl.umem_size_;
    umem_buffer_ = afxdp_ctl.umem_buffer_;

    // initialize frame allocator
    uint64_t frame_pool_size = afxdp_ctl.num_frames_ / NUM_QUEUES;
    frame_pool_ = new SharedPool<uint64_t, /*Sync=*/false>(frame_pool_size);
    uint64_t frame_pool_offset = FRAME_SIZE * frame_pool_size * queue_id_;
    for (uint64_t i = 0; i < frame_pool_size; i++) {
        auto frame_offset =
            i * FRAME_SIZE + XDP_PACKET_HEADROOM + frame_pool_offset;
        push_frame(frame_offset);
    }

    /* Get offsets for the following mmap */
    if (xsk_get_mmap_offsets(umem_fd_, &off)) {
        perror("xsk_get_mmap_offsets failed");
        goto out;
    }

    /* RX Ring */
    rx_map_size_ = off.rx.desc + RX_RING_SIZE * sizeof(struct xdp_desc);
    rx_map_ = mmap(NULL, rx_map_size_, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_POPULATE, xsk_fd_, XDP_PGOFF_RX_RING);
    if (rx_map_ == MAP_FAILED) {
        perror("rx mmap failed");
        goto out;
    }
    rx->mask = RX_RING_SIZE - 1;
    rx->size = RX_RING_SIZE;
    rx->producer = (uint32_t *)((char *)rx_map_ + off.rx.producer);
    rx->consumer = (uint32_t *)((char *)rx_map_ + off.rx.consumer);
    rx->flags = (uint32_t *)((char *)rx_map_ + off.rx.flags);
    rx->ring = rx_map_ + off.rx.desc;
    rx->cached_prod = *rx->producer;
    rx->cached_cons = *rx->consumer;

    /* TX Ring */
    tx_map_size_ = off.tx.desc + TX_RING_SIZE * sizeof(struct xdp_desc);
    tx_map_ = mmap(NULL, tx_map_size_, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_POPULATE, xsk_fd_, XDP_PGOFF_TX_RING);
    if (tx_map_ == MAP_FAILED) {
        perror("tx mmap failed");
        goto out;
    }
    tx->mask = TX_RING_SIZE - 1;
    tx->size = TX_RING_SIZE;
    tx->producer = (uint32_t *)((char *)tx_map_ + off.tx.producer);
    tx->consumer = (uint32_t *)((char *)tx_map_ + off.tx.consumer);
    tx->flags = (uint32_t *)((char *)tx_map_ + off.tx.flags);
    tx->ring = tx_map_ + off.tx.desc;
    tx->cached_prod = *tx->producer;
    tx->cached_cons = *tx->consumer + TX_RING_SIZE;

    /* Fill Ring */
    fill_map_size_ = off.fr.desc + FILL_RING_SIZE * sizeof(__u64);
    fill_map_ =
        mmap(NULL, fill_map_size_, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, xsk_fd_, XDP_UMEM_PGOFF_FILL_RING);
    if (fill_map_ == MAP_FAILED) {
        perror("fill mmap failed");
        goto out;
    }
    fill->mask = FILL_RING_SIZE - 1;
    fill->size = FILL_RING_SIZE;
    fill->producer = (uint32_t *)((char *)fill_map_ + off.fr.producer);
    fill->consumer = (uint32_t *)((char *)fill_map_ + off.fr.consumer);
    fill->flags = (uint32_t *)((char *)fill_map_ + off.fr.flags);
    fill->ring = fill_map_ + off.fr.desc;
    fill->cached_cons = FILL_RING_SIZE;

    /* Completion Ring */
    comp_map_size_ = off.cr.desc + COMP_RING_SIZE * sizeof(__u64);
    comp_map_ = mmap(NULL, comp_map_size_, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, xsk_fd_,
                     XDP_UMEM_PGOFF_COMPLETION_RING);
    if (comp_map_ == MAP_FAILED) {
        perror("comp mmap failed");
        goto out;
    }

    comp->mask = COMP_RING_SIZE - 1;
    comp->size = COMP_RING_SIZE;
    comp->producer = (uint32_t *)((char *)comp_map_ + off.cr.producer);
    comp->consumer = (uint32_t *)((char *)comp_map_ + off.cr.consumer);
    comp->flags = (uint32_t *)((char *)comp_map_ + off.cr.flags);
    comp->ring = comp_map_ + off.cr.desc;

    return 0;
out:
    destroy_afxdp_socket();
    return -1;
}

uint32_t AFXDPSocket::pull_complete_queue() {
    uint32_t idx_cq, completed;
    completed = xsk_ring_cons__peek(&complete_queue_,
                                    XSK_RING_CONS__DEFAULT_NUM_DESCS, &idx_cq);
    if (completed > 0) {
        for (int i = 0; i < completed; i++) {
            uint64_t frame_offset =
                *xsk_ring_cons__comp_addr(&complete_queue_, idx_cq++);
            if (FrameBuf::is_txpulltime_free(frame_offset, umem_buffer_)) {
                push_frame(frame_offset);
                /**
                 * Yang: I susspect this is a bug that AWS ENA driver will pull
                 * the same frame multiple times. A temp fix is we mark it as
                 * not txpulltime free, so that the next time polling will not
                 * double free it.
                 */
                FrameBuf::mark_not_txpulltime_free(frame_offset, umem_buffer_);
            }
            // In other cases, the transport layer should handle frame freeing.
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
    VLOG(2) << "tx send_packets num_frames = " << 1;
    while (xsk_ring_prod__reserve(&send_queue_, 1, &send_index) != 1) {
        LOG_EVERY_N(WARNING, 1000000)
            << "send_queue is full. Busy waiting... unpulled_tx_pkts "
            << unpulled_tx_pkts_ << " send_queue_free_entries "
            << send_queue_free_entries(1);
        kick_tx();
    }
    struct xdp_desc *desc = xsk_ring_prod__tx_desc(&send_queue_, send_index);
    desc->addr = frame.frame_offset;
    desc->len = frame.frame_len;
    xsk_ring_prod__submit(&send_queue_, 1);
    unpulled_tx_pkts_++;

    uint32_t pull_tx_pkts = 0;
    do {
        kick_tx();
        pull_tx_pkts += pull_complete_queue();
    } while (unpulled_tx_pkts_ > FILL_RING_SIZE / 2);

    return pull_tx_pkts;
}

uint32_t AFXDPSocket::send_packets(std::vector<frame_desc> &frames) {
    // reserving slots in the send queue.
    uint32_t send_index;
    auto num_frames = frames.size();
    VLOG(2) << "tx send_packets num_frames = " << num_frames;
    while (xsk_ring_prod__reserve(&send_queue_, num_frames, &send_index) !=
           num_frames) {
        LOG_EVERY_N(WARNING, 1000000)
            << "send_queue is full. Busy waiting... unpulled_tx_pkts "
            << unpulled_tx_pkts_ << " send_queue_free_entries "
            << send_queue_free_entries(num_frames) << " num_frames "
            << num_frames;
        kick_tx();
    }
    for (int i = 0; i < num_frames; i++) {
        struct xdp_desc *desc =
            xsk_ring_prod__tx_desc(&send_queue_, send_index++);
        desc->addr = frames[i].frame_offset;
        desc->len = frames[i].frame_len;
    }
    xsk_ring_prod__submit(&send_queue_, num_frames);
    unpulled_tx_pkts_ += num_frames;

    uint32_t pull_tx_pkts = 0;
    do {
        kick_tx();
        pull_tx_pkts += pull_complete_queue();
    } while (unpulled_tx_pkts_ > FILL_RING_SIZE / 2);

    return pull_tx_pkts;
}

void AFXDPSocket::populate_fill_queue(uint32_t nb_frames) {
    // TODO(yang): figure out why cloudlab needs xsk_prod_nb_free().
#ifdef AWS_ENA
    auto stock_frames = nb_frames;
#else
    auto stock_frames = xsk_prod_nb_free(&fill_queue_, nb_frames);
#endif
    if (stock_frames <= 0) return;

    uint32_t idx_fq;
    int ret = xsk_ring_prod__reserve(&fill_queue_, stock_frames, &idx_fq);
    if (ret <= 0) return;

    for (int i = 0; i < ret; i++)
        *xsk_ring_prod__fill_addr(&fill_queue_, idx_fq++) = pop_frame();

    fill_queue_entries_ += ret;
    VLOG(2) << "afxdp reserved fill_queue slots = " << ret
            << " fill_queue_entries_ = " << fill_queue_entries_;

    xsk_ring_prod__submit(&fill_queue_, ret);
}

std::vector<AFXDPSocket::frame_desc> AFXDPSocket::recv_packets(
    uint32_t nb_frames) {
    std::vector<AFXDPSocket::frame_desc> frames;
    uint32_t idx_rx, rcvd;
    rcvd = xsk_ring_cons__peek(&recv_queue_, nb_frames, &idx_rx);
    if (!rcvd) {
        kick_rx();
        return frames;
    }
    fill_queue_entries_ -= rcvd;
    VLOG(2) << "rx recv_packets num_frames = " << rcvd
            << " fill_queue_entries_ = " << fill_queue_entries_;

    for (int i = 0; i < rcvd; i++) {
        const struct xdp_desc *desc =
            xsk_ring_cons__rx_desc(&recv_queue_, idx_rx++);
        frames.push_back({desc->addr, desc->len});
    }

    xsk_ring_cons__release(&recv_queue_, rcvd);

    // Do filling even it is a small batch to control tail latency.
    populate_fill_queue(rcvd);

    return frames;
}

std::string AFXDPSocket::to_string() const {
    std::string s;
    s += Format(
        "\n\t\t[AFXDP] free frames: %u, unpulled tx pkts: %u, fill queue "
        "entries: %u",
        frame_pool_->size(), unpulled_tx_pkts_, fill_queue_entries_);
    return s;
}

void AFXDPSocket::shutdown() {
    // pull_complete_queue to make sure all frames are tx successfully.
    while (unpulled_tx_pkts_) {
        kick_tx();
        pull_complete_queue();
    }
}

AFXDPSocket::~AFXDPSocket() {
    destroy_afxdp_socket();
    delete frame_pool_;
}
}  // namespace uccl
