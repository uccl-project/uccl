#include "util_afxdp.h"

namespace uccl {

AFXDPFactory afxdp_ctl;

void AFXDPFactory::init(const char* interface_name,
                        const char* xdp_program_path) {
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

    // load the client_xdp program and attach it to the network interface
    LOG(INFO) << "loading client_xdp...";

    afxdp_ctl.program_ =
        xdp_program__open_file("client_xdp.o", "client_xdp", NULL);
    CHECK(!libxdp_get_error(afxdp_ctl.program_))
        << "error: could not load client_xdp program";

    LOG(INFO) << "client_xdp loaded successfully.";
    LOG(INFO) << "attaching client_xdp to network interface";

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
            CHECK(false) << "error: failed to attach client_xdp program to "
                            "interface";
        }
    }

    // allow unlimited locking of memory, so all memory needed for packet
    // buffers can be locked
    struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};
    CHECK(setrlimit(RLIMIT_MEMLOCK, &rlim)) << "error: could not setrlimit";
}

AFXDPSocket* AFXDPFactory::createSocket(int queue_id, int num_frames) {
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

    for (auto socket : afxdp_ctl.socket_q_) {
        delete socket;
    }
    afxdp_ctl.socket_q_.clear();
}

AFXDPSocket::AFXDPSocket(int queue_id, int num_frames) {
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
        frame_pool_->push(j * FRAME_SIZE);
    }
}

AFXDPSocket::~AFXDPSocket() {
    delete frame_pool_;
    if (xsk_) xsk_socket__delete(xsk_);
    if (umem_) xsk_umem__delete(umem_);
    free(umem_buffer_);
}
}  // namespace uccl
