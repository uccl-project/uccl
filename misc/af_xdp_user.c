/* SPDX-License-Identifier: GPL-2.0 */

#include <arpa/inet.h>
#include <assert.h>
#include <bpf/bpf.h>
#include <config.h>
#include <errno.h>
#include <getopt.h>
#include <linux/icmpv6.h>
#include <linux/if_ether.h>
#include <linux/if_link.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <locale.h>
#include <net/if.h>
#include <ofi_mem.h>
#include <poll.h>
#include <pthread.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_tagged.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <xdp/libxdp.h>
#include <xdp/xsk.h>

#include "../afxdp/common/common_libbpf.h"
#include "../afxdp/common/common_params.h"
#include "../afxdp/common/common_user_bpf_xdp.h"

#define NUM_FRAMES 4096
#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE
#define RX_BATCH_SIZE 64
#define INVALID_UMEM_FRAME UINT64_MAX

static struct xdp_program *prog;
int xsk_map_fd;
bool custom_xsk = false;
struct config cfg = {
    .ifindex = -1,
};

struct xsk_umem_info {
    struct xsk_ring_prod fq;
    struct xsk_ring_cons cq;
    struct xsk_umem *umem;
    void *buffer;
};
struct stats_record {
    uint64_t timestamp;
    uint64_t rx_packets;
    uint64_t rx_bytes;
    uint64_t tx_packets;
    uint64_t tx_bytes;
};
struct xsk_socket_info {
    struct xsk_ring_cons rx;
    struct xsk_ring_prod tx;
    struct xsk_umem_info *umem;
    struct xsk_socket *xsk;

    uint64_t umem_frame_addr[NUM_FRAMES];
    uint32_t umem_frame_free;

    uint32_t outstanding_tx;

    struct stats_record stats;
    struct stats_record prev_stats;
};

static inline __u32 xsk_ring_prod__free(struct xsk_ring_prod *r) {
    r->cached_cons = *r->consumer + r->size;
    return r->cached_cons - r->cached_prod;
}

static const char *__doc__ = "AF_XDP kernel bypass example\n";

static const struct option_wrapper long_options[] = {

    {{"help", no_argument, NULL, 'h'},
     "Show help",
     false},

    {{"dev", required_argument, NULL, 'd'},
     "Operate on device <ifname>",
     "<ifname>",
     true},

    {{"skb-mode", no_argument, NULL, 'S'},
     "Install XDP program in SKB (AKA generic) mode"},

    {{"native-mode", no_argument, NULL, 'N'},
     "Install XDP program in native mode"},

    {{"auto-mode", no_argument, NULL, 'A'},
     "Auto-detect SKB or native mode"},

    {{"force", no_argument, NULL, 'F'},
     "Force install, replacing existing program on interface"},

    {{"copy", no_argument, NULL, 'c'},
     "Force copy mode"},

    {{"zero-copy", no_argument, NULL, 'z'},
     "Force zero-copy mode"},

    {{"queue", required_argument, NULL, 'Q'},
     "Configure interface receive queue for AF_XDP, default=0"},

    {{"poll-mode", no_argument, NULL, 'p'},
     "Use the poll() API waiting for packets to arrive"},

    {{"quiet", no_argument, NULL, 'q'},
     "Quiet mode (no output)"},

    {{"filename", required_argument, NULL, 1},
     "Load program from <file>",
     "<file>"},

    {{"progname", required_argument, NULL, 2},
     "Load program from function <name> in the ELF file",
     "<name>"},

    {{0, 0, NULL, 0}, NULL, false}};

static bool global_exit;

static struct xsk_umem_info *configure_xsk_umem(void *buffer, uint64_t size) {
    struct xsk_umem_info *umem;
    int ret;

    umem = calloc(1, sizeof(*umem));
    if (!umem)
        return NULL;

    ret = xsk_umem__create(&umem->umem, buffer, size, &umem->fq, &umem->cq,
                           NULL);
    if (ret) {
        errno = -ret;
        return NULL;
    }

    umem->buffer = buffer;
    return umem;
}

static uint64_t xsk_alloc_umem_frame(struct xsk_socket_info *xsk) {
    uint64_t frame;
    if (xsk->umem_frame_free == 0)
        return INVALID_UMEM_FRAME;

    frame = xsk->umem_frame_addr[--xsk->umem_frame_free];
    xsk->umem_frame_addr[xsk->umem_frame_free] = INVALID_UMEM_FRAME;
    return frame;
}

static void xsk_free_umem_frame(struct xsk_socket_info *xsk, uint64_t frame) {
    assert(xsk->umem_frame_free < NUM_FRAMES);

    xsk->umem_frame_addr[xsk->umem_frame_free++] = frame;
}

static uint64_t xsk_umem_free_frames(struct xsk_socket_info *xsk) {
    return xsk->umem_frame_free;
}

static struct xsk_socket_info *xsk_configure_socket(struct config *cfg,
                                                    struct xsk_umem_info *umem,
                                                    int queue_id) {
    struct xsk_socket_config xsk_cfg;
    struct xsk_socket_info *xsk_info;
    uint32_t idx;
    int i;
    int ret;
    uint32_t prog_id;

    xsk_info = calloc(1, sizeof(*xsk_info));
    if (!xsk_info)
        return NULL;

    xsk_info->umem = umem;
    xsk_cfg.rx_size = XSK_RING_CONS__DEFAULT_NUM_DESCS;
    xsk_cfg.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;
    xsk_cfg.xdp_flags = cfg->xdp_flags;
    xsk_cfg.bind_flags = cfg->xsk_bind_flags;
    xsk_cfg.libbpf_flags = (custom_xsk) ? XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD : 0;
    ret = xsk_socket__create(&xsk_info->xsk, cfg->ifname,
                             /*cfg->xsk_if_queue*/ queue_id, umem->umem, &xsk_info->rx,
                             &xsk_info->tx, &xsk_cfg);
    if (ret)
        goto error_exit;

    if (custom_xsk) {
        ret = xsk_socket__update_xskmap(xsk_info->xsk, xsk_map_fd);
        if (ret)
            goto error_exit;
    } else {
        /* Getting the program ID must be after the xdp_socket__create() call */
        if (bpf_xdp_query_id(cfg->ifindex, cfg->xdp_flags, &prog_id))
            goto error_exit;
    }

    /* Initialize umem frame allocation */
    for (i = 0; i < NUM_FRAMES; i++)
        xsk_info->umem_frame_addr[i] = i * FRAME_SIZE;

    xsk_info->umem_frame_free = NUM_FRAMES;

    /* Stuff the receive path with buffers, we assume we have enough */
    ret = xsk_ring_prod__reserve(&xsk_info->umem->fq,
                                 XSK_RING_PROD__DEFAULT_NUM_DESCS,
                                 &idx);

    if (ret != XSK_RING_PROD__DEFAULT_NUM_DESCS)
        goto error_exit;

    for (i = 0; i < XSK_RING_PROD__DEFAULT_NUM_DESCS; i++)
        *xsk_ring_prod__fill_addr(&xsk_info->umem->fq, idx++) =
            xsk_alloc_umem_frame(xsk_info);

    xsk_ring_prod__submit(&xsk_info->umem->fq,
                          XSK_RING_PROD__DEFAULT_NUM_DESCS);

    return xsk_info;

error_exit:
    errno = -ret;
    return NULL;
}

static void complete_tx(struct xsk_socket_info *xsk) {
    unsigned int completed;
    uint32_t idx_cq;

    if (!xsk->outstanding_tx)
        return;

    sendto(xsk_socket__fd(xsk->xsk), NULL, 0, MSG_DONTWAIT, NULL, 0);

    /* Collect/free completed TX buffers */
    completed = xsk_ring_cons__peek(&xsk->umem->cq,
                                    XSK_RING_CONS__DEFAULT_NUM_DESCS,
                                    &idx_cq);

    if (completed > 0) {
        for (int i = 0; i < completed; i++)
            xsk_free_umem_frame(xsk,
                                *xsk_ring_cons__comp_addr(&xsk->umem->cq,
                                                          idx_cq++));

        xsk_ring_cons__release(&xsk->umem->cq, completed);
        xsk->outstanding_tx -= completed < xsk->outstanding_tx ? completed : xsk->outstanding_tx;
    }
}

static inline __sum16 csum16_add(__sum16 csum, __be16 addend) {
    uint16_t res = (uint16_t)csum;

    res += (__u16)addend;
    return (__sum16)(res + (res < (__u16)addend));
}

static inline __sum16 csum16_sub(__sum16 csum, __be16 addend) {
    return csum16_add(csum, ~addend);
}

static inline void csum_replace2(__sum16 *sum, __be16 old, __be16 new) {
    *sum = ~csum16_add(csum16_sub(~(*sum), old), new);
}

static bool process_packet(struct xsk_socket_info *xsk,
                           uint64_t addr, uint32_t len) {
    uint8_t *pkt = xsk_umem__get_data(xsk->umem->buffer, addr);

    int ret;
    uint32_t tx_idx = 0;
    uint8_t tmp_mac[ETH_ALEN];
    __be32 tmp_ip;
    uint16_t tmp_port;
    struct ethhdr *eth = (struct ethhdr *)pkt;
    struct iphdr *iph = (struct iphdr *)(eth + 1);
    struct udphdr *udph = (struct udphdr *)((unsigned char *)iph + (iph->ihl << 2));

    memcpy(tmp_mac, eth->h_dest, ETH_ALEN);
    memcpy(eth->h_dest, eth->h_source, ETH_ALEN);
    memcpy(eth->h_source, tmp_mac, ETH_ALEN);

    memcpy(&tmp_ip, &iph->saddr, sizeof(tmp_ip));
    memcpy(&iph->saddr, &iph->daddr, sizeof(tmp_ip));
    memcpy(&iph->daddr, &tmp_ip, sizeof(tmp_ip));

    // Swap source and destination port
    tmp_port = udph->source;
    udph->source = udph->dest;
    udph->dest = tmp_port;

    // Causing transmission erros, but why?
    // iph->check = compute_ip_checksum(iph);
    udph->check = 0;

    /* Here we sent the packet out of the receive port. Note that
     * we allocate one entry and schedule it. Your design would be
     * faster if you do batch processing/transmission */

    ret = xsk_ring_prod__reserve(&xsk->tx, 1, &tx_idx);
    if (ret != 1) {
        /* No more transmit slots, drop the packet */
        return false;
    }

    xsk_ring_prod__tx_desc(&xsk->tx, tx_idx)->addr = addr;
    xsk_ring_prod__tx_desc(&xsk->tx, tx_idx)->len = len;
    xsk_ring_prod__submit(&xsk->tx, 1);
    xsk->outstanding_tx++;

    xsk->stats.tx_bytes += len;
    xsk->stats.tx_packets++;
    return true;
}

static void handle_receive_packets(struct xsk_socket_info *xsk) {
    unsigned int rcvd, stock_frames, i;
    uint32_t idx_rx = 0, idx_fq = 0;
    int ret;

    rcvd = xsk_ring_cons__peek(&xsk->rx, RX_BATCH_SIZE, &idx_rx);
    if (!rcvd)
        return;

    /* Stuff the ring with as much frames as possible */
    stock_frames = xsk_prod_nb_free(&xsk->umem->fq,
                                    xsk_umem_free_frames(xsk));

    if (stock_frames > 0) {
        ret = xsk_ring_prod__reserve(&xsk->umem->fq, stock_frames,
                                     &idx_fq);

        /* This should not happen, but just in case */
        while (ret != stock_frames)
            ret = xsk_ring_prod__reserve(&xsk->umem->fq, rcvd,
                                         &idx_fq);

        for (i = 0; i < stock_frames; i++)
            *xsk_ring_prod__fill_addr(&xsk->umem->fq, idx_fq++) =
                xsk_alloc_umem_frame(xsk);

        xsk_ring_prod__submit(&xsk->umem->fq, stock_frames);
    }

    /* Process received packets */
    for (i = 0; i < rcvd; i++) {
        uint64_t addr = xsk_ring_cons__rx_desc(&xsk->rx, idx_rx)->addr;
        uint32_t len = xsk_ring_cons__rx_desc(&xsk->rx, idx_rx++)->len;

        if (!process_packet(xsk, addr, len))
            xsk_free_umem_frame(xsk, addr);

        xsk->stats.rx_bytes += len;
    }

    xsk_ring_cons__release(&xsk->rx, rcvd);
    xsk->stats.rx_packets += rcvd;

    /* Do we need to wake up the kernel for transmission */
    complete_tx(xsk);
}

static void rx_and_process(struct config *cfg,
                           struct xsk_socket_info *xsk_socket) {
    struct pollfd fds[2];
    int ret, nfds = 1;

    memset(fds, 0, sizeof(fds));
    fds[0].fd = xsk_socket__fd(xsk_socket->xsk);
    fds[0].events = POLLIN;

    while (!global_exit) {
        if (cfg->xsk_poll_mode) {
            ret = poll(fds, nfds, -1);
            if (ret <= 0 || ret > 1)
                continue;
        }
        handle_receive_packets(xsk_socket);
    }
}

#define NANOSEC_PER_SEC 1000000000 /* 10^9 */
static uint64_t gettime(void) {
    struct timespec t;
    int res;

    res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res < 0) {
        fprintf(stderr, "Error with gettimeofday! (%i)\n", res);
        exit(EXIT_FAIL);
    }
    return (uint64_t)t.tv_sec * NANOSEC_PER_SEC + t.tv_nsec;
}

static double calc_period(struct stats_record *r, struct stats_record *p) {
    double period_ = 0;
    __u64 period = 0;

    period = r->timestamp - p->timestamp;
    if (period > 0)
        period_ = ((double)period / NANOSEC_PER_SEC);

    return period_;
}

static void stats_print(struct stats_record *stats_rec,
                        struct stats_record *stats_prev) {
    uint64_t packets, bytes;
    double period;
    double pps; /* packets per sec */
    double bps; /* bits per sec */

    char *fmt =
        "%-12s %'11lld pkts (%'10.0f pps)"
        " %'11lld Kbytes (%'6.0f Mbits/s)"
        " period:%f\n";

    period = calc_period(stats_rec, stats_prev);
    if (period == 0)
        period = 1;

    packets = stats_rec->rx_packets - stats_prev->rx_packets;
    pps = packets / period;

    bytes = stats_rec->rx_bytes - stats_prev->rx_bytes;
    bps = (bytes * 8) / period / 1000000;

    printf(fmt, "AF_XDP RX:", stats_rec->rx_packets, pps,
           stats_rec->rx_bytes / 1000, bps,
           period);

    packets = stats_rec->tx_packets - stats_prev->tx_packets;
    pps = packets / period;

    bytes = stats_rec->tx_bytes - stats_prev->tx_bytes;
    bps = (bytes * 8) / period / 1000000;

    printf(fmt, "       TX:", stats_rec->tx_packets, pps,
           stats_rec->tx_bytes / 1000, bps,
           period);

    printf("\n");
}

static void *stats_poll(void *arg) {
    unsigned int interval = 2;
    struct xsk_socket_info *xsk = arg;
    static struct stats_record previous_stats = {0};

    previous_stats.timestamp = gettime();

    /* Trick to pretty printf with thousands separators use %' */
    setlocale(LC_NUMERIC, "en_US");

    while (!global_exit) {
        sleep(interval);
        xsk->stats.timestamp = gettime();
        stats_print(&xsk->stats, &previous_stats);
        previous_stats = xsk->stats;
    }
    return NULL;
}

static void exit_application(int signal) {
    int err;

    cfg.unload_all = true;
    err = do_unload(&cfg);
    if (err) {
        fprintf(stderr, "Couldn't detach XDP program on iface '%s' : (%d)\n",
                cfg.ifname, err);
    }

    signal = signal;
    global_exit = true;
}

#define PP_PRINTERR(call, retv)                                     \
    fprintf(stderr, "%s(): %s:%-4d, ret=%d (%s)\n", call, __FILE__, \
            __LINE__, (int)retv, fi_strerror((int)-retv))

#define PP_ERR(fmt, ...)                                          \
    fprintf(stderr, "[%s] %s:%-4d: " fmt "\n", "error", __FILE__, \
            __LINE__, ##__VA_ARGS__)

static int print_short_info(struct fi_info *info) {
    struct fi_info *cur;

    for (cur = info; cur; cur = cur->next) {
        printf("provider: %s\n", cur->fabric_attr->prov_name);
        printf("    fabric: %s\n", cur->fabric_attr->name),
            printf("    domain: %s\n", cur->domain_attr->name),
            printf("    version: %d.%d\n", FI_MAJOR(cur->fabric_attr->prov_version),
                   FI_MINOR(cur->fabric_attr->prov_version));
        printf("    type: %s\n", fi_tostr(&cur->ep_attr->type, FI_TYPE_EP_TYPE));
        printf("    protocol: %s\n", fi_tostr(&cur->ep_attr->protocol, FI_TYPE_PROTOCOL));
    }
    return EXIT_SUCCESS;
}

uint16_t oob_dst_port = 8890;

static void *send_fi_name(void *arg) {
    // Get fi_name.
    int ret = EXIT_SUCCESS;

    struct fi_info *fi_pep, *fi, *hints;
    struct fid_fabric *fabric;
    struct fi_eq_attr eq_attr;
    struct fid_eq *eq;
    struct fid_domain *domain;
    struct fid_ep *ep;
    struct fi_cq_attr cq_attr;
    struct fid_cq *txcq, *rxcq;
    struct fi_av_attr av_attr;
    struct fid_av *av;
    fi_addr_t local_fi_addr, remote_fi_addr;
    void *local_name, *rem_name;
    struct fi_context tx_ctx[2], rx_ctx[2];

    hints = fi_allocinfo();
    if (!hints)
        return EXIT_FAILURE;

    hints->ep_attr->type = FI_EP_DGRAM;
    hints->caps = FI_MSG;
    hints->mode = FI_CONTEXT;
    hints->domain_attr->mr_mode = FI_MR_UNSPEC;
    hints->domain_attr->name = "enp39s0";
    // hints->fabric_attr->name = "172.31.64.0/20";
    hints->fabric_attr->prov_name = "udp";  // "sockets" -> TCP

    ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                     NULL, NULL, 0, hints, &fi);
    if (ret) {
        PP_PRINTERR("fi_getinfo", ret);
        return ret;
    }
    print_short_info(fi);

    ret = fi_fabric(fi->fabric_attr, &(fabric), NULL);
    if (ret) {
        PP_PRINTERR("fi_fabric", ret);
        return ret;
    }

    ret = fi_eq_open(fabric, &(eq_attr), &(eq), NULL);
    if (ret) {
        PP_PRINTERR("fi_eq_open", ret);
        return ret;
    }

    ret = fi_domain(fabric, fi, &(domain), NULL);
    if (ret) {
        PP_PRINTERR("fi_domain", ret);
        return ret;
    }

    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.wait_obj = FI_WAIT_NONE;
    cq_attr.size = 1024;
    ret = fi_cq_open(domain, &(cq_attr), &(txcq), &(txcq));
    if (ret) {
        PP_PRINTERR("fi_cq_open", ret);
        return ret;
    }
    ret = fi_cq_open(domain, &(cq_attr), &(rxcq), &(rxcq));
    if (ret) {
        PP_PRINTERR("fi_cq_open", ret);
        return ret;
    }

    av_attr.type = FI_AV_MAP;
    ret = fi_av_open(domain, &(av_attr), &(av), NULL);
    if (ret) {
        PP_PRINTERR("fi_av_open", ret);
        return ret;
    }

    ret = fi_endpoint(domain, fi, &(ep), NULL);
    if (ret) {
        PP_PRINTERR("fi_endpoint", ret);
        return ret;
    }

#define PP_EP_BIND(ep, fd, flags)                        \
    do {                                                 \
        int ret;                                         \
        if ((fd)) {                                      \
            ret = fi_ep_bind((ep), &(fd)->fid, (flags)); \
            if (ret) {                                   \
                PP_PRINTERR("fi_ep_bind", ret);          \
                return ret;                              \
            }                                            \
        }                                                \
    } while (0)

    PP_EP_BIND(ep, eq, 0);
    PP_EP_BIND(ep, av, 0);
    PP_EP_BIND(ep, txcq, FI_TRANSMIT);
    PP_EP_BIND(ep, rxcq, FI_RECV);

    ret = fi_enable(ep);
    if (ret) {
        PP_PRINTERR("fi_enable", ret);
        return ret;
    }

    size_t addrlen = 0;
    local_name = NULL;
    ret = fi_getname(&ep->fid, local_name, &addrlen);
    if ((ret != -FI_ETOOSMALL) || (addrlen <= 0)) {
        PP_ERR("fi_getname didn't return length\n");
        return -EMSGSIZE;
    }
    local_name = malloc(addrlen);

    ret = fi_getname(&ep->fid, local_name, &addrlen);
    if (ret) {
        PP_PRINTERR("fi_getname", ret);
        return ret;
    }

    char ep_name_buf[128];
    size_t size = 0;
    fi_av_straddr(av, local_name, NULL, &size);

    fi_av_straddr(av, local_name, ep_name_buf, &size);

    printf("OFI EP prov %s name %s straddr %s\n",
           fi->fabric_attr->prov_name,
           fi->fabric_attr->name, ep_name_buf);

    int sockfd, connfd, len;
    struct sockaddr_in servaddr, cli;

    // socket create and verification
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    } else
        printf("Socket successfully created..\n");
    bzero(&servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(oob_dst_port);

    // Binding newly created socket to given IP and verification
    if ((bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr))) != 0) {
        printf("socket bind failed...\n");
        exit(0);
    } else
        printf("Socket successfully binded..\n");

    int flags = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &flags, sizeof(int)) != 0) {
        printf("socket reuse failed...\n");
        exit(0);
    }

    // Now server is ready to listen and verification
    if ((listen(sockfd, 5)) != 0) {
        printf("Listen failed...\n");
        exit(0);
    } else
        printf("Server listening..\n");
    len = sizeof(cli);

    while (1) {
        // Accept the data packet from client and verification
        connfd = accept(sockfd, (struct sockaddr *)&cli, &len);
        if (connfd < 0) {
            printf("server accept failed...\n");
            exit(0);
        } else
            printf("server accept the client...\n");

        ret = write(connfd, &addrlen, sizeof(addrlen));
        if (ret != sizeof(addrlen)) {
            printf("error write addrlen\n");
            exit(0);
        }
        ret = write(connfd, local_name, addrlen);
        if (ret != addrlen) {
            printf("error write local_name\n");
            exit(0);
        }
        close(connfd);
    }
    return ret;
}

// Avoid putting it into main to cause stack overflow.
char errmsg[1024];

int main(int argc, char **argv) {
    int ret;
    void *packet_buffer;
    uint64_t packet_buffer_size;
    DECLARE_LIBBPF_OPTS(bpf_object_open_opts, opts);
    DECLARE_LIBXDP_OPTS(xdp_program_opts, xdp_opts, 0);
    struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};
    struct xsk_umem_info *umem;
    struct xsk_socket_info *xsk_socket;
    pthread_t stats_poll_thread;
    int err;

    /* Global shutdown handler */
    signal(SIGINT, exit_application);

    /* Cmdline options can change progname */
    parse_cmdline_args(argc, argv, long_options, &cfg, __doc__);

    /* Required option */
    if (cfg.ifindex == -1) {
        fprintf(stderr, "ERROR: Required option --dev missing\n\n");
        usage(argv[0], __doc__, long_options, (argc == 1));
        return EXIT_FAIL_OPTION;
    }

    /* Load custom program if configured */
    if (cfg.filename[0] != 0) {
        struct bpf_map *map;

        custom_xsk = true;
        xdp_opts.open_filename = cfg.filename;
        xdp_opts.prog_name = cfg.progname;
        xdp_opts.opts = &opts;

        if (cfg.progname[0] != 0) {
            xdp_opts.open_filename = cfg.filename;
            xdp_opts.prog_name = cfg.progname;
            xdp_opts.opts = &opts;

            prog = xdp_program__create(&xdp_opts);
        } else {
            prog = xdp_program__open_file(cfg.filename,
                                          NULL, &opts);
        }
        err = libxdp_get_error(prog);
        if (err) {
            libxdp_strerror(err, errmsg, sizeof(errmsg));
            fprintf(stderr, "ERR: loading program: %s\n", errmsg);
            return err;
        }

        err = xdp_program__attach(prog, cfg.ifindex, cfg.attach_mode, 0);
        if (err) {
            libxdp_strerror(err, errmsg, sizeof(errmsg));
            fprintf(stderr, "Couldn't attach XDP program on iface '%s' : %s (%d)\n",
                    cfg.ifname, errmsg, err);
            return err;
        }

        /* We also need to load the xsks_map */
        map = bpf_object__find_map_by_name(xdp_program__bpf_obj(prog), "xsks_map");
        xsk_map_fd = bpf_map__fd(map);
        if (xsk_map_fd < 0) {
            fprintf(stderr, "ERROR: no xsks map found: %s\n",
                    strerror(xsk_map_fd));
            exit(EXIT_FAILURE);
        }
    }

    /* Allow unlimited locking of memory, so all memory needed for packet
     * buffers can be locked.
     */
    if (setrlimit(RLIMIT_MEMLOCK, &rlim)) {
        fprintf(stderr, "ERROR: setrlimit(RLIMIT_MEMLOCK) \"%s\"\n",
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for NUM_FRAMES of the default XDP frame size */
    packet_buffer_size = NUM_FRAMES * FRAME_SIZE;
    if (posix_memalign(&packet_buffer,
                       getpagesize(), /* PAGE_SIZE aligned */
                       packet_buffer_size)) {
        fprintf(stderr, "ERROR: Can't allocate buffer memory \"%s\"\n",
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Initialize shared packet_buffer for umem usage */
    umem = configure_xsk_umem(packet_buffer, packet_buffer_size);
    if (umem == NULL) {
        fprintf(stderr, "ERROR: Can't create umem \"%s\"\n",
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Open and configure the AF_XDP (xsk) socket */
    xsk_socket = xsk_configure_socket(&cfg, umem, 0);
    if (xsk_socket == NULL) {
        fprintf(stderr, "ERROR: Can't setup AF_XDP socket \"%s\"\n",
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Start thread to do statistics display */
    if (verbose) {
        ret = pthread_create(&stats_poll_thread, NULL, stats_poll,
                             xsk_socket);
        if (ret) {
            fprintf(stderr,
                    "ERROR: Failed creating statistics thread "
                    "\"%s\"\n",
                    strerror(errno));
            exit(EXIT_FAILURE);
        }
    }

    pthread_t send_fi_name_thread;
    pthread_create(&send_fi_name_thread, NULL, send_fi_name,
                   NULL);

    /* Receive and count packets than drop them */
    rx_and_process(&cfg, xsk_socket);

    /* Cleanup */
    xsk_socket__delete(xsk_socket->xsk);
    xsk_umem__delete(umem->umem);

    return EXIT_OK;
}
