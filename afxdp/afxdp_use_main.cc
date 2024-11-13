#include <assert.h>
#include <fcntl.h>
#include <linux/if_xdp.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <netinet/ether.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <xdp/xsk.h>

#include <atomic>
#include <thread>

#include "transport_config.h"

#define IF_NAME DEV_DEFAULT
#define SHM_NAME "UMEM_SHM"
#define QUEUE_ID 0
#define FILL_RING_SIZE                  \
    (XSK_RING_PROD__DEFAULT_NUM_DESCS * \
     2)  // recommened to be RX_RING_SIZE + NIC RING SIZE
#define COMP_RING_SIZE XSK_RING_CONS__DEFAULT_NUM_DESCS
#define TX_RING_SIZE XSK_RING_PROD__DEFAULT_NUM_DESCS
#define RX_RING_SIZE XSK_RING_CONS__DEFAULT_NUM_DESCS

#define SOCKET_PATH "/tmp/privileged_socket"
#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE

typedef __u64 u64;
typedef __u32 u32;
typedef __u16 u16;
typedef __u8 u8;

std::atomic<u64> nr_rx_packets[2] = {0, 0};
std::atomic<u64> nr_tx_packets[2] = {0, 0};

struct xsk_socket_info {
    int xsk_fd;

    int umem_fd;

    void *umem_area;
    size_t umem_size;

    struct xsk_ring_prod fq;
    void *fill_map;
    size_t fill_map_size;

    struct xsk_ring_cons cq;
    void *comp_map;
    size_t comp_map_size;

    struct xsk_ring_cons rx;
    void *rx_map;
    size_t rx_map_size;

    struct xsk_ring_prod tx;
    void *tx_map;
    size_t tx_map_size;

    /* stats for l2fwd */
    u64 outstanding_tx;
};

int receive_fd(int sockfd, int *fd) {
    assert(sockfd >= 0);
    struct msghdr msg;
    struct iovec iov;
    char buf[CMSG_SPACE(sizeof(int))];
    struct cmsghdr *cmsg;

    iov.iov_base = buf;
    iov.iov_len = sizeof(buf);

    msg.msg_name = 0;
    msg.msg_namelen = 0;

    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);
    if (recvmsg(sockfd, &msg, 0) < 0) {
        perror("recvmsg failed\n");
        return -1;
    }
    cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == NULL || cmsg->cmsg_type != SCM_RIGHTS) {
        perror("recvmsg failed\n");
        return -1;
    }
    *fd = *((int *)CMSG_DATA(cmsg));
    return 0;
}

void *attach_shm(size_t size) {
    int fd;
    void *addr;

    fd = shm_open(SHM_NAME, O_RDWR, 0);
    if (fd == -1) {
        perror("shm_open");
        return MAP_FAILED;
    }

    addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        perror("mmap");
        return MAP_FAILED;
    }

    return addr;
}

void detach_shm(void *addr, size_t size) {
    if (munmap(addr, size) == -1) {
        perror("munmap");
        exit(EXIT_FAILURE);
    }
}

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

void xsk_mmap_offsets_v1(struct xdp_mmap_offsets *off) {
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

int xsk_get_mmap_offsets(int fd, struct xdp_mmap_offsets *off) {
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

void destroy_afxdp_socket(struct xsk_socket_info *xsk_info) {
    if (!xsk_info) return;
    if (xsk_info->rx_map && xsk_info->rx_map != MAP_FAILED)
        munmap(xsk_info->rx_map, xsk_info->rx_map_size);
    if (xsk_info->tx_map && xsk_info->tx_map != MAP_FAILED)
        munmap(xsk_info->tx_map, xsk_info->tx_map_size);
    if (xsk_info->fill_map && xsk_info->fill_map != MAP_FAILED)
        munmap(xsk_info->fill_map, xsk_info->fill_map_size);
    if (xsk_info->comp_map && xsk_info->comp_map != MAP_FAILED)
        munmap(xsk_info->comp_map, xsk_info->comp_map_size);
    if (xsk_info->umem_area && xsk_info->umem_area != MAP_FAILED)
        detach_shm(xsk_info->umem_area, xsk_info->umem_size);
}

/**
 * @brief: Manually map UMEM and build four rings for a AF_XDP socket
 * @note: (RX/TX/FILL/COMP_RING_SIZE, NUM_FRAMES, FRAME_SIZE) need negotiating
 * with privileged processes
 */
int create_afxdp_socket(struct xsk_socket_info *xsk_info, int xsk_fd,
                        int umem_fd) {
    struct xsk_ring_cons *rx = &xsk_info->rx;
    struct xsk_ring_prod *tx = &xsk_info->tx;
    struct xsk_ring_prod *fill = &xsk_info->fq;
    struct xsk_ring_cons *comp = &xsk_info->cq;
    struct xdp_mmap_offsets off;

    memset(xsk_info, 0, sizeof(*xsk_info));

    /* Map UMEM */
    xsk_info->umem_size = NUM_FRAMES * FRAME_SIZE;
    xsk_info->umem_area = attach_shm(xsk_info->umem_size);
    if (xsk_info->umem_area == MAP_FAILED) {
        perror("mmap");
        goto out;
    }

    /* Get offsets for the following mmap */
    if (xsk_get_mmap_offsets(umem_fd, &off)) {
        perror("xsk_get_mmap_offsets failed");
        goto out;
    }

    /* RX Ring */
    xsk_info->rx_map_size =
        off.rx.desc + RX_RING_SIZE * sizeof(struct xdp_desc);
    xsk_info->rx_map =
        mmap(NULL, xsk_info->rx_map_size, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, xsk_fd, XDP_PGOFF_RX_RING);
    if (xsk_info->rx_map == MAP_FAILED) {
        perror("rx mmap failed");
        goto out;
    }
    rx->mask = RX_RING_SIZE - 1;
    rx->size = RX_RING_SIZE;
    rx->producer = (uint32_t *)((char *)xsk_info->rx_map + off.rx.producer);
    rx->consumer = (uint32_t *)((char *)xsk_info->rx_map + off.rx.consumer);
    rx->flags = (uint32_t *)((char *)xsk_info->rx_map + off.rx.flags);
    rx->ring = xsk_info->rx_map + off.rx.desc;
    rx->cached_prod = *rx->producer;
    rx->cached_cons = *rx->consumer;

    /* TX Ring */
    xsk_info->tx_map_size =
        off.tx.desc + TX_RING_SIZE * sizeof(struct xdp_desc);
    xsk_info->tx_map =
        mmap(NULL, xsk_info->tx_map_size, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, xsk_fd, XDP_PGOFF_TX_RING);
    if (xsk_info->tx_map == MAP_FAILED) {
        perror("tx mmap failed");
        goto out;
    }
    tx->mask = TX_RING_SIZE - 1;
    tx->size = TX_RING_SIZE;
    tx->producer = (uint32_t *)((char *)xsk_info->tx_map + off.tx.producer);
    tx->consumer = (uint32_t *)((char *)xsk_info->tx_map + off.tx.consumer);
    tx->flags = (uint32_t *)((char *)xsk_info->tx_map + off.tx.flags);
    tx->ring = xsk_info->tx_map + off.tx.desc;
    tx->cached_prod = *tx->producer;
    tx->cached_cons = *tx->consumer + TX_RING_SIZE;

    /* Fill Ring */
    xsk_info->fill_map_size = off.fr.desc + FILL_RING_SIZE * sizeof(__u64);
    xsk_info->fill_map =
        mmap(NULL, xsk_info->fill_map_size, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, xsk_fd, XDP_UMEM_PGOFF_FILL_RING);
    if (xsk_info->fill_map == MAP_FAILED) {
        perror("fill mmap failed");
        goto out;
    }
    fill->mask = FILL_RING_SIZE - 1;
    fill->size = FILL_RING_SIZE;
    fill->producer = (uint32_t *)((char *)xsk_info->fill_map + off.fr.producer);
    fill->consumer = (uint32_t *)((char *)xsk_info->fill_map + off.fr.consumer);
    fill->flags = (uint32_t *)((char *)xsk_info->fill_map + off.fr.flags);
    fill->ring = xsk_info->fill_map + off.fr.desc;
    fill->cached_cons = FILL_RING_SIZE;

    /* Completion Ring */
    xsk_info->comp_map_size = off.cr.desc + COMP_RING_SIZE * sizeof(__u64);
    xsk_info->comp_map =
        mmap(NULL, xsk_info->comp_map_size, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, xsk_fd, XDP_UMEM_PGOFF_COMPLETION_RING);
    if (xsk_info->comp_map == MAP_FAILED) {
        perror("comp mmap failed");
        goto out;
    }

    comp->mask = COMP_RING_SIZE - 1;
    comp->size = COMP_RING_SIZE;
    comp->producer = (uint32_t *)((char *)xsk_info->comp_map + off.cr.producer);
    comp->consumer = (uint32_t *)((char *)xsk_info->comp_map + off.cr.consumer);
    comp->flags = (uint32_t *)((char *)xsk_info->comp_map + off.cr.flags);
    comp->ring = xsk_info->comp_map + off.cr.desc;

    xsk_info->xsk_fd = xsk_fd;
    xsk_info->umem_fd = umem_fd;

    return 0;
out:
    destroy_afxdp_socket(xsk_info);
    return -1;
}

int populate_fill_ring(struct xsk_ring_prod *fq) {
    int ret, i;
    __u32 idx;

    ret = xsk_ring_prod__reserve(fq, FILL_RING_SIZE, &idx);
    if (ret != FILL_RING_SIZE) {
        fprintf(stderr, "xsk_ring_prod__reserve failed, %d\n", ret);
        return -1;
    }
    for (i = 0; i < FILL_RING_SIZE; i++)
        *xsk_ring_prod__fill_addr(fq, idx++) = i * FRAME_SIZE;
    xsk_ring_prod__submit(fq, FILL_RING_SIZE);

    return 0;
}

void rx_drop(struct xsk_socket_info *xsk, int qid) {
    unsigned int rcvd, i;
    u32 idx_rx = 0, idx_fq = 0;
    int ret;

    rcvd = xsk_ring_cons__peek(&xsk->rx, 32, &idx_rx);
    if (!rcvd) {
        if (xsk_ring_prod__needs_wakeup(&xsk->fq)) {
            recvfrom(xsk->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, NULL);
        }
        return;
    }

    ret = xsk_ring_prod__reserve(&xsk->fq, rcvd, &idx_fq);
    while (ret != rcvd) {
        if (ret < 0) exit(EXIT_FAILURE);
        if (xsk_ring_prod__needs_wakeup(&xsk->fq)) {
            recvfrom(xsk->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, NULL);
        }
        ret = xsk_ring_prod__reserve(&xsk->fq, rcvd, &idx_fq);
    }

    for (i = 0; i < rcvd; i++) {
        const struct xdp_desc *desc =
            xsk_ring_cons__rx_desc(&xsk->rx, idx_rx++);
        u64 addr = desc->addr;
        u32 len = desc->len;
        u64 orig = xsk_umem__extract_addr(addr);

        addr = xsk_umem__add_offset_to_addr(addr);
        char *pkt = (char *)xsk_umem__get_data(xsk->umem_area, addr);

        (void)pkt;
        (void)len;

        *xsk_ring_prod__fill_addr(&xsk->fq, idx_fq++) = orig;
    }

    xsk_ring_prod__submit(&xsk->fq, rcvd);
    xsk_ring_cons__release(&xsk->rx, rcvd);

    nr_rx_packets[qid] += rcvd;
}

static inline void swap_mac_addresses(void *data) {
    struct ether_header *eth = (struct ether_header *)data;
    struct iphdr *ip = (struct iphdr *)(data + sizeof(struct ether_header));
    struct udphdr *udp = (struct udphdr *)((void *)ip + sizeof(struct iphdr));

    struct ether_addr *src_addr = (struct ether_addr *)&eth->ether_shost;
    struct ether_addr *dst_addr = (struct ether_addr *)&eth->ether_dhost;
    struct ether_addr tmp;
    uint32_t tmp_ip;
    uint16_t tmp_port;

    tmp = *src_addr;
    *src_addr = *dst_addr;
    *dst_addr = tmp;

    tmp_ip = ip->saddr;
    ip->saddr = ip->daddr;
    ip->daddr = tmp_ip;

    tmp_port = udp->source;
    udp->source = udp->dest;
    udp->dest = tmp_port;
}

static void kick_tx(struct xsk_socket_info *xsk) {
    (void)sendto(xsk->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, 0);
}

static inline void complete_tx_l2fwd(struct xsk_socket_info *xsk) {
    u32 idx_cq = 0, idx_fq = 0;
    unsigned int rcvd;
    size_t ndescs;

    if (!xsk->outstanding_tx) return;

    ndescs = (xsk->outstanding_tx > 32) ? 32 : xsk->outstanding_tx;

    /* re-add completed Tx buffers */
    rcvd = xsk_ring_cons__peek(&xsk->cq, ndescs, &idx_cq);
    if (rcvd > 0) {
        unsigned int i;
        int ret;

        ret = xsk_ring_prod__reserve(&xsk->fq, rcvd, &idx_fq);
        while (ret != rcvd) {
            if (ret < 0) exit(EXIT_FAILURE);
            if (xsk_ring_prod__needs_wakeup(&xsk->fq)) {
                recvfrom(xsk->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, NULL);
            }
            ret = xsk_ring_prod__reserve(&xsk->fq, rcvd, &idx_fq);
        }

        for (i = 0; i < rcvd; i++)
            *xsk_ring_prod__fill_addr(&xsk->fq, idx_fq++) =
                *xsk_ring_cons__comp_addr(&xsk->cq, idx_cq++);

        xsk_ring_prod__submit(&xsk->fq, rcvd);
        xsk_ring_cons__release(&xsk->cq, rcvd);
        xsk->outstanding_tx -= rcvd;
    }
}

void l2fwd(struct xsk_socket_info *xsk, int qid) {
    u32 idx_rx = 0, idx_tx = 0;
    unsigned int rcvd, i;
    int ret;

    complete_tx_l2fwd(xsk);

    rcvd = xsk_ring_cons__peek(&xsk->rx, 32, &idx_rx);
    if (!rcvd) {
        if (xsk_ring_prod__needs_wakeup(&xsk->fq)) {
            recvfrom(xsk->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, NULL);
        }
        return;
    }

    ret = xsk_ring_prod__reserve(&xsk->tx, rcvd, &idx_tx);
    while (ret != rcvd) {
        if (ret < 0) exit(EXIT_FAILURE);
        complete_tx_l2fwd(xsk);
        if (xsk_ring_prod__needs_wakeup(&xsk->tx)) {
            kick_tx(xsk);
        }
        ret = xsk_ring_prod__reserve(&xsk->tx, rcvd, &idx_tx);
    }

    for (i = 0; i < rcvd; i++) {
        const struct xdp_desc *desc =
            xsk_ring_cons__rx_desc(&xsk->rx, idx_rx++);
        u64 addr = desc->addr;
        u32 len = desc->len;
        u64 orig = addr;

        addr = xsk_umem__add_offset_to_addr(addr);
        char *pkt = (char *)xsk_umem__get_data(xsk->umem_area, addr);

        swap_mac_addresses(pkt);

        struct xdp_desc *tx_desc = xsk_ring_prod__tx_desc(&xsk->tx, idx_tx++);

        tx_desc->options = 0;
        tx_desc->addr = orig;
        tx_desc->len = len;
    }

    xsk_ring_prod__submit(&xsk->tx, rcvd);
    xsk_ring_cons__release(&xsk->rx, rcvd);

    xsk->outstanding_tx += rcvd;
    nr_rx_packets[qid] += rcvd;
    nr_tx_packets[qid] += rcvd;
}

void *dump_nr_rx_packets(void *arg) {
    u64 prev_nr_rx_packets[2] = {0, 0};
    u64 prev_nr_tx_packets[2] = {0, 0};
    while (1) {
        for (int i = 0; i < 2; i++) {
            printf("Rx %d pps: %.2f Mpps, Tx pps: %.2f Mpps\n", i,
                   (nr_rx_packets[i] - prev_nr_rx_packets[i]) / 1e6,
                   (nr_tx_packets[i] - prev_nr_tx_packets[i]) / 1e6);
            prev_nr_rx_packets[i] = nr_rx_packets[i];
            prev_nr_tx_packets[i] = nr_tx_packets[i];
            sleep(1);
        }
    }
}

int main() {
    int ret;
    int client_sock;
    struct sockaddr_un addr;

    struct xsk_socket_info xsk_info;
    struct xsk_socket_info xsk_info2;
    std::thread *t1, *t2;

    // Create a UNIX domain socket to receive file descriptors
    if ((client_sock = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, SOCKET_PATH);

    if (connect(client_sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("connect");
        exit(EXIT_FAILURE);
    }

    // Step1: receive the file descriptors for AF_XDP socket and UMEM
    int xsk_fd, xsk_fd2, umem_fd;
    assert(receive_fd(client_sock, &xsk_fd) == 0);
    assert(receive_fd(client_sock, &xsk_fd2) == 0);
    assert(receive_fd(client_sock, &umem_fd) == 0);

    // Step2: map UMEM and build four rings for the AF_XDP socket
    ret = create_afxdp_socket(&xsk_info, xsk_fd, umem_fd);
    if (ret) {
        fprintf(stderr, "xsk_socket__create_shared failed, %d\n", ret);
        goto out;
    }

    ret = create_afxdp_socket(&xsk_info2, xsk_fd2, umem_fd);
    if (ret) {
        fprintf(stderr, "xsk_socket__create_shared failed, %d\n", ret);
        goto out;
    }

    printf("AF_XDP socket successfully shared.\n");

    if (populate_fill_ring(&xsk_info.fq)) {
        fprintf(stderr, "populate_fill_ring failed\n");
        goto out;
    }

    if (populate_fill_ring(&xsk_info2.fq)) {
        fprintf(stderr, "populate_fill_ring failed\n");
        goto out;
    }

    pthread_t dump_thread;
    pthread_create(&dump_thread, NULL, dump_nr_rx_packets, NULL);

    t1 = new std::thread([&xsk_info]() {
        while (1) {
            // rx_drop(&xsk_info, 0);
            l2fwd(&xsk_info, 0);
        }
    });

    t2 = new std::thread([&xsk_info2]() {
        while (1) {
            // rx_drop(&xsk_info, 1);
            l2fwd(&xsk_info2, 1);
        }
    });

    t1->join();
    t2->join();

out:
    destroy_afxdp_socket(&xsk_info);
    close(client_sock);
    return 0;
}