#include <assert.h>
#include <fcntl.h>
#include <linux/if_xdp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <xdp/xsk.h>

#include "transport_config.h"

#define IF_NAME DEV_DEFAULT
#define SHM_NAME "UMEM_SHM"
#define QUEUE_ID 0
#define SOCKET_PATH "/tmp/privileged_socket"
#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE

struct xsk_umem *umem;

struct xsk_socket *xsk;
struct xsk_ring_prod fill_ring;
struct xsk_ring_cons comp_ring;
struct xsk_ring_cons rx_ring;
struct xsk_ring_prod tx_ring;

struct xsk_socket *xsk2;
struct xsk_ring_prod fill_ring2;
struct xsk_ring_cons comp_ring2;
struct xsk_ring_cons rx_ring2;
struct xsk_ring_prod tx_ring2;

int send_fd(int sockfd, int fd) {
    assert(sockfd >= 0);
    assert(fd >= 0);
    struct msghdr msg;
    struct cmsghdr *cmsg;
    struct iovec iov;
    char buf[CMSG_SPACE(sizeof(fd))];
    memset(&msg, 0, sizeof(msg));
    memset(buf, 0, sizeof(buf));
    const char *name = "fd";
    iov.iov_base = (void *)name;
    iov.iov_len = 4;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    cmsg = CMSG_FIRSTHDR(&msg);

    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd));

    *((int *)CMSG_DATA(cmsg)) = fd;

    msg.msg_controllen = CMSG_SPACE(sizeof(fd));

    if (sendmsg(sockfd, &msg, 0) < 0) {
        fprintf(stderr, "sendmsg failed\n");
        return -1;
    }
    return 0;
}

void *create_shm(size_t size) {
    int fd;
    void *addr;

    /* unlink it if we exit excpetionally before */
    shm_unlink(SHM_NAME);

    fd = shm_open(SHM_NAME, O_CREAT | O_RDWR | O_EXCL, 0666);
    if (fd == -1) {
        perror("shm_open");
        return MAP_FAILED;
    }

    if (ftruncate(fd, size) == -1) {
        perror("ftruncate");
        return MAP_FAILED;
    }

    addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        perror("mmap");
        return MAP_FAILED;
    }

    return addr;
}

void destroy_shm(void *addr, size_t size) {
    munmap(addr, size);
    shm_unlink(SHM_NAME);
}

int main() {
    int server_sock, client_sock;
    struct sockaddr_un addr;
    uint64_t umem_size = NUM_FRAMES * FRAME_SIZE;
    struct xsk_umem_config umem_cfg = {.fill_size = FILL_RING_SIZE,
                                       .comp_size = COMP_RING_SIZE,
                                       .frame_size = FRAME_SIZE,
                                       .frame_headroom = 0,
                                       .flags = 0};
    struct xsk_socket_config xsk_cfg = {.rx_size = RX_RING_SIZE,
                                        .tx_size = TX_RING_SIZE,
                                        .libbpf_flags = 0,
                                        .xdp_flags = 0,
                                        .bind_flags = XDP_USE_NEED_WAKEUP};

    // Create a UNIX domain socket to send file descriptors
    if ((server_sock = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, SOCKET_PATH);

    mode_t old_mask = umask(0);  // set directory priviledge

    unlink(SOCKET_PATH);
    if (bind(server_sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    umask(old_mask);  // restore

    if (listen(server_sock, 5) == -1) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    printf("Waiting for non-privileged process to connect...\n");
    if ((client_sock = accept(server_sock, NULL, NULL)) == -1) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    // Step1: prepare a large shared memory for UMEM
    old_mask = umask(0);  // set directory priviledge
    void *umem_area = NULL;
    umem_area = create_shm(umem_size);
    if (umem_area == MAP_FAILED) {
        perror("mmap");
        goto out;
    }
    umask(old_mask);  // restore

    // Step2: create UMEM
    if (xsk_umem__create(&umem, umem_area, umem_size, &fill_ring, &comp_ring,
                         &umem_cfg)) {
        perror("xsk_umem__create");
        goto out;
    }
    // Step3: create a AF_XDP socket and bind it to a NIC queue and the UMEM
    if (xsk_socket__create(&xsk, IF_NAME, QUEUE_ID, umem, &rx_ring, &tx_ring,
                           &xsk_cfg)) {
        perror("xsk_socket__create");
        goto out;
    }

    if (xsk_socket__create_shared(&xsk2, IF_NAME, QUEUE_ID + 1, umem, &rx_ring2, &tx_ring2,
                                &fill_ring2, &comp_ring2, &xsk_cfg)) {
        perror("xsk_socket__create_shared");
        goto out;
    }

    /* Note: Actually, xsk_socket__fd(xsk) == xsk_umem__fd(umem), there is no
     * need to send both. We send both of them just for demonstration. This is
     * because libxdp would reuse the UMEM's fd for the ***first socket***
     * binding to the UMEM, the UMEM fd is also created by socket(AF_XDP,
     * SOCK_RAW, 0). See xsk_socket__create_shared()
     */
    assert(xsk_socket__fd(xsk) == xsk_umem__fd(umem));

    // Step4: send the file descriptors for the AF_XDP socket and UMEM
    if (send_fd(client_sock, xsk_socket__fd(xsk))) goto out;
    if (send_fd(client_sock, xsk_socket__fd(xsk2))) goto out;
    if (send_fd(client_sock, xsk_umem__fd(umem))) goto out;

    while (1) {
        sleep(1);
    }

out:
    if (umem_area != MAP_FAILED) {
        destroy_shm(umem_area, umem_size);
    }
    close(client_sock);
    close(server_sock);
    unlink(SOCKET_PATH);

    return 0;
}