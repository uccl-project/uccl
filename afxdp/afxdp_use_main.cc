#include <linux/if_xdp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <xdp/xsk.h>

#define SOCKET_PATH "/tmp/privileged_socket"
#define NUM_FRAMES 4096
#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE

struct xsk_umem *umem;
struct xsk_socket *xsk;
struct xsk_ring_prod fill_ring;
struct xsk_ring_cons comp_ring;
struct xsk_ring_cons rx_ring;
struct xsk_ring_prod tx_ring;

int receive_fd(int unix_sock) {
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char dummy_data[64];
    char
        buf[CMSG_SPACE(sizeof(int))];  // Allocate space for FD and control data
    struct iovec io = {.iov_base = (void *)dummy_data,
                       .iov_len = 2};  // Dummy data for iovec
    int fd = -1;

    // Zero out the control message buffer to avoid uninitialized memory issues
    memset(buf, 0, sizeof(buf));

    // Prepare the msghdr structure
    msg.msg_iov = &io;      // I/O vector for message body (dummy data here)
    msg.msg_iovlen = 1;     // One element in the I/O vector
    msg.msg_control = buf;  // Control message buffer
    msg.msg_controllen = sizeof(buf);  // Size of control message buffer

    // Call recvmsg to receive the file descriptor
    if (recvmsg(unix_sock, &msg, 0) == -1) {
        perror("recvmsg");
        return -1;
    }

    // Extract the received file descriptor from the ancillary data
    cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg && cmsg->cmsg_level == SOL_SOCKET &&
        cmsg->cmsg_type == SCM_RIGHTS) {
        memcpy(&fd, CMSG_DATA(cmsg),
               sizeof(fd));  // Copy the file descriptor from cmsg
    } else {
        fprintf(stderr, "No file descriptor received.\n");
        return -1;
    }

    return fd;  // Return the received file descriptor
}

int main() {
    int client_sock;
    struct sockaddr_un addr;
    void *umem_area;
    uint64_t umem_size = NUM_FRAMES * FRAME_SIZE;
    struct xsk_umem_config umem_cfg = {.fill_size = NUM_FRAMES,
                                       .comp_size = NUM_FRAMES,
                                       .frame_size = FRAME_SIZE,
                                       .frame_headroom = 0,
                                       .flags = 0};

    struct xsk_socket_config xsk_cfg = {.rx_size = 2048,
                                        .tx_size = 2048,
                                        .libbpf_flags = 0,
                                        .xdp_flags = 0,
                                        .bind_flags = XDP_USE_NEED_WAKEUP};

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

    // Receive the file descriptors for AF_XDP socket and UMEM
    int xsk_socket_fd = receive_fd(client_sock);
    int umem_fd = receive_fd(client_sock);

    // Map the UMEM area
    // umem_area = mmap(NULL, umem_size, PROT_READ | PROT_WRITE,
    //                  MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, umem_fd, 0);
    // if (umem_area == MAP_FAILED) {
    //     perror("mmap");
    //     exit(EXIT_FAILURE);
    // }

    // Recreate the UMEM in this process (non-privileged)
    int ret = xsk_umem__create_with_fd(&umem, umem_fd, umem_area, umem_size,
                                       &fill_ring, &comp_ring, &umem_cfg);
    if (ret) {
        printf("error code %d\n", -ret);
        perror("xsk_umem__create_with_fd");
        close(umem_fd);
        close(client_sock);
        exit(EXIT_FAILURE);
    }

    // Create a shared socket using xsk_socket__create_shared
    ret = xsk_socket__create_shared(&xsk, "ens6", 0, umem, &rx_ring, &tx_ring,
                                    &fill_ring, &comp_ring, &xsk_cfg);
    if (ret) {
        printf("error code %d\n", -ret);
        perror("xsk_socket__create_shared");
        exit(EXIT_FAILURE);
    }

    printf("AF_XDP socket successfully shared.\n");

    close(client_sock);
    return 0;
}
