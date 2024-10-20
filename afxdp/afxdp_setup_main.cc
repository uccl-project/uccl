#include <linux/if_xdp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>  // For chmod()
#include <sys/un.h>
#include <unistd.h>
#include <xdp/xsk.h>

#define SOCKET_PATH "/tmp/privileged_socket"
#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE
#define NUM_FRAMES 4096

void set_socket_permissions(const char *path) {
    // Set the permissions of the UNIX domain socket to allow non-privileged
    // access
    if (chmod(path, 0666) == -1) {  // Permissions: Read/Write for everyone
        perror("chmod");
        exit(EXIT_FAILURE);
    }
}

struct xsk_umem *umem;
struct xsk_ring_prod fill_ring;
struct xsk_ring_cons comp_ring;
struct xsk_socket *xsk;
struct xsk_ring_cons rx_ring;
struct xsk_ring_prod tx_ring;

void send_fd(int unix_sock, int fd) {
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char dummy_data[64];
    char buf[CMSG_SPACE(sizeof(fd))];  // Allocate space for FD and control data
    struct iovec io = {.iov_base = (void *)dummy_data,
                       .iov_len = 2};  // Dummy data for iovec

    // Zero out the control message buffer
    memset(buf, 0, sizeof(buf));

    // Set up the msghdr structure
    msg.msg_iov = &io;      // I/O vector for message body (dummy data here)
    msg.msg_iovlen = 1;     // One element in the I/O vector
    msg.msg_control = buf;  // Control message buffer
    msg.msg_controllen = sizeof(buf);  // Size of control message buffer

    // Set up the cmsghdr structure
    cmsg = CMSG_FIRSTHDR(&msg);  // Initialize the first control message header
    cmsg->cmsg_level = SOL_SOCKET;          // Specify the socket level
    cmsg->cmsg_type = SCM_RIGHTS;           // Specify the type (passing FD)
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd));  // Length of the control message

    // Copy the file descriptor into the control message data
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));

    // Send the message, including the file descriptor
    if (sendmsg(unix_sock, &msg, 0) == -1) {
        perror("sendmsg");
        exit(EXIT_FAILURE);
    }
}

int main() {
    int server_sock, client_sock;
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

    // Create a UNIX domain socket to send file descriptors
    if ((server_sock = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, SOCKET_PATH);
    unlink(SOCKET_PATH);
    if (bind(server_sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    // Set the permissions on the UNIX domain socket file
    set_socket_permissions(SOCKET_PATH);

    if (listen(server_sock, 5) == -1) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    printf("Waiting for non-privileged process to connect...\n");
    if ((client_sock = accept(server_sock, NULL, NULL)) == -1) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    printf("Connected!\n");

    // Allocate UMEM and create AF_XDP socket
    umem_area = mmap(NULL, umem_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (umem_area == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }
    if (xsk_umem__create(&umem, umem_area, umem_size, &fill_ring, &comp_ring,
                         &umem_cfg)) {
        perror("xsk_umem__create");
        exit(EXIT_FAILURE);
    }
    if (xsk_socket__create(&xsk, "ens6", 0, umem, &rx_ring, &tx_ring,
                           &xsk_cfg)) {
        perror("xsk_socket__create");
        exit(EXIT_FAILURE);
    }

    printf("sending fds!\n");
    // Send the file descriptors for the AF_XDP socket and UMEM
    send_fd(client_sock, xsk_socket__fd(xsk));  // Send AF_XDP socket FD
    send_fd(client_sock, xsk_umem__fd(umem));   // Send UMEM FD

    close(client_sock);
    close(server_sock);

    while (true) {
        sleep(1);
    }
    return 0;
}
