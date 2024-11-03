#include <assert.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <linux/if_xdp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <xdp/libxdp.h>
#include <xdp/xsk.h>

#include <deque>
#include <mutex>

#include "transport_config.h"
#include "util.h"

using namespace uccl;

#define IF_NAME interface_name
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
#define NUM_FRAMES (64 * 4096)

volatile bool quit = false;
int interface_index;
char interface_name_attach[256];
struct xdp_program *program_attach;
bool attached_native;
bool attached_skb;

struct xsk_umem *umem;
void *umem_area;
uint64_t umem_size;
struct xsk_ring_prod fill_ring;
struct xsk_ring_cons comp_ring;
struct xsk_socket *xsk;
struct xsk_ring_cons rx_ring;
struct xsk_ring_prod tx_ring;

void load_program(const char *interface_name, const char *ebpf_filename,
                  const char *section_name) {
    // we can only run xdp programs as root
    CHECK(geteuid() == 0) << "error: this program must be run as root";

    strcpy(interface_name_attach, interface_name);
    // find the network interface that matches the interface name
    interface_index = get_dev_index(interface_name);

    CHECK(interface_index != -1)
        << "error: could not find any network interface matching "
        << interface_name;

    // load the ebpf program
    LOG(INFO) << "loading " << section_name << "...";
    program_attach = xdp_program__open_file(ebpf_filename, section_name, NULL);
    CHECK(!libxdp_get_error(program_attach))
        << "error: could not load " << ebpf_filename << " program";
    LOG(INFO) << ebpf_filename << " loaded successfully.";

    // attach the ebpf program to the network interface
    LOG(INFO) << "attaching " << ebpf_filename << " to network interface";
    int ret = xdp_program__attach(program_attach, interface_index,
                                  XDP_MODE_NATIVE, 0);
    if (ret == 0) {
        attached_native = true;
    } else {
        LOG(INFO) << "falling back to skb mode...";
        ret = xdp_program__attach(program_attach, interface_index, XDP_MODE_SKB,
                                  0);
        if (ret == 0) {
            attached_skb = true;
        } else {
            LOG(ERROR) << "error: failed to attach " << ebpf_filename
                       << " program to interface";
        }
    }
    LOG(INFO) << ebpf_filename << " attached successfully.";

    // allow unlimited locking of memory, so all memory needed for packet
    // buffers can be locked
    struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};
    CHECK(!setrlimit(RLIMIT_MEMLOCK, &rlim)) << "error: could not setrlimit";
}

void update_xsks_map() {
    // We also need to load and update the xsks_map for receiving packets
    struct bpf_map *map = bpf_object__find_map_by_name(
        xdp_program__bpf_obj(program_attach), "xsks_map");
    int xsk_map_fd = bpf_map__fd(map);
    CHECK(xsk_map_fd >= 0) << "ERROR: no xsks map found: "
                           << strerror(xsk_map_fd);

    int ret = xsk_socket__update_xskmap(xsk, xsk_map_fd);
    CHECK_EQ(ret, 0) << "ERROR: xsks map update fails: "
                     << strerror(xsk_map_fd);
}

void create_umem_and_xsk() {
    program_attach = nullptr;
    umem = nullptr;
    xsk = nullptr;
    umem_size = NUM_FRAMES * FRAME_SIZE;
    memset(&fill_ring, 0, sizeof(fill_ring));
    memset(&comp_ring, 0, sizeof(comp_ring));
    memset(&rx_ring, 0, sizeof(rx_ring));
    memset(&tx_ring, 0, sizeof(tx_ring));

    struct xsk_umem_config umem_cfg = {.fill_size = FILL_RING_SIZE,
                                       .comp_size = COMP_RING_SIZE,
                                       .frame_size = FRAME_SIZE,
                                       .frame_headroom = 0,
                                       .flags = 0};
    struct xsk_socket_config xsk_cfg = {
        .rx_size = RX_RING_SIZE,
        .tx_size = TX_RING_SIZE,
        .libbpf_flags = XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD,
        .xdp_flags = XDP_ZEROCOPY,
        .bind_flags = XDP_USE_NEED_WAKEUP};

    // Step0: load the ebpf program
    // load_program(IF_NAME, "ebpf_transport_pktloss.o", "ebpf_transport");
    load_program(IF_NAME, "ebpf_transport.o", "ebpf_transport");

    // Step1: prepare a large shared memory for UMEM
    if (umem_area == nullptr) {
        mode_t old_mask = umask(0);  // set directory priviledge
        umem_area = create_shm(SHM_NAME, umem_size);
        if (umem_area == MAP_FAILED) {
            perror("mmap");
            goto out;
        }
        umask(old_mask);  // restore
    }

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

    /* Note: Actually, xsk_socket__fd(xsk) == xsk_umem__fd(umem), there is no
     * need to send both. We send both of them just for demonstration. This is
     * because libxdp would reuse the UMEM's fd for the ***first socket***
     * binding to the UMEM, the UMEM fd is also created by socket(AF_XDP,
     * SOCK_RAW, 0). See xsk_socket__create_shared()
     */
    DCHECK_EQ(xsk_socket__fd(xsk), xsk_umem__fd(umem));

    // Step4: update the xsks_map for receiving packets
    update_xsks_map();

    return;

out:
    if (umem_area != MAP_FAILED) {
        destroy_shm(SHM_NAME, umem_area, umem_size);
    }
    unlink(SOCKET_PATH);
    exit(EXIT_FAILURE);
}

void destroy_umem_and_xsk(bool free_shm = false) {
    if (program_attach) {
        if (attached_native)
            xdp_program__detach(program_attach, interface_index,
                                XDP_MODE_NATIVE, 0);

        if (attached_skb)
            xdp_program__detach(program_attach, interface_index, XDP_MODE_SKB,
                                0);

        xdp_program__close(program_attach);
    }

    if (xsk) xsk_socket__delete(xsk);
    if (umem) xsk_umem__delete(umem);
    if (umem_area && free_shm) destroy_shm(SHM_NAME, umem_area, umem_size);
}

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
    destroy_umem_and_xsk(/*free_shm =*/true);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);

    int server_sock, client_sock;
    struct sockaddr_un addr;

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

    if (listen(server_sock, 128) == -1) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    while (true) {
        printf("Waiting for non-privileged process to connect...\n");
        if ((client_sock = accept(server_sock, NULL, NULL)) == -1) {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        destroy_umem_and_xsk();
        create_umem_and_xsk();

        // Step5: send the file descriptors for the AF_XDP socket and UMEM
        if (send_fd(client_sock, xsk_socket__fd(xsk))) break;
        if (send_fd(client_sock, xsk_umem__fd(umem))) break;

        close(client_sock);
    }

    close(server_sock);

    return 0;
}