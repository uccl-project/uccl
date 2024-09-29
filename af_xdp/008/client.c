/*
    UDP client (userspace)

    Runs on Ubuntu 22.04 LTS 64bit with Linux Kernel 6.5+ *ONLY*

    Derived from
   https://github.com/xdp-project/xdp-tutorial/tree/master/advanced03-AF_XDP
*/

#define _GNU_SOURCE

#include <arpa/inet.h>
#include <assert.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
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
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>
#include <xdp/libxdp.h>
#include <xdp/xsk.h>

#define NUM_CPUS 1

const char* INTERFACE_NAME = "ens5";

const uint8_t SERVER_ETHERNET_ADDRESS[] = {0x0e, 0x47, 0x06, 0x90, 0x9a, 0xc5};

const uint8_t CLIENT_ETHERNET_ADDRESS[] = {0x0e, 0x8c, 0xef, 0x4e, 0x54, 0xa3};

const uint32_t SERVER_IPV4_ADDRESS = 0xac1f24a4;  // 172.31.36.164

const uint16_t SERVER_PORT = 40000;

const uint32_t CLIENT_IPV4_ADDRESS = 0xac1f21ce;  // 172.31.33.206

const uint16_t CLIENT_PORT = 40000;

const int PAYLOAD_BYTES = 32;

const int SEND_BATCH_SIZE = 256;

const int RECV_BATCH_SIZE = 256;

const bool busy_poll = false;

#define NUM_FRAMES (4096 * 16)

#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE

#define INVALID_FRAME UINT64_MAX

struct socket_t {
    void* buffer;
    struct xsk_umem* umem;
    struct xsk_ring_cons recv_queue;
    struct xsk_ring_prod send_queue;
    struct xsk_ring_cons complete_queue;
    struct xsk_ring_prod fill_queue;
    struct xsk_socket* xsk;
    // TODO(yang): make frames and num_frames thread-safe.
    uint64_t frames[NUM_FRAMES];
    uint32_t num_frames;
    uint64_t sent_packets;
    uint32_t counter;
    int queue_id;
};

struct client_t {
    int interface_index;
    struct xdp_program* program;
    bool attached_native;
    bool attached_skb;
    struct socket_t socket[NUM_CPUS];
    pthread_t stats_thread;
    pthread_t send_thread[NUM_CPUS];
    pthread_t recv_thread[NUM_CPUS];
    uint64_t previous_sent_packets;
};

static void* stats_thread(void* arg);
static void* send_thread(void* arg);
static void* recv_thread(void* arg);

int client_init(struct client_t* client, const char* interface_name) {
    // we can only run xdp programs as root
    if (geteuid() != 0) {
        printf("\nerror: this program must be run as root\n\n");
        return 1;
    }

    // find the network interface that matches the interface name
    {
        bool found = false;

        struct ifaddrs* addrs;
        if (getifaddrs(&addrs) != 0) {
            printf("\nerror: getifaddrs failed\n\n");
            return 1;
        }

        for (struct ifaddrs* iap = addrs; iap != NULL; iap = iap->ifa_next) {
            if (iap->ifa_addr && (iap->ifa_flags & IFF_UP) &&
                iap->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in* sa = (struct sockaddr_in*)iap->ifa_addr;
                if (strcmp(interface_name, iap->ifa_name) == 0) {
                    printf("found network interface: '%s'\n", iap->ifa_name);
                    client->interface_index = if_nametoindex(iap->ifa_name);
                    if (!client->interface_index) {
                        printf("\nerror: if_nametoindex failed\n\n");
                        return 1;
                    }
                    found = true;
                    break;
                }
            }
        }

        freeifaddrs(addrs);

        if (!found) {
            printf(
                "\nerror: could not find any network interface matching "
                "'%s'\n\n",
                interface_name);
            return 1;
        }
    }

    // load the client_xdp program and attach it to the network interface
    printf("loading client_xdp...\n");

    client->program =
        xdp_program__open_file("client_xdp.o", "client_xdp", NULL);
    if (libxdp_get_error(client->program)) {
        printf("\nerror: could not load client_xdp program\n\n");
        return 1;
    }

    printf("client_xdp loaded successfully.\n");

    printf("attaching client_xdp to network interface\n");

    int ret = xdp_program__attach(client->program, client->interface_index,
                                  XDP_MODE_NATIVE, 0);
    if (ret == 0) {
        client->attached_native = true;
    } else {
        printf("falling back to skb mode...\n");
        ret = xdp_program__attach(client->program, client->interface_index,
                                  XDP_MODE_SKB, 0);
        if (ret == 0) {
            client->attached_skb = true;
        } else {
            printf(
                "\nerror: failed to attach client_xdp program to "
                "interface\n\n");
            return 1;
        }
    }

    // allow unlimited locking of memory, so all memory needed for packet
    // buffers can be locked
    struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};

    if (setrlimit(RLIMIT_MEMLOCK, &rlim)) {
        printf("\nerror: could not setrlimit\n\n");
        return 1;
    }

    // per-CPU socket setup
    for (int i = 0; i < NUM_CPUS; i++) {
        // allocate buffer for umem
        const int buffer_size = NUM_FRAMES * FRAME_SIZE;

        if (posix_memalign(&client->socket[i].buffer, getpagesize(),
                           buffer_size)) {
            printf("\nerror: could not allocate buffer\n\n");
            return 1;
        }

        // allocate umem
        ret =
            xsk_umem__create(&client->socket[i].umem, client->socket[i].buffer,
                             buffer_size, &client->socket[i].fill_queue,
                             &client->socket[i].complete_queue, NULL);
        if (ret) {
            printf("\nerror: could not create umem\n\n");
            return 1;
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

        int queue_id = i;

        ret = xsk_socket__create(&client->socket[i].xsk, interface_name,
                                 queue_id, client->socket[i].umem,
                                 &client->socket[i].recv_queue,
                                 &client->socket[i].send_queue, &xsk_config);
        if (ret) {
            printf("\nerror: could not create xsk socket [%d]\n\n", queue_id);
            return 1;
        }

        // initialize frame allocator
        for (int j = 0; j < NUM_FRAMES; j++) {
            client->socket[i].frames[j] = j * FRAME_SIZE;
        }

        client->socket[i].num_frames = NUM_FRAMES;

        // set socket queue id for later use
        client->socket[i].queue_id = i;
    }

    // create stats thread
    ret = pthread_create(&client->stats_thread, NULL, stats_thread, client);
    if (ret) {
        printf("\nerror: could not create stats thread\n\n");
        return 1;
    }

    // create socket threads
    for (int i = 0; i < NUM_CPUS; i++) {
        ret = pthread_create(&client->recv_thread[i], NULL, recv_thread,
                             &client->socket[i]);
        if (ret) {
            printf("\nerror: could not create socket recv thread #%d\n\n", i);
            return 1;
        }

        ret = pthread_create(&client->send_thread[i], NULL, send_thread,
                             &client->socket[i]);
        if (ret) {
            printf("\nerror: could not create socket send thread #%d\n\n", i);
            return 1;
        }
    }

    return 0;
}

void client_shutdown(struct client_t* client) {
    assert(client);

    for (int i = 0; i < NUM_CPUS; i++) {
        pthread_join(client->recv_thread[i], NULL);
        pthread_join(client->send_thread[i], NULL);
    }

    for (int i = 0; i < NUM_CPUS; i++) {
        if (client->socket[i].xsk) {
            xsk_socket__delete(client->socket[i].xsk);
        }

        if (client->socket[i].umem) {
            xsk_umem__delete(client->socket[i].umem);
        }

        free(client->socket[i].buffer);
    }

    if (client->program != NULL) {
        if (client->attached_native) {
            xdp_program__detach(client->program, client->interface_index,
                                XDP_MODE_NATIVE, 0);
        }

        if (client->attached_skb) {
            xdp_program__detach(client->program, client->interface_index,
                                XDP_MODE_SKB, 0);
        }

        xdp_program__close(client->program);
    }
}

volatile bool quit;

static void* stats_thread(void* arg) {
    struct client_t* client = (struct client_t*)arg;

    while (!quit) {
        usleep(1000000);

        uint64_t sent_packets = 0;
        for (int i = 0; i < NUM_CPUS; i++) {
            sent_packets += client->socket[i].sent_packets;
        }

        uint64_t sent_delta = sent_packets - client->previous_sent_packets;

        printf("sent delta %" PRId64 "\n", sent_delta);

        client->previous_sent_packets = sent_packets;
    }

    return NULL;
}

bool pin_thread_to_cpu(int cpu) {
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu < 0 || cpu >= num_cpus) return false;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    pthread_t current_thread = pthread_self();

    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

static struct client_t client;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
}

void clean_shutdown_handler(int signal) {
    (void)signal;
    quit = true;
}

static void cleanup() {
    client_shutdown(&client);
    fflush(stdout);
}

uint64_t socket_alloc_frame(struct socket_t* socket) {
    if (socket->num_frames == 0) return INVALID_FRAME;
    socket->num_frames--;
    uint64_t frame = socket->frames[socket->num_frames];
    socket->frames[socket->num_frames] = INVALID_FRAME;
    return frame;
}

void socket_free_frame(struct socket_t* socket, uint64_t frame) {
    assert(socket->num_frames < NUM_FRAMES);
    socket->frames[socket->num_frames] = frame;
    socket->num_frames++;
}

uint64_t socket_num_free_frames(struct socket_t* socket) {
    return socket->num_frames;
}

uint16_t ipv4_checksum(const void* data, size_t header_length) {
    unsigned long sum = 0;

    const uint16_t* p = (const uint16_t*)data;

    while (header_length > 1) {
        sum += *p++;
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        header_length -= 2;
    }

    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return ~sum;
}

int client_generate_packet(void* data, int payload_bytes, uint32_t counter) {
    struct ethhdr* eth = data;
    struct iphdr* ip = data + sizeof(struct ethhdr);
    struct udphdr* udp = (void*)ip + sizeof(struct iphdr);

    // generate ethernet header
    memcpy(eth->h_dest, SERVER_ETHERNET_ADDRESS, ETH_ALEN);
    memcpy(eth->h_source, CLIENT_ETHERNET_ADDRESS, ETH_ALEN);
    eth->h_proto = htons(ETH_P_IP);

    // generate ip header
    ip->ihl = 5;
    ip->version = 4;
    ip->tos = 0x0;
    ip->id = 0;
    ip->frag_off = htons(0x4000);
    ip->ttl = 64;
    ip->tot_len =
        htons(sizeof(struct iphdr) + sizeof(struct udphdr) + payload_bytes);
    ip->protocol = IPPROTO_UDP;
    ip->saddr = htonl(CLIENT_IPV4_ADDRESS);
    ip->daddr = htonl(SERVER_IPV4_ADDRESS);
    ip->check = 0;
    ip->check = ipv4_checksum(ip, sizeof(struct iphdr));

    // generate udp header
    udp->source = htons(CLIENT_PORT);
    udp->dest = htons(SERVER_PORT);
    udp->len = htons(sizeof(struct udphdr) + payload_bytes);
    udp->check = 0;

    // generate udp payload
    uint8_t* payload = (void*)udp + sizeof(struct udphdr);

    for (int i = 0; i < payload_bytes; i++) {
        payload[i] = i;
    }

    return sizeof(struct ethhdr) + sizeof(struct iphdr) +
           sizeof(struct udphdr) + payload_bytes;
}

void socket_send(struct socket_t* socket, int queue_id) {
    // don't do anything if we don't have enough free packets to send a batch
    printf("socket->num_frames = %u\n", socket->num_frames);
    if (socket->num_frames < SEND_BATCH_SIZE) return;

    // queue packets to send
    int send_index;
    int result = xsk_ring_prod__reserve(&socket->send_queue, SEND_BATCH_SIZE,
                                        &send_index);
    if (result == 0) return;

    int num_packets = 0;
    uint64_t packet_address[SEND_BATCH_SIZE];
    int packet_length[SEND_BATCH_SIZE];

    while (true) {
        uint64_t frame = socket_alloc_frame(socket);

        assert(frame != INVALID_FRAME);  // this should never happen

        uint8_t* packet = socket->buffer + frame;

        packet_address[num_packets] = frame;
        packet_length[num_packets] = client_generate_packet(
            packet, PAYLOAD_BYTES, socket->counter + num_packets);

        num_packets++;

        if (num_packets == SEND_BATCH_SIZE) break;
    }

    for (int i = 0; i < num_packets; i++) {
        struct xdp_desc* desc =
            xsk_ring_prod__tx_desc(&socket->send_queue, send_index + i);
        desc->addr = packet_address[i];
        desc->len = packet_length[i];
    }

    xsk_ring_prod__submit(&socket->send_queue, num_packets);

    // send queued packets
    if (xsk_ring_prod__needs_wakeup(&socket->send_queue))
        sendto(xsk_socket__fd(socket->xsk), NULL, 0, MSG_DONTWAIT, NULL, 0);

    // mark completed sent packet frames as free to be reused
    uint32_t complete_index;

    unsigned int completed =
        xsk_ring_cons__peek(&socket->complete_queue,
                            XSK_RING_CONS__DEFAULT_NUM_DESCS, &complete_index);

    printf("completed = %d\n", completed);
    if (completed > 0) {
        for (int i = 0; i < completed; i++) {
            socket_free_frame(socket,
                              *xsk_ring_cons__comp_addr(&socket->complete_queue,
                                                        complete_index++));
        }

        xsk_ring_cons__release(&socket->complete_queue, completed);
        __sync_fetch_and_add(&socket->sent_packets, completed);
        socket->counter += completed;
    }
}

static void* send_thread(void* arg) {
    struct socket_t* socket = (struct socket_t*)arg;

    int queue_id = socket->queue_id;

    printf("started socket send thread for queue #%d\n", queue_id);

    pin_thread_to_cpu(queue_id);

    while (!quit) {
        socket_send(socket, queue_id);
        sleep(1);
    }
}

void socket_recv(struct socket_t* socket, int queue_id) {
    // Check any packet received, in order to drive packet receiving in the
    // kernel.
    uint32_t idx_rx = 0, idx_fq, rcvd;
    rcvd = xsk_ring_cons__peek(&socket->recv_queue, RECV_BATCH_SIZE, &idx_rx);
    if (!rcvd) return;

    /* Stuff the ring with as much frames as possible */
    int stock_frames =
        xsk_prod_nb_free(&socket->fill_queue, socket_num_free_frames(socket));

    if (stock_frames > 0) {
        int ret =
            xsk_ring_prod__reserve(&socket->fill_queue, stock_frames, &idx_fq);

        /* This should not happen, but just in case */
        while (ret != stock_frames)
            ret = xsk_ring_prod__reserve(&socket->fill_queue, rcvd, &idx_fq);

        for (int i = 0; i < stock_frames; i++)
            *xsk_ring_prod__fill_addr(&socket->fill_queue, idx_fq++) =
                socket_alloc_frame(socket);

        xsk_ring_prod__submit(&socket->fill_queue, stock_frames);
    }

    for (int i = 0; i < rcvd; i++) {
        uint64_t addr =
            xsk_ring_cons__rx_desc(&socket->recv_queue, idx_rx)->addr;
        uint32_t len =
            xsk_ring_cons__rx_desc(&socket->recv_queue, idx_rx++)->len;
        // Doing some packet processing here...
        socket_free_frame(socket, addr);
    }
    xsk_ring_cons__release(&socket->recv_queue, rcvd);
}

static void* recv_thread(void* arg) {
    struct socket_t* socket = (struct socket_t*)arg;

    int queue_id = socket->queue_id;
    uint32_t idx_rx = 0;

    /* Stuff the receive path with buffers, we assume we have enough */
    int ret = xsk_ring_prod__reserve(&socket->fill_queue,
                                     XSK_RING_PROD__DEFAULT_NUM_DESCS, &idx_rx);

    if (ret != XSK_RING_PROD__DEFAULT_NUM_DESCS) exit(0);

    for (int i = 0; i < XSK_RING_PROD__DEFAULT_NUM_DESCS; i++)
        *xsk_ring_prod__fill_addr(&socket->fill_queue, idx_rx++) =
            socket_alloc_frame(socket);

    xsk_ring_prod__submit(&socket->fill_queue,
                          XSK_RING_PROD__DEFAULT_NUM_DESCS);

    printf("started socket recv thread for queue #%d\n", queue_id);

    pin_thread_to_cpu(NUM_CPUS + queue_id);

    struct pollfd fds[2];
    int nfds = 1;

    memset(fds, 0, sizeof(fds));
    fds[0].fd = xsk_socket__fd(socket->xsk);
    fds[0].events = POLLIN;

    while (!quit) {
        if (!busy_poll) {
            ret = poll(fds, nfds, 1000);
            if (ret <= 0 || ret > 1) continue;
        }
        socket_recv(socket, queue_id);
    }
}

int main(int argc, char* argv[]) {
    printf("\n[client]\n");

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, clean_shutdown_handler);
    signal(SIGHUP, clean_shutdown_handler);

    if (client_init(&client, INTERFACE_NAME) != 0) {
        cleanup();
        return 1;
    }

    while (!quit) {
        usleep(1000);
    }

    cleanup();

    printf("\n");

    return 0;
}
