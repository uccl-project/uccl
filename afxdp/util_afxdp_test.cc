#include "util_afxdp.h"

#include <pthread.h>
#include <signal.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include "util.h"
#include "util_umem.h"

using namespace uccl;

const uint32_t NUM_QUEUES = 1;

const char* INTERFACE_NAME = "ens6";
const uint8_t SERVER_ETHERNET_ADDRESS[] = {0x0a, 0xff, 0xea, 0x86, 0x04, 0xd9};
const uint8_t CLIENT_ETHERNET_ADDRESS[] = {0x0a, 0xff, 0xdf, 0x30, 0xe7, 0x59};
const uint32_t SERVER_IPV4_ADDRESS = 0xac1f16f9;  // 172.31.22.249
const uint32_t CLIENT_IPV4_ADDRESS = 0xac1f10c6;  // 172.31.16.198
const uint16_t SERVER_PORT = 40000;
const uint16_t CLIENT_PORT[8] = {40000, 40001, 40002, 40003,
                                 40004, 40005, 40006, 40007};

const int SEND_BATCH_SIZE = 1;
const int RECV_BATCH_SIZE = 32;
// 256 is reserved for xdp_meta, 42 is reserved for eth+ip+udp
// Max payload under AFXDP is 4096-256-42;
const int PAYLOAD_BYTES = 64;
// tune this to change packet rate
const int MAX_INFLIGHT_PKTS = 1;
// sleep gives unstable rate and latency
const int SEND_INTV_US = 0;
const int RTO_US = 2000;

const bool busy_poll = true;

#define NUM_FRAMES (4096 * 16)
#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE
#define INVALID_FRAME UINT64_MAX

struct socket_t {
    AFXDPSocket* afxdp_socket;
    std::atomic<uint64_t> sent_packets;
    uint64_t last_stall_time;
    uint32_t counter;
    std::vector<uint64_t> rtts;
    std::mutex rtts_lock;
};

struct client_t {
    struct socket_t socket[NUM_QUEUES];
    pthread_t send_thread[NUM_QUEUES];
    pthread_t recv_thread[NUM_QUEUES];
    pthread_t stats_thread;
    uint64_t previous_sent_packets;
};

static struct client_t client;
std::atomic<uint64_t> inflight_pkts{0};
volatile bool quit;

static void* stats_thread(void* arg);
static void* send_thread(void* arg);
static void* recv_thread(void* arg);

int client_init(struct client_t* client, const char* interface_name) {
    AFXDPFactory::init(interface_name, "ebpf_client.o", "ebpf_client");

    // per-CPU socket setup
    for (int i = 0; i < NUM_QUEUES; i++) {
        client->socket[i].afxdp_socket =
            AFXDPFactory::CreateSocket(i, NUM_FRAMES);
    }

    int ret;

    // create socket threads
    for (int i = 0; i < NUM_QUEUES; i++) {
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

    // create stats thread
    ret = pthread_create(&client->stats_thread, NULL, stats_thread, client);
    if (ret) {
        printf("\nerror: could not create stats thread\n\n");
        return 1;
    }

    return 0;
}

void client_shutdown(struct client_t* client) {
    assert(client);

    for (int i = 0; i < NUM_QUEUES; i++) {
        pthread_join(client->recv_thread[i], NULL);
        pthread_join(client->send_thread[i], NULL);
    }
    pthread_join(client->stats_thread, NULL);

    AFXDPFactory::shutdown();
}

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

int client_generate_packet(void* data, int payload_bytes, uint32_t counter,
                           int queue_id) {
    struct ethhdr* eth = (struct ethhdr*)data;
    struct iphdr* ip = (struct iphdr*)((char*)data + sizeof(struct ethhdr));
    struct udphdr* udp = (struct udphdr*)((char*)ip + sizeof(struct iphdr));

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

    // generate udp header: using different ports to bypass per-flow rate
    // limiting
    udp->source = htons(
        CLIENT_PORT[counter % (sizeof(CLIENT_PORT) / sizeof(CLIENT_PORT[0]))]);
    udp->dest = htons(SERVER_PORT);
    udp->len = htons(sizeof(struct udphdr) + payload_bytes);
    udp->check = 0;

    // generate udp payload
    uint8_t* payload = (uint8_t*)((char*)udp + sizeof(struct udphdr));
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          now.time_since_epoch())
                          .count();
    assert(payload_bytes >= sizeof(uint64_t) + sizeof(uint32_t));
    memcpy(payload, &now_us, sizeof(uint64_t));
    memcpy(payload + sizeof(uint64_t), &counter, sizeof(uint32_t));

    return sizeof(struct ethhdr) + sizeof(struct iphdr) +
           sizeof(struct udphdr) + payload_bytes;
}

void socket_send(struct socket_t* socket, int queue_id) {
    if (inflight_pkts >= MAX_INFLIGHT_PKTS) {
        auto now_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();
        if (socket->last_stall_time == 0) {
            socket->last_stall_time = now_us;
        } else if (now_us - socket->last_stall_time > RTO_US) {
            // These inflight packets get lost, we just ignore them
            printf("queue %d tx stall detected, forcing tx...\n", queue_id);
            inflight_pkts = 0;
        }
        return;
    }
    socket->last_stall_time = 0;

    std::vector<AFXDPSocket::frame_desc> frames;
    for (int i = 0; i < SEND_BATCH_SIZE; i++) {
        // the 256B before frame_offset is xdp metedata
        uint64_t frame_offset = socket->afxdp_socket->frame_pool_->pop();
        uint8_t* packet =
            (uint8_t*)socket->afxdp_socket->umem_buffer_ + frame_offset;
        uint32_t frame_len = client_generate_packet(
            packet, PAYLOAD_BYTES, socket->counter + i, queue_id);
        FrameBuf::mark_pulltime_free(frame_offset,
                                     socket->afxdp_socket->umem_buffer_);
        frames.emplace_back(AFXDPSocket::frame_desc({frame_offset, frame_len}));
    }
    inflight_pkts += SEND_BATCH_SIZE;
    auto completed = socket->afxdp_socket->send_packets(frames);
    socket->sent_packets += completed;
    socket->counter += completed;
}

void socket_recv(struct socket_t* socket, int queue_id) {
    // Check any packet received, in order to drive packet receiving path for
    // other kernel transport.
    auto frames = socket->afxdp_socket->recv_packets(RECV_BATCH_SIZE);
    uint32_t rcvd = frames.size();
    inflight_pkts -= rcvd;

    VLOG(3) << "rx fill_queue rcvd = " << rcvd
            << ", inflight_pkts = " << inflight_pkts.load();

    for (int i = 0; i < rcvd; i++) {
        uint64_t frame_offset = frames[i].frame_offset;
        uint32_t len = frames[i].frame_len;
        uint8_t* pkt =
            (uint8_t*)socket->afxdp_socket->umem_buffer_ + frame_offset;
        VLOG(3) << "recv: " << std::hex << frame_offset << " " << std::dec
                << len;

        // Doing some packet processing here...
        struct ethhdr* eth = (struct ethhdr*)pkt;
        struct iphdr* ip = (struct iphdr*)((char*)pkt + sizeof(struct ethhdr));
        struct udphdr* udp = (struct udphdr*)((char*)ip + sizeof(struct iphdr));
        uint8_t* payload = (uint8_t*)((char*)udp + sizeof(struct udphdr));
        uint64_t now_us = *(uint64_t*)payload;
        uint32_t counter = *(uint32_t*)(payload + sizeof(uint64_t));
        auto now = std::chrono::high_resolution_clock::now();
        uint64_t now_us2 =
            std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch())
                .count();
        uint64_t rtt = now_us2 - now_us;
        {
            std::lock_guard<std::mutex> lock(socket->rtts_lock);
            socket->rtts.push_back(rtt);
        }
        socket->afxdp_socket->frame_pool_->push(frame_offset);
    }
}

static void* send_thread(void* arg) {
    struct socket_t* socket = (struct socket_t*)arg;

    int queue_id = socket->afxdp_socket->queue_id_;

    printf("started socket send thread for queue #%d\n", queue_id);

    pin_thread_to_cpu(queue_id);

    while (!quit) {
        socket_send(socket, queue_id);
        if (SEND_INTV_US) usleep(SEND_INTV_US);
    }
    return NULL;
}

static void* recv_thread(void* arg) {
    struct socket_t* socket = (struct socket_t*)arg;
    int queue_id = socket->afxdp_socket->queue_id_;
    printf("started socket recv thread for queue #%d\n", queue_id);

    pin_thread_to_cpu(NUM_QUEUES + queue_id);

    struct pollfd fds[2];
    int nfds = 1;

    memset(fds, 0, sizeof(fds));
    fds[0].fd = xsk_socket__fd(socket->afxdp_socket->xsk_);
    fds[0].events = POLLIN;

    int ret;
    while (!quit) {
        if (!busy_poll) {
            ret = poll(fds, nfds, 1000);
            if (ret <= 0 || ret > 1) continue;
        }
        socket_recv(socket, queue_id);
    }
    return NULL;
}

uint64_t aggregate_sent_packets(struct client_t* client) {
    uint64_t sent_packets = 0;
    for (int i = 0; i < NUM_QUEUES; i++)
        sent_packets += client->socket[i].sent_packets.load();
    return sent_packets;
}

std::vector<uint64_t> aggregate_rtts(struct client_t* client) {
    std::vector<uint64_t> rtts;
    for (int i = 0; i < NUM_QUEUES; i++) {
        std::lock_guard<std::mutex> lock(client->socket[i].rtts_lock);
        rtts.insert(rtts.end(), client->socket[i].rtts.begin(),
                    client->socket[i].rtts.end());
    }
    return rtts;
}

static void* stats_thread(void* arg) {
    struct client_t* client = (struct client_t*)arg;

    auto start = std::chrono::high_resolution_clock::now();
    auto start_pkts = aggregate_sent_packets(client);
    auto end = start;
    auto end_pkts = start_pkts;
    while (!quit) {
        // Put before usleep to avoid counting it for tput calculation
        end = std::chrono::high_resolution_clock::now();
        end_pkts = aggregate_sent_packets(client);
        usleep(1000000);
        uint64_t sent_packets = aggregate_sent_packets(client);
        auto rtts = aggregate_rtts(client);
        auto med_latency = Percentile(rtts, 50);
        auto tail_latency = Percentile(rtts, 99);
        uint64_t sent_delta = sent_packets - client->previous_sent_packets;
        client->previous_sent_packets = sent_packets;

        printf("send delta: %lu, med rtt: %lu us, tail rtt: %lu us\n",
               sent_delta, med_latency, tail_latency);
    }
    uint64_t duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto throughput = (end_pkts - start_pkts) * 1.0 / duration * 1000;
    // 42B: eth+ip+udp, 24B: 4B FCS + 8B frame delimiter + 12B interframe gap
    auto bw_gbps = throughput * (PAYLOAD_BYTES + 42 + 24) * 8.0 / 1024 / 1024;

    auto rtts = aggregate_rtts(client);
    auto med_latency = Percentile(rtts, 50);
    auto tail_latency = Percentile(rtts, 99);

    printf(
        "Throughput: %.2f Kpkts/s, BW: %.2f Gbps, med rtt: %lu us, tail rtt: "
        "%lu us\n",
        throughput, bw_gbps, med_latency, tail_latency);

    return NULL;
}

// TO RUN THE TEST:
// On server: sudo ./server
// On client: sudo ./util_afxdp_test
int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    printf("\n[client]\n");

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, clean_shutdown_handler);
    signal(SIGHUP, clean_shutdown_handler);
    signal(SIGALRM, clean_shutdown_handler);
    alarm(10);

    int pshared;
    int ret;

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
