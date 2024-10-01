/*
 *  A client timing the roundtrip time of a message sent to a server multiple
 * times. Usage: ./client.out -a <address> -p <port> -b <message_size (bytes)>
 */
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <inttypes.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

#include "util.h"
#include "util_tcp.h"

void error(char *msg) {
    perror(msg);
    exit(0);
}

const int SEND_BATCH_SIZE = 1;
const int RECV_BATCH_SIZE = 32;
const int MAX_INFLIGHT_PKTS = 32;  // tune this to change packet rate
const int SEND_INTV_US = 0;
const int MAX_EVENTS_ONCE = 32;

std::vector<uint64_t> rtts;
std::atomic<uint64_t> sent_packets{0};
std::atomic<uint64_t> inflight_pkts{0};

// TODO(Yang): using multiple TCP connections to saturate the link
static void *send_thread(void *arg) {
    pin_thread_to_cpu(0);

    struct Config *config = (struct Config *)arg;
    int epfd = epoll_create(1);

    // Monitor all sockets for EPOLLOUT
    for (int i = 0; i < NUM_SOCKETS; i++) {
        struct epoll_event ev;
        ev.events = EPOLLOUT | EPOLLEXCLUSIVE;
        ev.data.fd = config->sockfds[i];

        if (epoll_ctl(epfd, EPOLL_CTL_ADD, config->sockfds[i], &ev) == -1) {
            perror("epoll_ctl()\n");
            exit(EXIT_FAILURE);
        }
    }

    uint8_t *wbuffer = (uint8_t *)malloc(config->n_bytes);
    struct epoll_event events[MAX_EVENTS_ONCE];
    while (!quit) {
        if (inflight_pkts >= MAX_INFLIGHT_PKTS) {
            continue;
        }
        int nfds = epoll_wait(epfd, events, MAX_EVENTS_ONCE, -1);
        for (int i = 0; i < nfds; i++) {
            assert(events[i].events & EPOLLOUT);
            for (int j = 0; j < SEND_BATCH_SIZE; j++) {
                auto now = std::chrono::high_resolution_clock::now();
                *(uint64_t *)wbuffer =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        now.time_since_epoch())
                        .count();
                *(uint32_t *)(wbuffer + sizeof(uint64_t)) = inflight_pkts + i;

                send_message(config->n_bytes, events[i].data.fd, wbuffer,
                             &quit);
            }
            inflight_pkts += SEND_BATCH_SIZE;
            sent_packets += SEND_BATCH_SIZE;
        }
        if (SEND_INTV_US) usleep(SEND_INTV_US);
    }
    free(wbuffer);
    return NULL;
}

static void *recv_thread(void *arg) {
    pin_thread_to_cpu(1);

    struct Config *config = (struct Config *)arg;
    int epfd = epoll_create(1);

    // Monitor all sockets for EPOLLIN
    for (int i = 0; i < NUM_SOCKETS; i++) {
        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLEXCLUSIVE;
        ev.data.fd = config->sockfds[i];

        if (epoll_ctl(epfd, EPOLL_CTL_ADD, config->sockfds[i], &ev) == -1) {
            perror("epoll_ctl()\n");
            exit(EXIT_FAILURE);
        }
    }

    uint8_t *rbuffer = (uint8_t *)malloc(config->n_bytes);
    struct epoll_event events[MAX_EVENTS_ONCE];
    while (!quit) {
        int nfds = epoll_wait(epfd, events, MAX_EVENTS_ONCE, -1);
        for (int i = 0; i < nfds; i++) {
            assert(events[i].events & EPOLLIN);
            for (int j = 0; j < RECV_BATCH_SIZE; j++) {
                receive_message(config->n_bytes, events[i].data.fd, rbuffer,
                                &quit);

                uint64_t now_us = *(uint64_t *)rbuffer;
                uint32_t counter = *(uint32_t *)(rbuffer + sizeof(uint64_t));
                auto now = std::chrono::high_resolution_clock::now();
                uint64_t now_us2 =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        now.time_since_epoch())
                        .count();
                uint64_t rtt = now_us2 - now_us;
                rtts.push_back(rtt);
            }
            inflight_pkts -= RECV_BATCH_SIZE;
        }
    }
    free(rbuffer);
    return NULL;
}

static void *stats_thread(void *arg) {
    auto start = std::chrono::high_resolution_clock::now();
    auto start_pkts = sent_packets.load();
    uint64_t previous_sent_packets = sent_packets;
    while (!quit) {
        usleep(1000000);
        auto med_latency = Percentile(rtts, 50);
        auto tail_latency = Percentile(rtts, 99);
        uint64_t sent_delta = sent_packets - previous_sent_packets;
        previous_sent_packets = sent_packets;

        printf("send delta: %lu, med rtt: %lu us, tail rtt: %lu us\n",
               sent_delta, med_latency, tail_latency);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto end_pkts = sent_packets.load();
    uint64_t duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto throughput = (end_pkts - start_pkts) * 1.0 / duration * 1000;

    auto med_latency = Percentile(rtts, 50);
    auto tail_latency = Percentile(rtts, 99);

    printf("Throughput: %.2f Kpkts/s, med rtt: %lu us, tail rtt: %lu us\n",
           throughput, med_latency, tail_latency);

    return NULL;
}

void clean_shutdown_handler(int signal) {
    (void)signal;
    quit = true;
}

int main(int argc, char *argv[]) {
    signal(SIGALRM, clean_shutdown_handler);
    alarm(10);

    struct sockaddr_in serv_addr;
    struct hostent *server;

    struct Config config = get_config(argc, argv);
    server = gethostbyname(config.address);
    if (server == NULL) {
        fprintf(stderr, "ERROR, no such host\n");
        exit(0);
    }
    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(config.port);

    for (int i = 0; i < NUM_SOCKETS; i++) {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            error("ERROR opening socket");
        }
        // Connect and set nonblocking and nodelay
        if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) <
            0) {
            error("ERROR connecting");
        }
        fcntl(sockfd, F_SETFL, O_NONBLOCK);
        int flag = 1;
        setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
                   sizeof(int));
        config.sockfds[i] = sockfd;
    }

    printf("Connection successful! Starting...\n");
    fflush(stdout);

    pthread_t recv_thread_ctl;
    if (pthread_create(&recv_thread_ctl, NULL, recv_thread, &config)) {
        error("ERROR creating recv thread");
    }

    pthread_t send_thread_ctl;
    if (pthread_create(&send_thread_ctl, NULL, send_thread, &config)) {
        error("ERROR creating send thread");
    }

    pthread_t stats_thread_ctl;
    if (pthread_create(&stats_thread_ctl, NULL, stats_thread, NULL)) {
        printf("\nerror: could not create stats thread\n\n");
        return 1;
    }

    while (!quit) {
        usleep(1000);
    }

    pthread_join(recv_thread_ctl, NULL);
    pthread_join(send_thread_ctl, NULL);
    pthread_join(stats_thread_ctl, NULL);

    for (int i = 0; i < NUM_SOCKETS; i++) {
        close(config.sockfds[i]);
    }

    return 0;
}
