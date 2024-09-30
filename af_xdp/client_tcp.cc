/*
 *  A client timing the roundtrip time of a message sent to a server multiple
 * times. Usage: ./client.out -a <address> -p <port> -b <message_size (bytes)>
 */
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
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <vector>
#include <atomic>
#include <mutex>

#include "util.h"
#include "util_tcp.h"

void error(char *msg) {
    perror(msg);
    exit(0);
}

std::vector<uint64_t> rtts;
std::atomic<uint64_t> sent_packets {0};

static void *stats_thread(void *arg) {
    uint64_t previous_sent_packets = sent_packets;
    while (true) {
        usleep(1000000);
        auto med_latency = Percentile(rtts, 50);
        auto tail_latency = Percentile(rtts, 99);
        uint64_t sent_delta = sent_packets - previous_sent_packets;
        printf("send delta: %lu, med rtt: %lu us, tail rtt: %lu us\n",
               sent_delta, med_latency, tail_latency);

        previous_sent_packets = sent_packets;
    }

    return NULL;
}

void clean_shutdown_handler(int signal) {
    (void)signal;
    exit(0);
}

int main(int argc, char *argv[]) {
    signal(SIGALRM, clean_shutdown_handler);
    alarm(10);

    int sockfd;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    struct Config config = get_config(argc, argv);

    // Init buffers
    uint8_t *rbuffer = (uint8_t *)malloc(config.n_bytes);
    uint8_t *wbuffer = (uint8_t *)malloc(config.n_bytes);

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        error("ERROR opening socket");
    }
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

    // Connect and set nonblocking and nodelay
    if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        error("ERROR connecting");
    }
    fcntl(sockfd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag, sizeof(int));

    pthread_t stats_thread_ctl;
    // create stats thread
    int ret = pthread_create(&stats_thread_ctl, NULL, stats_thread, NULL);
    if (ret) {
        printf("\nerror: could not create stats thread\n\n");
        return 1;
    }

    printf("Connection successful! Starting...\n");
    fflush(stdout);

    // Timed send-receive loop
    for (size_t i = 0;; i++) {
        auto tstart = std::chrono::high_resolution_clock::now();
        send_message(config.n_bytes, sockfd, wbuffer);
        receive_message(config.n_bytes, sockfd, rbuffer);
        auto tend = std::chrono::high_resolution_clock::now();

        rtts.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart)
                .count());
        sent_packets++;
    }
    close(sockfd);
    free(rbuffer);
    free(wbuffer);

    return 0;
}
