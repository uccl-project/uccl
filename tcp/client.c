/*
 *  A client timing the roundtrip time of a message sent to a server multiple times.
 *  Usage: ./client.out -a <address> -p <port> -b <message_size (bytes)>
 */
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <inttypes.h>
#include <ctype.h>
#include <chrono>

#include "connection.h"

void error(char *msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[]) {
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
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, 
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(config.port);

    // Connect and set nonblocking and nodelay
    if (connect(sockfd,(struct sockaddr *)&serv_addr,sizeof(serv_addr)) < 0) {
        error("ERROR connecting");
    }
    fcntl(sockfd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag, sizeof(int));

    printf("Connection successful! Starting...");
    fflush( stdout );

    // Timed send-receive loop
    uint64_t *times_send = (uint64_t*)malloc(sizeof(uint64_t) * N_ROUNDS);
    uint64_t *times_recv = (uint64_t*)malloc(sizeof(uint64_t) * N_ROUNDS);
    for (size_t i = 0; i < N_ROUNDS; i++) {

        auto tstart = std::chrono::high_resolution_clock::now();        

        send_message(config.n_bytes, sockfd, wbuffer);
        auto tsend = std::chrono::high_resolution_clock::now();
        receive_message(config.n_bytes, sockfd, rbuffer);

        auto tend = std::chrono::high_resolution_clock::now();

        times_send[i] = std::chrono::duration_cast<std::chrono::microseconds>(tsend - tstart).count();
        times_recv[i] = std::chrono::duration_cast<std::chrono::microseconds>(tend - tsend).count();
    }
    close(sockfd);
    printf("Done!\nSummary: (time_send,\ttime_recv)");
    for (size_t i = 0; i < N_ROUNDS; i++) {
        printf("(%lu,\t%lu,\t%ld us)\n", times_send[i], times_recv[i], (times_send[i] + times_recv[i]));
    }
    free(times_send);
    free(times_recv);
    free(rbuffer);
    free(wbuffer);

    return 0;
}

