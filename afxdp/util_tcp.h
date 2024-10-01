#include <errno.h>
#include <stdint.h>

// Need around 100k to echieve 8Gbps
#define PAYLOAD_BYTES 32
#define DEFAULT_PORT 40000
#define DEFAULT_ADDRESS "127.0.0.1"
#define NUM_SOCKETS 8  // only impact tcp_ep
#define MAX_EVENTS_ONCE 32

struct Config {
    char *address;
    int port;
    int n_bytes;
    int sockfds[NUM_SOCKETS];
};

void print_config(struct Config config) {
    printf("Address: %s, Port: %d, N_bytes: %d\n", config.address, config.port,
           config.n_bytes);
}

// Parse command line args to extract config. Default values used when arg
// missing
struct Config get_config(int argc, char *argv[]) {
    struct Config config;
    int c;
    config.n_bytes = PAYLOAD_BYTES;
    config.port = DEFAULT_PORT;
    config.address = DEFAULT_ADDRESS;

    while ((c = getopt(argc, argv, "a:p:b:")) != -1) {
        switch (c) {
            case 'a':
                config.address = optarg;
                break;
            case 'p':
                config.port = atoi(optarg);
                break;
            case 'b':
                config.n_bytes = atoi(optarg);
                break;
            default: /* '?' */
                fprintf(stderr, "Usage: %s [-b bytes] [-a address] [-p port]\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    print_config(config);
    return config;
}

void panic(char *msg) {
    perror(msg);
    exit(0);
}

// Starts a measure
uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// Obtains ticks successive to a rdtsc() call
uint64_t rdtscp() {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

volatile bool quit;

// Reads from the given socket into the given buffer n_bytes bytes
int receive_message(size_t n_bytes, int sockfd, uint8_t *buffer,
                    volatile bool *quit) {
    int bytes_read = 0;
    int r;
    while (bytes_read < n_bytes && !(*quit)) {
        // Make sure we read exactly n_bytes
        r = read(sockfd, buffer + bytes_read, n_bytes - bytes_read);
        if (r < 0 && !(errno == EAGAIN || errno == EWOULDBLOCK)) {
            panic("ERROR reading from socket");
        }
        if (r > 0) {
            bytes_read += r;
        }
    }
    return bytes_read;
}

int receive_message_early_return(size_t n_bytes, int sockfd, uint8_t *buffer,
                                 volatile bool *quit) {
    int r = read(sockfd, buffer, n_bytes);
    // Indicate this read would block.
    if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return 0;
    }
    return r + receive_message(n_bytes - r, sockfd, buffer + r, quit);
}

// Writes n_bytes from the given buffer to the given socekt
int send_message(size_t n_bytes, int sockfd, uint8_t *buffer,
                 volatile bool *quit) {
    int bytes_sent = 0;
    int r;
    while (bytes_sent < n_bytes && !(*quit)) {
        // Make sure we write exactly n_bytes
        r = write(sockfd, buffer + bytes_sent, n_bytes - bytes_sent);
        if (r < 0 && !(errno == EAGAIN || errno == EWOULDBLOCK)) {
            panic("ERROR writing to socket");
        }
        if (r > 0) {
            bytes_sent += r;
        }
    }
    return bytes_sent;
}

int send_message_early_return(size_t n_bytes, int sockfd, uint8_t *buffer,
                              volatile bool *quit) {
    int r = write(sockfd, buffer, n_bytes);
    // Indicate this write would block.
    if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return 0;
    }
    return r + send_message(n_bytes - r, sockfd, buffer + r, quit);
}
