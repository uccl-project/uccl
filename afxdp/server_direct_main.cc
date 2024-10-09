#include <assert.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <ifaddrs.h>
#include <inttypes.h>
#include <memory.h>
#include <net/if.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <xdp/libxdp.h>

const char *INTERFACE_NAME = "ens6";

struct server_t {
    int interface_index;
    struct xdp_program *program;
    bool attached_native;
    bool attached_skb;
    int received_packets_fd;
    int num_cpus;
    uint64_t current_received_packets;
    uint64_t previous_received_packets;
};

uint64_t server_get_received_packets(struct server_t *server);

int server_init(struct server_t *server, const char *interface_name) {
    // we can only run xdp programs as root

    if (geteuid() != 0) {
        printf("\nerror: this program must be run as root\n\n");
        return 1;
    }

    // find the network interface that matches the interface name
    {
        bool found = false;

        struct ifaddrs *addrs;
        if (getifaddrs(&addrs) != 0) {
            printf("\nerror: getifaddrs failed\n\n");
            return 1;
        }

        for (struct ifaddrs *iap = addrs; iap != NULL; iap = iap->ifa_next) {
            if (iap->ifa_addr && (iap->ifa_flags & IFF_UP) &&
                iap->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in *sa = (struct sockaddr_in *)iap->ifa_addr;
                if (strcmp(interface_name, iap->ifa_name) == 0) {
                    printf("found network interface: '%s'\n", iap->ifa_name);
                    server->interface_index = if_nametoindex(iap->ifa_name);
                    if (!server->interface_index) {
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

    // load the ebpf_server program and attach it to the network interface

    printf("loading ebpf_server...\n");

    server->program =
        xdp_program__open_file("ebpf_server_direct.o", "ebpf_server_direct", NULL);
    if (libxdp_get_error(server->program)) {
        printf("\nerror: could not load ebpf_server program\n\n");
        return 1;
    }

    printf("ebpf_server loaded successfully.\n");

    printf("attaching ebpf_server to network interface\n");

    int ret = xdp_program__attach(server->program, server->interface_index,
                                  XDP_MODE_NATIVE, 0);
    if (ret == 0) {
        server->attached_native = true;
    } else {
        printf("falling back to skb mode...\n");
        ret = xdp_program__attach(server->program, server->interface_index,
                                  XDP_MODE_SKB, 0);
        if (ret == 0) {
            server->attached_skb = true;
        } else {
            printf(
                "\nerror: failed to attach ebpf_server program to "
                "interface\n\n");
            return 1;
        }
    }

    // look up receive packets map

    server->received_packets_fd =
        bpf_obj_get("/sys/fs/bpf/received_packets_map");
    if (server->received_packets_fd <= 0) {
        printf("\nerror: could not get received packets map: %s\n\n",
               strerror(errno));
        return 1;
    }

    // get number of possible cpus and store the current received packets value
    // in previous, so we don't get large numbers on first update when we run
    // the program repeatedly

    server->num_cpus = libbpf_num_possible_cpus();

    server->previous_received_packets = server_get_received_packets(server);

    return 0;
}

uint64_t server_get_received_packets(struct server_t *server) {
    __u64 thread_received_packets[server->num_cpus];
    int key = 0;
    if (bpf_map_lookup_elem(server->received_packets_fd, &key,
                            thread_received_packets) != 0) {
        printf("\nerror: could not look up received packets map: %s\n\n",
               strerror(errno));
        exit(1);
    }

    uint64_t received_packets = 0;
    for (int i = 0; i < server->num_cpus; i++) {
        received_packets += thread_received_packets[i];
    }

    return received_packets;
}

void server_shutdown(struct server_t *server) {
    assert(server);

    if (server->program != NULL) {
        if (server->attached_native) {
            xdp_program__detach(server->program, server->interface_index,
                                XDP_MODE_NATIVE, 0);
        }
        if (server->attached_skb) {
            xdp_program__detach(server->program, server->interface_index,
                                XDP_MODE_SKB, 0);
        }
        xdp_program__close(server->program);
    }
}

static struct server_t server;

volatile bool quit;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
}

void clean_shutdown_handler(int signal) {
    (void)signal;
    quit = true;
}

static void cleanup() {
    server_shutdown(&server);
    fflush(stdout);
}

int main(int argc, char *argv[]) {
    printf("\n[server]\n");

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, clean_shutdown_handler);
    signal(SIGHUP, clean_shutdown_handler);

    if (server_init(&server, INTERFACE_NAME) != 0) {
        cleanup();
        return 1;
    }

    while (!quit) {
        usleep(1000000);

        uint64_t received_packets = server_get_received_packets(&server);

        uint64_t received_delta =
            received_packets - server.previous_received_packets;

        printf("received delta %" PRId64 "\n", received_delta);

        server.previous_received_packets = received_packets;
    }

    cleanup();

    printf("\n");

    return 0;
}
