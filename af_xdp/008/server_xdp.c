/*
    UDP server XDP program

    Counts IPv4 UDP packets received on port 40000

    USAGE:

        clang -Ilibbpf/src -g -O2 -target bpf -c server_xdp.c -o server_xdp.o
        sudo cat /sys/kernel/debug/tracing/trace_pipe
*/

#include <linux/types.h>
#include <bpf/bpf_helpers.h>
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/if_vlan.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/ipv6.h>
#include <linux/string.h>
#include <linux/udp.h>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define bpf_ntohs(x) __builtin_bswap16(x)
#define bpf_htons(x) __builtin_bswap16(x)
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define bpf_ntohs(x) (x)
#define bpf_htons(x) (x)
#else
#error "Endianness detection needs to be set up for your compiler?!"
#endif

// #define DEBUG 1

#if DEBUG
#define debug_printf bpf_printk
#else  // #if DEBUG
#define debug_printf(...) \
    do {                  \
    } while (0)
#endif  // #if DEBUG

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, int);
    __type(value, __u64);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} received_packets_map SEC(".maps");

SEC("server_xdp")
int server_xdp_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;

    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;

    if ((void *)eth + sizeof(struct ethhdr) < data_end) {
        if (eth->h_proto == __constant_htons(ETH_P_IP))  // IPV4
        {
            struct iphdr *ip = data + sizeof(struct ethhdr);

            if ((void *)ip + sizeof(struct iphdr) < data_end) {
                if (ip->protocol == IPPROTO_UDP)  // UDP
                {
                    struct udphdr *udp = (void *)ip + sizeof(struct iphdr);

                    if ((void *)udp + sizeof(struct udphdr) <= data_end) {
                        if (udp->dest == __constant_htons(40000)) {
                            void *payload = (void *)udp + sizeof(struct udphdr);

                            int payload_bytes = data_end - payload;

                            debug_printf("server received %d byte packet", payload_bytes);

                            int zero = 0;
                            __u64 *packets_received = (__u64 *)bpf_map_lookup_elem(&received_packets_map, &zero);
                            if (packets_received) {
                                __sync_fetch_and_add(packets_received, 1);
                            }

                            return XDP_DROP;
                        }
                    }
                }
            }
        }
    }

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
