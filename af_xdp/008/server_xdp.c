/*
    UDP server XDP program

    Counts IPv4 UDP packets received on port 40000

    USAGE:

        clang -Ilibbpf/src -g -O2 -target bpf -c server_xdp.c -o server_xdp.o
        sudo cat /sys/kernel/debug/tracing/trace_pipe
*/
// fmt: off
#include <linux/types.h>
// fmt: on
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

static inline __u16 compute_ip_checksum(struct iphdr *ip) {
    __u32 csum = 0;
    __u16 *next_ip_u16 = (__u16 *)ip;

    ip->check = 0;

    for (int i = 0; i < (sizeof(*ip) >> 1); i++) {
        csum += *next_ip_u16++;
    }

    return ~((csum & 0xffff) + (csum >> 16));
}

static inline void prepare_packet(struct ethhdr *eth, struct iphdr *ip,
                                  struct udphdr *udp) {
    unsigned char tmp_mac[ETH_ALEN];
    __be32 tmp_ip;
    __be16 tmp_port;

    memcpy(tmp_mac, eth->h_source, ETH_ALEN);
    memcpy(eth->h_source, eth->h_dest, ETH_ALEN);
    memcpy(eth->h_dest, tmp_mac, ETH_ALEN);

    tmp_ip = ip->saddr;
    ip->saddr = ip->daddr;
    ip->daddr = tmp_ip;

    tmp_port = udp->source;
    udp->source = udp->dest;
    udp->dest = tmp_port;

    udp->check = 0;
    ip->check = compute_ip_checksum(ip);
}

SEC("server_xdp")
int server_xdp_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)eth + sizeof(struct ethhdr) > data_end) return XDP_PASS;
    if (eth->h_proto != __constant_htons(ETH_P_IP)) return XDP_PASS;

    struct iphdr *ip = data + sizeof(struct ethhdr);
    if ((void *)ip + sizeof(struct iphdr) > data_end) return XDP_PASS;
    if (ip->protocol != IPPROTO_UDP) return XDP_PASS;

    struct udphdr *udp = (void *)ip + sizeof(struct iphdr);
    if ((void *)udp + sizeof(struct udphdr) > data_end) return XDP_PASS;
    if (udp->dest != __constant_htons(40000)) return XDP_PASS;

    void *payload = (void *)udp + sizeof(struct udphdr);
    int payload_bytes = data_end - payload;

    debug_printf("server received %d byte packet", payload_bytes);

    int zero = 0;
    __u64 *packets_received =
        (__u64 *)bpf_map_lookup_elem(&received_packets_map, &zero);

    if (packets_received) {
        __sync_fetch_and_add(packets_received, 1);
    }

    prepare_packet(eth, ip, udp);

    return XDP_TX;
}

char _license[] SEC("license") = "GPL";
