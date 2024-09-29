/*
    UDP client XDP program

    Counts IPv4 UDP packets received on port 40000

    USAGE:

        clang -Ilibbpf/src -g -O2 -target bpf -c client_xdp.c -o client_xdp.o
        sudo cat /sys/kernel/debug/tracing/trace_pipe
*/

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

SEC("client_xdp")
int client_xdp_filter(struct xdp_md *ctx) {
    // nop
    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
