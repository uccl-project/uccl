/*
    UDP client (userspace)

    Runs on Ubuntu 22.04 LTS 64bit with Linux Kernel 6.5+ *ONLY*

    Derived from https://github.com/xdp-project/xdp-tutorial/tree/master/advanced03-AF_XDP
*/

#define _GNU_SOURCE

#include <memory.h>
#include <stdio.h>
#include <signal.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <ifaddrs.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <xdp/xsk.h>
#include <xdp/libxdp.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <net/if.h>
#include <linux/if_link.h>
#include <linux/if_ether.h>
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <inttypes.h>

const char * INTERFACE_NAME = "enp8s0f0";

const uint8_t CLIENT_ETHERNET_ADDRESS[] = { 0xa0, 0x36, 0x9f, 0x68, 0xeb, 0x98 };

const uint8_t SERVER_ETHERNET_ADDRESS[] = { 0xa0, 0x36, 0x9f, 0x1e, 0x1a, 0xec };

const uint32_t CLIENT_IPV4_ADDRESS = 0xc0a8b779; // 192.168.183.121

const uint32_t SERVER_IPV4_ADDRESS = 0xc0a8b77c; // 192.168.183.124

const uint16_t SERVER_PORT = 40000;

const uint16_t CLIENT_PORT = 40000;

const int PAYLOAD_BYTES = 100;

const int SEND_BATCH_SIZE = 256;

#define NUM_FRAMES 4096

#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE

#define INVALID_FRAME UINT64_MAX

struct client_t
{
    int interface_index;
    struct xdp_program * program;
    bool attached_native;
    bool attached_skb;
    void * buffer;
    struct xsk_umem * umem;
    struct xsk_ring_prod send_queue;
    struct xsk_ring_cons complete_queue;
    struct xsk_ring_prod fill_queue; // not used
    struct xsk_socket * xsk;
    uint64_t frames[NUM_FRAMES];
    uint32_t num_frames;
    uint64_t current_sent_packets;
    uint64_t previous_sent_packets;
    pthread_t stats_thread;
};

volatile bool quit;

static void * stats_thread( void * arg )
{
    struct client_t * client = (struct client_t*) arg;

    while ( !quit )
    {
        usleep( 1000000 );

        uint64_t sent_packets = client->current_sent_packets;

        uint64_t sent_delta = sent_packets - client->previous_sent_packets;

        printf( "sent delta %" PRId64 "\n", sent_delta );

        client->previous_sent_packets = sent_packets;
    }

    return NULL;
}

bool pin_thread_to_cpu( int cpu ) 
{
    int num_cpus = sysconf( _SC_NPROCESSORS_ONLN );
    if ( cpu < 0 || cpu >= num_cpus  )
        return false;

    cpu_set_t cpuset;
    CPU_ZERO( &cpuset );
    CPU_SET( cpu, &cpuset );

    pthread_t current_thread = pthread_self();    

    pthread_setaffinity_np( current_thread, sizeof(cpu_set_t), &cpuset );
}

int client_init( struct client_t * client, const char * interface_name )
{
    // we can only run xdp programs as root

    if ( geteuid() != 0 ) 
    {
        printf( "\nerror: this program must be run as root\n\n" );
        return 1;
    }

    // find the network interface that matches the interface name
    {
        bool found = false;

        struct ifaddrs * addrs;
        if ( getifaddrs( &addrs ) != 0 )
        {
            printf( "\nerror: getifaddrs failed\n\n" );
            return 1;
        }

        for ( struct ifaddrs * iap = addrs; iap != NULL; iap = iap->ifa_next ) 
        {
            if ( iap->ifa_addr && ( iap->ifa_flags & IFF_UP ) && iap->ifa_addr->sa_family == AF_INET )
            {
                struct sockaddr_in * sa = (struct sockaddr_in*) iap->ifa_addr;
                if ( strcmp( interface_name, iap->ifa_name ) == 0 )
                {
                    printf( "found network interface: '%s'\n", iap->ifa_name );
                    client->interface_index = if_nametoindex( iap->ifa_name );
                    if ( !client->interface_index ) 
                    {
                        printf( "\nerror: if_nametoindex failed\n\n" );
                        return 1;
                    }
                    found = true;
                    break;
                }
            }
        }

        freeifaddrs( addrs );

        if ( !found )
        {
            printf( "\nerror: could not find any network interface matching '%s'\n\n", interface_name );
            return 1;
        }
    }

    // load the client_xdp program and attach it to the network interface

    printf( "loading client_xdp...\n" );

    client->program = xdp_program__open_file( "client_xdp.o", "client_xdp", NULL );
    if ( libxdp_get_error( client->program ) ) 
    {
        printf( "\nerror: could not load client_xdp program\n\n");
        return 1;
    }

    printf( "client_xdp loaded successfully.\n" );

    printf( "attaching client_xdp to network interface\n" );

    int ret = xdp_program__attach( client->program, client->interface_index, XDP_MODE_NATIVE, 0 );
    if ( ret == 0 )
    {
        client->attached_native = true;
    } 
    else
    {
        printf( "falling back to skb mode...\n" );
        ret = xdp_program__attach( client->program, client->interface_index, XDP_MODE_SKB, 0 );
        if ( ret == 0 )
        {
            client->attached_skb = true;
        }
        else
        {
            printf( "\nerror: failed to attach client_xdp program to interface\n\n" );
            return 1;
        }
    }

    // allow unlimited locking of memory, so all memory needed for packet buffers can be locked

    struct rlimit rlim = { RLIM_INFINITY, RLIM_INFINITY };

    if ( setrlimit( RLIMIT_MEMLOCK, &rlim ) ) 
    {
        printf( "\nerror: could not setrlimit\n\n");
        return 1;
    }

    // allocate buffer for umem

    const int buffer_size = NUM_FRAMES * FRAME_SIZE;

    if ( posix_memalign( &client->buffer, getpagesize(), buffer_size ) ) 
    {
        printf( "\nerror: could not allocate buffer\n\n" );
        return 1;
    }

    // allocate umem

    ret = xsk_umem__create( &client->umem, client->buffer, buffer_size, &client->fill_queue, &client->complete_queue, NULL );
    if ( ret ) 
    {
        printf( "\nerror: could not create umem\n\n" );
        return 1;
    }

    // create xsk socket and assign to network interface queue 0

    struct xsk_socket_config xsk_config;

    memset( &xsk_config, 0, sizeof(xsk_config) );

    xsk_config.rx_size = 0;
    xsk_config.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;
    xsk_config.xdp_flags = 0;
    xsk_config.bind_flags = 0;
    xsk_config.libbpf_flags = XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD;

    int queue_id = 0;

    ret = xsk_socket__create( &client->xsk, interface_name, queue_id, client->umem, NULL, &client->send_queue, &xsk_config );
    if ( ret )
    {
        printf( "\nerror: could not create xsk socket\n\n" );
        return 1;
    }

    // pin this thread to CPU 0 so it matches the queue id

    pin_thread_to_cpu( 0 );

    // initialize frame allocator

    for ( int i = 0; i < NUM_FRAMES; i++ )
    {
        client->frames[i] = i * FRAME_SIZE;
    }

    client->num_frames = NUM_FRAMES;

    // create stats thread

    ret = pthread_create( &client->stats_thread, NULL, stats_thread, client );
    if ( ret ) 
    {
        printf( "\nerror: could not create stats thread\n\n" );
        return 1;
    }

    printf("client init: OK!\n");
    return 0;
}

void client_shutdown( struct client_t * client )
{
    assert( client );

    if ( client->program != NULL )
    {
        if ( client->attached_native )
        {
            xdp_program__detach( client->program, client->interface_index, XDP_MODE_NATIVE, 0 );
        }

        if ( client->attached_skb )
        {
            xdp_program__detach( client->program, client->interface_index, XDP_MODE_SKB, 0 );
        }

        xdp_program__close( client->program );

        xsk_socket__delete( client->xsk );

        xsk_umem__delete( client->umem );

        free( client->buffer );
    }
}

static struct client_t client;

void interrupt_handler( int signal )
{
    (void) signal; quit = true;
}

void clean_shutdown_handler( int signal )
{
    (void) signal;
    quit = true;
}

static void cleanup()
{
    client_shutdown( &client );
    fflush( stdout );
}

uint64_t client_alloc_frame( struct client_t * client )
{
    if ( client->num_frames == 0 )
        return INVALID_FRAME;
    client->num_frames--;
    uint64_t frame = client->frames[client->num_frames];
    client->frames[client->num_frames] = INVALID_FRAME;
    return frame;
}

void client_free_frame( struct client_t * client, uint64_t frame )
{
    assert( client->num_frames < NUM_FRAMES );
    client->frames[client->num_frames] = frame;
    client->num_frames++;
}

uint16_t ipv4_checksum( const void * data, size_t header_length )
{
    unsigned long sum = 0;

    const uint16_t * p = (const uint16_t*) data;

    while ( header_length > 1 )
    {
        sum += *p++;
        if ( sum & 0x80000000 )
        {
            sum = ( sum & 0xFFFF ) + ( sum >> 16 );
        }
        header_length -= 2;
    }

    while ( sum >> 16 )
    {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return ~sum;
}

int client_generate_packet( void * data, int payload_bytes )
{
    struct ethhdr * eth = data;
    struct iphdr  * ip  = data + sizeof( struct ethhdr );
    struct udphdr * udp = (void*) ip + sizeof( struct iphdr );

    // generate ethernet header

    memcpy( eth->h_dest, SERVER_ETHERNET_ADDRESS, ETH_ALEN );
    memcpy( eth->h_source, CLIENT_ETHERNET_ADDRESS, ETH_ALEN );
    eth->h_proto = htons( ETH_P_IP );

    // generate ip header

    ip->ihl      = 5;
    ip->version  = 4;
    ip->tos      = 0x0;
    ip->id       = 0;
    ip->frag_off = htons(0x4000);
    ip->ttl      = 64;
    ip->tot_len  = htons( sizeof(struct iphdr) + sizeof(struct udphdr) + payload_bytes );
    ip->protocol = IPPROTO_UDP;
    ip->saddr    = CLIENT_IPV4_ADDRESS;
    ip->daddr    = SERVER_IPV4_ADDRESS;
    ip->check    = 0; 
    ip->check    = ipv4_checksum( ip, sizeof( struct iphdr ) );

    // generate udp header

    udp->source  = htons( CLIENT_PORT );
    udp->dest    = htons( SERVER_PORT );
    udp->len     = htons( sizeof(struct udphdr) + payload_bytes );
    udp->check   = 0;

    // generate udp payload

    uint8_t * payload = (void*) udp + sizeof( struct udphdr );

    for ( int i = 0; i < payload_bytes; i++ )
    {
        payload[i] = i;
    }

    return sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr) + payload_bytes; 
}

int kick_tx( struct client_t * client )
{
    int err = 0;
    int ret;

    ret = sendto(xsk_socket__fd(client->xsk), NULL, 0, MSG_DONTWAIT, NULL, 0);
    if ( ret < 0 )
    {
        // On error, -1 is returned, and errno is set */
        fprintf(stderr, "WARN: %s() sendto() failed with errno:%d\n", __func__, errno);
        err = errno;
    }
    return err;
}

void client_update( struct client_t * client )
{
    // don't do anything if we don't have enough free packets to send a batch

    if ( client->num_frames < SEND_BATCH_SIZE )
        return;

    // queue packets to send

    int send_index;
    int result = xsk_ring_prod__reserve( &client->send_queue, SEND_BATCH_SIZE, &send_index );
    if ( result == 0 ) 
    {
        return;
    }

    int num_packets = 0;
    uint64_t packet_address[SEND_BATCH_SIZE];
    int packet_length[SEND_BATCH_SIZE];

    while ( true )
    {
        uint64_t frame = client_alloc_frame( client );

        assert( frame != INVALID_FRAME );   // this should never happen

        uint8_t * packet = client->buffer + frame;

        packet_address[num_packets] = frame;
        packet_length[num_packets] = client_generate_packet( packet, PAYLOAD_BYTES );

        num_packets++;

        if ( num_packets == SEND_BATCH_SIZE )
            break;
    }

    for ( int i = 0; i < num_packets; i++ )
    {
        struct xdp_desc * desc = xsk_ring_prod__tx_desc( &client->send_queue, send_index + i );
        desc->addr = packet_address[i];
        desc->len = packet_length[i];
    }

    xsk_ring_prod__submit( &client->send_queue, num_packets );
    // In copy mode (MODE_SKB), Tx is driven by a syscall so we need to use e.g. sendto() to really send the packets.
    if ( client->attached_skb )
    {
        kick_tx(client);
    }

    // mark completed sent packet frames as free to be reused

    uint32_t complete_index;

    unsigned int completed = xsk_ring_cons__peek( &client->complete_queue, XSK_RING_CONS__DEFAULT_NUM_DESCS, &complete_index );

    if ( completed > 0 ) 
    {
        for ( int i = 0; i < completed; i++ )
        {
            client_free_frame( client, *xsk_ring_cons__comp_addr( &client->complete_queue, complete_index++ ) );
        }

        xsk_ring_cons__release( &client->complete_queue, completed );

        __sync_fetch_and_add( &client->current_sent_packets, completed );
    }
}

int main( int argc, char * argv[] )
{
    printf( "\n[client]\n" );

    signal( SIGINT,  interrupt_handler );
    signal( SIGTERM, clean_shutdown_handler );
    signal( SIGHUP,  clean_shutdown_handler );

    if ( client_init( &client, INTERFACE_NAME ) != 0 )
    {
        cleanup();
        return 1;
    }

    while ( !quit )
    {
        client_update( &client );
    }

    cleanup();

    printf( "\n" );

    return 0;
}
