#include "transport_config.h"
#include "util.h"
#include "util/shared_pool.h"
#include <arpa/inet.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <glog/logging.h>
#include <linux/if_ether.h>
#include <linux/if_link.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <net/if.h>
#include <xdp/libxdp.h>
#include <xdp/xsk.h>
#include <atomic>
#include <chrono>
#include <vector>
#include <assert.h>
#include <errno.h>
#include <ifaddrs.h>
#include <inttypes.h>
#include <memory.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>

using namespace uccl;

#define MY_NUM_QUEUES 1

char const* INTERFACE_NAME = DEV_DEFAULT;

int const MY_RECV_BATCH_SIZE = 32;
bool const busy_poll = true;

#define NUM_FRAMES (4096 * 64)
#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE
#define INVALID_FRAME UINT64_MAX

struct socket_t {
  void* umem_buffer;
  struct xsk_umem* umem;
  struct xsk_ring_cons recv_queue;
  struct xsk_ring_prod send_queue;
  struct xsk_ring_cons complete_queue;
  struct xsk_ring_prod fill_queue;
  struct xsk_socket* xsk;
  std::unique_ptr<SharedPool<uint64_t, true>> frame_pool;
  uint64_t recv_packets;
  uint32_t counter;
  int queue_id;
  std::vector<uint64_t> half_rtts;
};

struct server_t {
  int interface_index;
  struct xdp_program* program;
  bool attached_native;
  bool attached_skb;
  struct socket_t socket[MY_NUM_QUEUES];
  pthread_t stats_thread;
  pthread_t recv_thread[MY_NUM_QUEUES];
  uint64_t previous_recv_packets;
};

static struct server_t server;
bool volatile quit;

static void* stats_thread(void* arg);
static void* recv_thread(void* arg);

int server_init(struct server_t* server, char const* interface_name) {
  // we can only run xdp programs as root
  if (geteuid() != 0) {
    printf("\nerror: this program must be run as root\n\n");
    return 1;
  }

  // find the network interface that matches the interface name
  {
    bool found = false;

    struct ifaddrs* addrs;
    if (getifaddrs(&addrs) != 0) {
      printf("\nerror: getifaddrs failed\n\n");
      return 1;
    }

    for (struct ifaddrs* iap = addrs; iap != NULL; iap = iap->ifa_next) {
      if (iap->ifa_addr && (iap->ifa_flags & IFF_UP) &&
          iap->ifa_addr->sa_family == AF_INET) {
        struct sockaddr_in* sa = (struct sockaddr_in*)iap->ifa_addr;
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
      xdp_program__open_file("ebpf_server.o", "ebpf_server", NULL);
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

  // allow unlimited locking of memory, so all memory needed for packet
  // buffers can be locked
  struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};

  if (setrlimit(RLIMIT_MEMLOCK, &rlim)) {
    printf("\nerror: could not setrlimit\n\n");
    return 1;
  }

  // per-CPU socket setup
  for (int i = 0; i < MY_NUM_QUEUES; i++) {
    // allocate umem_buffer for umem
    const size_t buffer_size = NUM_FRAMES * FRAME_SIZE;

    if (posix_memalign(&server->socket[i].umem_buffer, getpagesize(),
                       buffer_size)) {
      printf("\nerror: could not allocate umem_buffer\n\n");
      return 1;
    }

    // allocate umem
    ret = xsk_umem__create(
        &server->socket[i].umem, server->socket[i].umem_buffer, buffer_size,
        &server->socket[i].fill_queue, &server->socket[i].complete_queue, NULL);
    if (ret) {
      printf("\nerror: could not create umem\n\n");
      return 1;
    }

    // create xsk socket and assign to network interface queue
    struct xsk_socket_config xsk_config;

    memset(&xsk_config, 0, sizeof(xsk_config));

    xsk_config.rx_size = XSK_RING_CONS__DEFAULT_NUM_DESCS;
    xsk_config.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;
    xsk_config.xdp_flags = XDP_ZEROCOPY;  // force zero copy mode
    xsk_config.bind_flags =
        XDP_USE_NEED_WAKEUP;  // manually wake up the driver when it needs
                              // to do work to send packets
    xsk_config.libbpf_flags = XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD;

    int queue_id = i;

    ret = xsk_socket__create(&server->socket[i].xsk, interface_name, queue_id,
                             server->socket[i].umem,
                             &server->socket[i].recv_queue,
                             &server->socket[i].send_queue, &xsk_config);
    if (ret) {
      printf("\nerror: could not create xsk socket [%d]\n\n", queue_id);
      return 1;
    }
    // apply_setsockopt(xsk_socket__fd(server->socket[i].xsk));

    server->socket[i].frame_pool = std::make_unique<SharedPool<uint64_t, true>>(
        NUM_FRAMES, [](uint64_t frame) {});
    // initialize frame allocator
    for (int j = 0; j < NUM_FRAMES; j++) {
      server->socket[i].frame_pool->push(j * FRAME_SIZE + XDP_PACKET_HEADROOM);
    }

    // set socket queue id for later use
    server->socket[i].queue_id = i;
  }

  // create stats thread
  ret = pthread_create(&server->stats_thread, NULL, stats_thread, server);
  if (ret) {
    printf("\nerror: could not create stats thread\n\n");
    return 1;
  }

  // create socket threads
  for (int i = 0; i < MY_NUM_QUEUES; i++) {
    ret = pthread_create(&server->recv_thread[i], NULL, recv_thread,
                         &server->socket[i]);
    if (ret) {
      printf("\nerror: could not create socket recv thread #%d\n\n", i);
      return 1;
    }
  }

  return 0;
}

void server_shutdown(struct server_t* server) {
  assert(server);

  for (int i = 0; i < MY_NUM_QUEUES; i++) {
    pthread_join(server->recv_thread[i], NULL);
  }

  for (int i = 0; i < MY_NUM_QUEUES; i++) {
    if (server->socket[i].xsk) {
      xsk_socket__delete(server->socket[i].xsk);
    }

    if (server->socket[i].umem) {
      xsk_umem__delete(server->socket[i].umem);
    }

    free(server->socket[i].umem_buffer);
  }

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

static void* stats_thread(void* arg) {
  struct server_t* server = (struct server_t*)arg;

  while (!quit) {
    usleep(1000000);
    std::vector<uint64_t> half_rtts;

    uint64_t recv_packets = 0;
    for (int i = 0; i < MY_NUM_QUEUES; i++) {
      recv_packets += server->socket[i].recv_packets;
      half_rtts.insert(half_rtts.end(), server->socket[i].half_rtts.begin(),
                       server->socket[i].half_rtts.end());
    }
    auto med_latency = Percentile(half_rtts, 50);
    auto tail_latency = Percentile(half_rtts, 99);
    uint64_t recv_delta = recv_packets - server->previous_recv_packets;
    printf("recv delta: %lu, med rtt: %lu us, tail rtt: %lu us\n", recv_delta,
           med_latency, tail_latency);

    server->previous_recv_packets = recv_packets;
  }

  return NULL;
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
  server_shutdown(&server);
  fflush(stdout);
}

// Return whether this packet will be forwarded back.
bool process_packet_and_send(struct socket_t* socket, uint64_t frame_offset,
                             uint32_t len) {
  int ret;
  uint32_t tx_idx = 0;
  uint8_t tmp_mac[ETH_ALEN];
  __be32 tmp_ip;
  uint16_t tmp_port;
  uint8_t* pkt = (uint8_t*)socket->umem_buffer + frame_offset;
  struct ethhdr* eth = (struct ethhdr*)pkt;
  struct iphdr* ip = (struct iphdr*)((char*)pkt + sizeof(struct ethhdr));
  struct udphdr* udp = (struct udphdr*)((char*)ip + sizeof(struct iphdr));

  uint8_t* payload = (uint8_t*)((char*)udp + sizeof(struct udphdr));
  uint64_t now_us = *(uint64_t*)payload;
  uint32_t counter = *(uint32_t*)(payload + sizeof(uint64_t));
  auto now = std::chrono::high_resolution_clock::now();
  uint64_t now_us2 = std::chrono::duration_cast<std::chrono::microseconds>(
                         now.time_since_epoch())
                         .count();
  if (now_us2 > now_us) {
    // Note that this measure is not accuracy, as it cross nodes.
    uint64_t half_rtt = now_us2 - now_us;
    socket->half_rtts.push_back(half_rtt);
  }

  memcpy(tmp_mac, eth->h_dest, ETH_ALEN);
  memcpy(eth->h_dest, eth->h_source, ETH_ALEN);
  memcpy(eth->h_source, tmp_mac, ETH_ALEN);

  memcpy(&tmp_ip, &ip->saddr, sizeof(tmp_ip));
  memcpy(&ip->saddr, &ip->daddr, sizeof(tmp_ip));
  memcpy(&ip->daddr, &tmp_ip, sizeof(tmp_ip));
  ip->check = 0;
  ip->check = ipv4_checksum(ip, sizeof(struct iphdr));

  // Swap source and destination port
  tmp_port = udp->source;
  udp->source = udp->dest;
  udp->dest = tmp_port;
  udp->check = 0;

  /* Here we sent the packet out of the receive port. Note that
   * we allocate one entry and schedule it. Your design would be
   * faster if you do batch processing/transmission */
  ret = xsk_ring_prod__reserve(&socket->send_queue, 1, &tx_idx);
  if (ret != 1) {
    // No more transmit slots, drop the packet
    return false;
  }

  struct xdp_desc* desc = xsk_ring_prod__tx_desc(&socket->send_queue, tx_idx);
  desc->addr = frame_offset;
  desc->len = len;
  xsk_ring_prod__submit(&socket->send_queue, 1);

  return true;
}

void complete_tx(struct socket_t* socket) {
  unsigned int completed;
  uint32_t idx_cq;

  sendto(xsk_socket__fd(socket->xsk), NULL, 0, MSG_DONTWAIT, NULL, 0);

  // Collect/free completed TX buffers
  completed = xsk_ring_cons__peek(&socket->complete_queue,
                                  XSK_RING_CONS__DEFAULT_NUM_DESCS, &idx_cq);

  VLOG(3) << "rx complete_tx completed = " << completed;
  if (completed > 0) {
    for (int i = 0; i < completed; i++)
      socket->frame_pool->push(
          *xsk_ring_cons__comp_addr(&socket->complete_queue, idx_cq++));

    xsk_ring_cons__release(&socket->complete_queue, completed);
  }
}

void socket_recv(struct socket_t* socket, int queue_id) {
  // Check any packet received, in order to drive packet receiving path for
  // other kernel transport.
  uint32_t idx_rx, idx_fq, rcvd;
  rcvd = xsk_ring_cons__peek(&socket->recv_queue, MY_RECV_BATCH_SIZE, &idx_rx);
  if (!rcvd) return;

  /* Stuff the ring with as much frames as possible */
  int stock_frames = xsk_prod_nb_free(&socket->fill_queue, MY_RECV_BATCH_SIZE);

  if (stock_frames > 0) {
    int ret =
        xsk_ring_prod__reserve(&socket->fill_queue, stock_frames, &idx_fq);

    /* This should not happen, but just in case */
    while (ret != stock_frames)
      ret = xsk_ring_prod__reserve(&socket->fill_queue, rcvd, &idx_fq);

    for (int i = 0; i < stock_frames; i++)
      *xsk_ring_prod__fill_addr(&socket->fill_queue, idx_fq++) =
          socket->frame_pool->pop();

    xsk_ring_prod__submit(&socket->fill_queue, stock_frames);
  }

  VLOG(3) << "rx fill_queue rcvd = " << rcvd
          << ", stock_frames = " << stock_frames;
  for (int i = 0; i < rcvd; i++) {
    const struct xdp_desc* desc =
        xsk_ring_cons__rx_desc(&socket->recv_queue, idx_rx++);
    uint64_t frame_offset = desc->addr;
    uint32_t len = desc->len;

    // TODO(Yang): doing batched sending
    auto need_sending = process_packet_and_send(socket, frame_offset, len);
    if (!need_sending) {
      socket->frame_pool->push(frame_offset);
    }
  }
  xsk_ring_cons__release(&socket->recv_queue, rcvd);
  __sync_fetch_and_add(&socket->recv_packets, rcvd);

  complete_tx(socket);
}

static void* recv_thread(void* arg) {
  struct socket_t* socket = (struct socket_t*)arg;

  // We also need to load and update the xsks_map for receiving packets
  struct bpf_map* map = bpf_object__find_map_by_name(
      xdp_program__bpf_obj(server.program), "xsks_map");
  int xsk_map_fd = bpf_map__fd(map);
  if (xsk_map_fd < 0) {
    fprintf(stderr, "ERROR: no xsks map found: %s\n", strerror(xsk_map_fd));
    exit(0);
  }
  int ret = xsk_socket__update_xskmap(socket->xsk, xsk_map_fd);
  if (ret) {
    fprintf(stderr, "ERROR: xsks map update fails: %s\n", strerror(xsk_map_fd));
    exit(0);
  }

  /* Stuff the receive path with buffers, we assume we have enough */
  uint32_t idx_rx = 0;
  ret = xsk_ring_prod__reserve(&socket->fill_queue,
                               XSK_RING_PROD__DEFAULT_NUM_DESCS, &idx_rx);

  if (ret != XSK_RING_PROD__DEFAULT_NUM_DESCS) exit(0);

  for (int i = 0; i < XSK_RING_PROD__DEFAULT_NUM_DESCS; i++)
    *xsk_ring_prod__fill_addr(&socket->fill_queue, idx_rx++) =
        socket->frame_pool->pop();

  xsk_ring_prod__submit(&socket->fill_queue, XSK_RING_PROD__DEFAULT_NUM_DESCS);

  int queue_id = socket->queue_id;
  printf("started socket recv thread for queue #%d\n", queue_id);

  pin_thread_to_cpu(MY_NUM_QUEUES + queue_id);

  struct pollfd fds[2];
  int nfds = 1;

  memset(fds, 0, sizeof(fds));
  fds[0].fd = xsk_socket__fd(socket->xsk);
  fds[0].events = POLLIN;

  while (!quit) {
    if (!busy_poll) {
      ret = poll(fds, nfds, 1000);
      if (ret <= 0 || ret > 1) continue;
    }
    socket_recv(socket, queue_id);
  }
  return NULL;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  printf("\n[server]\n");

  signal(SIGINT, interrupt_handler);
  signal(SIGTERM, clean_shutdown_handler);
  signal(SIGHUP, clean_shutdown_handler);

  int pshared;
  int ret;

  if (server_init(&server, INTERFACE_NAME) != 0) {
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
