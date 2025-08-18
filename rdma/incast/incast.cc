#include "transport.h"
#include "transport_config.h"
#include "util.h"
#include "util/timer.h"
#include <arpa/inet.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <netdb.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#define PT
#define INCAST

#define GPU_MEM

// Each node has 8 ranks/nics/gpus/threads
#define NB_THREADS 8
#define INCAST_RANK 15

#define MAX_NODES 512
#define BASE_LISTEN_PORT 6666

#define INCAST_BASE_LISTEN_PORT 16666

#define MPI_LOG(level) LOG(level) << "Node:" << NODE_ID << " "

#define MAX_BUFFER_SIZE (16 * 1024 * 1024)
#define PT_NET_CHUNK_SIZE (1024 * 1024)
#define INCAST_NET_CHUNK_SIZE (1024 * 1024)
#define MAX_CHUNK (MAX_BUFFER_SIZE / PT_NET_CHUNK_SIZE)

DEFINE_uint32(pt_size, 4 * 1024 * 1024, "Message size of Permutation Traffic.");
DEFINE_uint32(incast_size, 4 * 1024 * 1024, "Message size of Incast Traffic.");
DEFINE_uint32(iterations, 200000, "Number of iterations to run.");

using namespace uccl;

std::vector<std::vector<uint64_t>> pt_rtts;
std::vector<std::vector<uint64_t>> incast_rtts;
std::vector<std::vector<uint64_t>> copy_incast_rtts;

static int NODE_ID;
static int NRANKS;

struct Socket {
  ConnID conn_id;
  void* buffer = nullptr;
  struct Mhandle* mhandle = nullptr;
  int nb_net_chunk = 0;
  struct ucclRequest ureq[MAX_CHUNK] = {};
  bool done[MAX_CHUNK] = {};
  uint64_t rtt[MAX_CHUNK] = {};
};

struct CommHandle {
  struct Socket incast[NB_THREADS * 2];
  struct Socket pt;
};
std::optional<RDMAEndpoint> ep;

std::vector<std::string> ips;
class NodeInfo;
std::vector<NodeInfo> nodes;

std::thread stats_thread;
std::atomic<uint64_t> pt_tx_cur_sec_bytes[NB_THREADS * 2] = {};
uint64_t pt_tx_prev_sec_bytes[NB_THREADS * 2] = {};
std::atomic<uint64_t> pt_rx_cur_sec_bytes[NB_THREADS * 2] = {};
uint64_t pt_rx_prev_sec_bytes[NB_THREADS * 2] = {};

std::atomic<uint64_t> incast_tx_cur_sec_bytes[NB_THREADS * 2] = {};
uint64_t incast_tx_prev_sec_bytes[NB_THREADS * 2] = {};
std::atomic<uint64_t> incast_rx_cur_sec_bytes[NB_THREADS * 2] = {};
uint64_t incast_rx_prev_sec_bytes[NB_THREADS * 2] = {};

static bool volatile quit = false;

static uint32_t volatile pt_cur_iteration[NB_THREADS * 2] = {};
static uint32_t volatile incast_cur_iteration[NB_THREADS * 2] = {};

void dump_rtts() {
  // Dump pt_rtts to ./pt_rtts.txt with "RankX value" format
  std::ofstream pt_file("./pt_rtts.txt");
  if (!pt_file.is_open()) {
    std::cerr << "Error: Could not open pt_rtts.txt for writing." << std::endl;
    return;
  }
  for (size_t i = 0; i < pt_rtts.size(); ++i) {
    for (auto const& rtt : pt_rtts[i]) {
      pt_file << "Rank" << i << " " << to_usec(rtt, freq_ghz) << "\n";
    }
  }
  pt_file.close();

  // Dump incast_rtts to ./incast_rtts.txt with same format
  // Incast benchmark is still runnning, so we copy the vector.
  copy_incast_rtts = incast_rtts;
  std::ofstream incast_file("./incast_rtts.txt");
  if (!incast_file.is_open()) {
    std::cerr << "Error: Could not open incast_rtts.txt for writing."
              << std::endl;
    return;
  }
  for (size_t i = 0; i < copy_incast_rtts.size(); ++i) {
    for (auto const& rtt : copy_incast_rtts[i]) {
      incast_file << "Rank" << i << " " << to_usec(rtt, freq_ghz) << "\n";
    }
  }
  incast_file.close();

  MPI_LOG(INFO) << "RTT data dumped successfully (RankX value format).";
}

void interrupt_handler(int signal) {
  (void)signal;
  quit = true;
}

int find_target_rank(std::string const& filePath, int sourceNode);

static void server_setup_func(int local_rank, int target_rank, bool incast);

static void client_setup_func(int local_rank, int target_rank, bool incast);

class NodeInfo {
 public:
  NodeInfo(std::string& ip, int target_rank)
      : ip_(ip), target_rank_(target_rank), listen_fd_(0) {
    DCHECK(!ip_.empty());
  }
  ~NodeInfo() = default;

  NodeInfo(NodeInfo const&) = delete;
  NodeInfo& operator=(NodeInfo const&) = delete;

  NodeInfo(NodeInfo&&) = default;
  NodeInfo& operator=(NodeInfo&&) = default;

  // FD of listening socket.
  int listen_fd_;

  // FD of listening socket for incast.
  int incast_listen_fd_;

  // Target rank.
  int target_rank_;
  // IP address.
  std::string ip_;

  // Communication handle for sending.
  struct CommHandle send_comm_;
  // Communication handle for receiving.
  struct CommHandle recv_comm_;
};

static void server_setup_func(int local_rank, int target_rank, bool incast) {
  std::string ip = nodes[target_rank].ip_;
  struct CommHandle* recv_comm = &nodes[target_rank].recv_comm_;
  int remote_dev;
  int remote_gpuidx;
  auto local_dev = local_rank % NB_THREADS;

  if (incast)
    MPI_LOG(INFO) << "Incast, " << local_rank << " <-- " << target_rank;
  else {
    MPI_LOG(INFO) << "PT, " << local_rank << " <-- " << target_rank;
  }

  auto fd = incast ? nodes[target_rank].incast_listen_fd_
                   : nodes[target_rank].listen_fd_;

  auto conn_id = ep->uccl_accept(local_dev, fd, local_rank, ip, &remote_dev,
                                 &remote_gpuidx);

  if (incast)
    MPI_LOG(INFO) << "Done Incast, " << local_rank << " <-- " << target_rank;
  else {
    MPI_LOG(INFO) << "Done PT, " << local_rank << " <-- " << target_rank;
  }

  DCHECK(target_rank % NB_THREADS == remote_dev);

  if (incast)
    recv_comm->incast[local_rank].conn_id = conn_id;
  else
    recv_comm->pt.conn_id = conn_id;
}

static void client_setup_func(int local_rank, int target_rank, bool incast) {
  std::string ip = nodes[target_rank].ip_;
  struct CommHandle* send_comm = &nodes[target_rank].send_comm_;
  int base_listen_port =
      (incast ? INCAST_BASE_LISTEN_PORT : BASE_LISTEN_PORT) + target_rank * 32;

  auto local_dev = local_rank % NB_THREADS;
  auto target_dev = target_rank % NB_THREADS;

  if (incast)
    MPI_LOG(INFO) << "Incast, " << local_rank << " --> " << target_rank
                  << ", port: " << base_listen_port + local_rank;
  else {
    MPI_LOG(INFO) << "PT, " << local_rank << " --> " << target_rank
                  << ", port: " << base_listen_port + local_rank;
  }

  auto conn_id =
      ep->uccl_connect(local_dev, local_rank, target_dev, target_rank, ip,
                       base_listen_port + local_rank);

  if (incast)
    MPI_LOG(INFO) << "Done Incast, " << local_rank << " --> " << target_rank;
  else {
    MPI_LOG(INFO) << "Done PT, " << local_rank << " --> " << target_rank;
  }

  if (incast)
    send_comm->incast[local_rank].conn_id = conn_id;
  else
    send_comm->pt.conn_id = conn_id;
}

std::vector<std::thread> ts;

void setup_connections(int local_rank, int target_rank) {
  std::thread t;

#ifdef PT
  // Setup connections for target rank.
  t = std::thread(client_setup_func, local_rank, target_rank, false);
  ts.push_back(std::move(t));
  t = std::thread(server_setup_func, local_rank, target_rank, false);
  ts.push_back(std::move(t));
#endif

#ifdef INCAST
  {
    if (local_rank == INCAST_RANK) {
      // This is incast rank, accept connections from all nodes except for
      // itself.
      for (int r = 0; r < NRANKS; r++) {
        if (r == INCAST_RANK) continue;
        t = std::thread(server_setup_func, INCAST_RANK, r, true);
        ts.push_back(std::move(t));
      }
      return;
    }

    // Connect to incast rank.
    t = std::thread(client_setup_func, local_rank, INCAST_RANK, true);
    ts.push_back(std::move(t));
  }
#endif
}

void pt_register_recv_buffer(int local_rank, int r) {
#ifdef GPU_MEM
  auto local_dev = local_rank % NB_THREADS;
  cudaSetDevice(local_dev);
  cudaMalloc(&nodes[r].recv_comm_.pt.buffer, MAX_BUFFER_SIZE);
#else
  nodes[r].recv_comm_.pt.buffer =
      mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  DCHECK(nodes[r].recv_comm_.pt.buffer != MAP_FAILED);
#endif

  ep->uccl_regmr((UcclFlow*)nodes[r].recv_comm_.pt.conn_id.context,
                 nodes[r].recv_comm_.pt.buffer, MAX_BUFFER_SIZE, 0,
                 &nodes[r].recv_comm_.pt.mhandle);
}

void pt_register_send_buffer(int local_rank, int r) {
#ifdef GPU_MEM
  auto local_dev = local_rank % NB_THREADS;
  cudaSetDevice(local_dev);
  cudaMalloc(&nodes[r].send_comm_.pt.buffer, MAX_BUFFER_SIZE);
#else
  nodes[r].send_comm_.pt.buffer =
      mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  DCHECK(nodes[r].send_comm_.pt.buffer != MAP_FAILED);
#endif

  ep->uccl_regmr((UcclFlow*)nodes[r].send_comm_.pt.conn_id.context,
                 nodes[r].send_comm_.pt.buffer, MAX_BUFFER_SIZE, 0,
                 &nodes[r].send_comm_.pt.mhandle);
}

void incast_register_recv_buffer(int local_rank, int r) {
#ifdef GPU_MEM
  auto local_dev = local_rank % NB_THREADS;
  cudaSetDevice(local_dev);
  cudaMalloc(&nodes[r].recv_comm_.incast[local_rank].buffer, MAX_BUFFER_SIZE);
#else
  nodes[r].recv_comm_.incast[local_rank].buffer =
      mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  DCHECK(nodes[r].recv_comm_.incast[local_rank].buffer != MAP_FAILED);
#endif

  ep->uccl_regmr(
      (UcclFlow*)nodes[r].recv_comm_.incast[local_rank].conn_id.context,
      nodes[r].recv_comm_.incast[local_rank].buffer, MAX_BUFFER_SIZE, 0,
      &nodes[r].recv_comm_.incast[local_rank].mhandle);
}

void incast_register_send_buffer(int local_rank, int r) {
#ifdef GPU_MEM
  auto local_dev = local_rank % NB_THREADS;
  cudaSetDevice(local_dev);
  cudaMalloc(&nodes[r].send_comm_.incast[local_rank].buffer, MAX_BUFFER_SIZE);
#else
  nodes[r].send_comm_.incast[local_rank].buffer =
      mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  DCHECK(nodes[r].send_comm_.incast[local_rank].buffer != MAP_FAILED);
#endif

  ep->uccl_regmr(
      (UcclFlow*)nodes[r].send_comm_.incast[local_rank].conn_id.context,
      nodes[r].send_comm_.incast[local_rank].buffer, MAX_BUFFER_SIZE, 0,
      &nodes[r].send_comm_.incast[local_rank].mhandle);
}

void incast_unregister_buffer(int local_rank, int r) {
  if (local_rank == INCAST_RANK) {
    ep->uccl_deregmr(nodes[r].recv_comm_.incast[local_rank].mhandle);
#ifdef GPU_MEM
    cudaFree(nodes[r].recv_comm_.incast[local_rank].buffer);
#else
    munmap(nodes[r].recv_comm_.incast[local_rank].buffer, MAX_BUFFER_SIZE);
#endif
  } else {
    ep->uccl_deregmr(nodes[r].send_comm_.incast[local_rank].mhandle);
#ifdef GPU_MEM
    cudaFree(nodes[r].send_comm_.incast[local_rank].buffer);
#else
    munmap(nodes[r].send_comm_.incast[local_rank].buffer, MAX_BUFFER_SIZE);
#endif
  }
}

void pt_unregister_buffer(int local_rank, int r) {
  ep->uccl_deregmr(nodes[r].send_comm_.pt.mhandle);
#ifdef GPU_MEM
  cudaFree(nodes[r].send_comm_.pt.buffer);
#else
  munmap(nodes[r].send_comm_.pt.buffer, MAX_BUFFER_SIZE);
#endif

  ep->uccl_deregmr(nodes[r].recv_comm_.pt.mhandle);
#ifdef GPU_MEM
  cudaFree(nodes[r].recv_comm_.pt.buffer);
#else
  munmap(nodes[r].recv_comm_.pt.buffer, MAX_BUFFER_SIZE);
#endif
}

void allocate_buffers_per_rank(int local_rank, int target_rank) {
#ifdef PT
  pt_register_send_buffer(local_rank, target_rank);
  pt_register_recv_buffer(local_rank, target_rank);
#endif

#ifdef INCAST
  {
    if (local_rank == INCAST_RANK) {
      // This is incast rank.
      for (int r = 0; r < NRANKS; r++) {
        if (r == local_rank) continue;
        incast_register_recv_buffer(local_rank, r);
      }

      return;
    }

    incast_register_send_buffer(local_rank, INCAST_RANK);
  }
#endif
}

void free_buffers(int local_rank, int target_rank) {
#ifdef PT
  pt_unregister_buffer(local_rank, target_rank);
#endif

#ifdef INCAST
  {
    if (local_rank == INCAST_RANK) {
      // This is incast rank.
      for (int r = 0; r < NRANKS; r++) {
        if (r == local_rank) continue;
        incast_unregister_buffer(local_rank, r);
      }

      return;
    }

    incast_unregister_buffer(local_rank, INCAST_RANK);
  }
#endif
}

void pt_listen_ports(int local_rank) {
  uint16_t base_listen_port = BASE_LISTEN_PORT + local_rank * 32;
  auto r = find_target_rank("matrix.txt", local_rank);

  nodes[r].listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  DCHECK(nodes[r].listen_fd_ >= 0);
  int flag = 1;
  DCHECK(setsockopt(nodes[r].listen_fd_, SOL_SOCKET, SO_REUSEADDR, &flag,
                    sizeof(int)) >= 0);
  struct sockaddr_in serv_addr;
  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(base_listen_port + r);
  int ret = bind(nodes[r].listen_fd_, (struct sockaddr*)&serv_addr,
                 sizeof(serv_addr));
  DCHECK(ret >= 0) << ret;
  ret = listen(nodes[r].listen_fd_, MAX_NODES);
  DCHECK(ret == 0) << ret;

  MPI_LOG(INFO) << local_rank << " listen on port " << base_listen_port + r
                << " for PT.";
}

void incast_listen_ports() {
  uint16_t base_listen_port = INCAST_BASE_LISTEN_PORT + INCAST_RANK * 32;

  for (int r = 0; r < NRANKS; r++) {
    if (r == INCAST_RANK) continue;
    nodes[r].incast_listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(nodes[r].incast_listen_fd_ >= 0);
    int flag = 1;
    DCHECK(setsockopt(nodes[r].incast_listen_fd_, SOL_SOCKET, SO_REUSEADDR,
                      &flag, sizeof(int)) >= 0);
    struct sockaddr_in serv_addr;
    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(base_listen_port + r);
    int ret = bind(nodes[r].incast_listen_fd_, (struct sockaddr*)&serv_addr,
                   sizeof(serv_addr));
    DCHECK(ret >= 0) << ret;
    ret = listen(nodes[r].incast_listen_fd_, MAX_NODES);
    DCHECK(ret == 0) << ret;

    MPI_LOG(INFO) << INCAST_RANK << " listen on port " << base_listen_port + r
                  << " for incast.";
  }
}

static void init_benchmark_per_rank(int local_rank) {
  int target_rank = find_target_rank("matrix.txt", local_rank);
  setup_connections(local_rank, target_rank);
}

static void exit_benchmark_per_rank(int local_rank) {
  int target_rank = find_target_rank("matrix.txt", local_rank);
  free_buffers(local_rank, target_rank);

  stats_thread.join();
}

bool test_p2p_send_done(int local_rank, int target_rank, int chunk_id,
                        bool incast) {
  bool done;
  auto* send_comm_ = &nodes[target_rank].send_comm_;
  if (incast) {
    if (send_comm_->incast[local_rank].done[chunk_id]) return true;

    done =
        ep->uccl_poll_ureq_once(&send_comm_->incast[local_rank].ureq[chunk_id]);
    if (done) {
      incast_tx_cur_sec_bytes[local_rank].fetch_add(
          send_comm_->incast[local_rank].ureq[chunk_id].send.data_len);
      send_comm_->incast[local_rank].done[chunk_id] = true;
      if (local_rank != 7 && local_rank != 15) {
        // Filter data from Rank 7 and 15
        incast_rtts[local_rank].push_back(
            rdtsc() - send_comm_->incast[local_rank].rtt[chunk_id]);
      }
    }
  } else {
    if (send_comm_->pt.done[chunk_id]) return true;

    done = ep->uccl_poll_ureq_once(&send_comm_->pt.ureq[chunk_id]);
    if (done) {
      pt_tx_cur_sec_bytes[local_rank].fetch_add(
          send_comm_->pt.ureq[chunk_id].send.data_len);
      send_comm_->pt.done[chunk_id] = true;
      if (local_rank != 3 && local_rank != 6 && local_rank != 7 &&
          local_rank != 15) {
        // Filter data from Rank 3,6,7,15
        pt_rtts[local_rank].push_back(rdtsc() - send_comm_->pt.rtt[chunk_id]);
      }
    }
  }

  return done;
}

bool test_p2p_recv_done(int local_rank, int target_rank, int chunk_id,
                        bool incast) {
  bool done;
  auto* recv_comm_ = &nodes[target_rank].recv_comm_;

  if (incast) {
    if (recv_comm_->incast[local_rank].done[chunk_id]) return true;

    done =
        ep->uccl_poll_ureq_once(&recv_comm_->incast[local_rank].ureq[chunk_id]);
    if (done) {
      incast_rx_cur_sec_bytes[local_rank].fetch_add(
          recv_comm_->incast[local_rank].ureq[chunk_id].recv.data_len[0]);
      recv_comm_->incast[local_rank].done[chunk_id] = true;
    }
  } else {
    if (recv_comm_->pt.done[chunk_id]) return true;

    done = ep->uccl_poll_ureq_once(&recv_comm_->pt.ureq[chunk_id]);
    if (done) {
      pt_rx_cur_sec_bytes[local_rank].fetch_add(
          recv_comm_->pt.ureq[chunk_id].recv.data_len[0]);
      recv_comm_->pt.done[chunk_id] = true;
    }
  }

  return done;
}

void p2p_send(int local_rank, int target_rank, int size, bool incast) {
  auto* send_comm_ = &nodes[target_rank].send_comm_;

  uint32_t offset = 0;
  uint32_t chunk_id = 0;

  if (incast) {
    while (offset < size) {
      int net_chunk_size =
          std::min(size - offset, (uint32_t)INCAST_NET_CHUNK_SIZE);

      send_comm_->incast[local_rank].rtt[chunk_id] = rdtsc();
      while (ep->uccl_send_async(
          (UcclFlow*)send_comm_->incast[local_rank].conn_id.context,
          send_comm_->incast[local_rank].mhandle,
          send_comm_->incast[local_rank].buffer, net_chunk_size,
          &send_comm_->incast[local_rank].ureq[chunk_id])) {
        send_comm_->incast[local_rank].rtt[chunk_id] = rdtsc();
      }

      send_comm_->incast[local_rank].done[chunk_id] = false;
      offset += net_chunk_size;
      chunk_id++;
    }

    send_comm_->incast[local_rank].nb_net_chunk = chunk_id;
  } else {
    while (offset < size) {
      int net_chunk_size = std::min(size - offset, (uint32_t)PT_NET_CHUNK_SIZE);
      send_comm_->pt.rtt[chunk_id] = rdtsc();
      while (ep->uccl_send_async((UcclFlow*)send_comm_->pt.conn_id.context,
                                 send_comm_->pt.mhandle, send_comm_->pt.buffer,
                                 net_chunk_size,
                                 &send_comm_->pt.ureq[chunk_id])) {
        send_comm_->pt.rtt[chunk_id] = rdtsc();
      }

      send_comm_->pt.done[chunk_id] = false;
      offset += net_chunk_size;
      chunk_id++;
    }

    send_comm_->pt.nb_net_chunk = chunk_id;
  }
  // MPI_LOG(INFO) << "p2p send to " << target_rank << ", " << chunk_id;
}

void p2p_receive(int local_rank, int target_rank, int size, bool incast) {
  auto* recv_comm_ = &nodes[target_rank].recv_comm_;

  uint32_t offset = 0;
  uint32_t chunk_id = 0;

  if (incast) {
    while (offset < size) {
      int net_chunk_size =
          std::min(size - offset, (uint32_t)INCAST_NET_CHUNK_SIZE);
      DCHECK(ep->uccl_recv_async(
                 (UcclFlow*)recv_comm_->incast[local_rank].conn_id.context,
                 &recv_comm_->incast[local_rank].mhandle,
                 &recv_comm_->incast[local_rank].buffer, &net_chunk_size, 1,
                 &recv_comm_->incast[local_rank].ureq[chunk_id]) == 0);

      recv_comm_->incast[local_rank].done[chunk_id] = false;
      offset += net_chunk_size;
      chunk_id++;
    }

    recv_comm_->incast[local_rank].nb_net_chunk = chunk_id;
  } else {
    while (offset < size) {
      int net_chunk_size = std::min(size - offset, (uint32_t)PT_NET_CHUNK_SIZE);
      DCHECK(ep->uccl_recv_async((UcclFlow*)recv_comm_->pt.conn_id.context,
                                 &recv_comm_->pt.mhandle,
                                 &recv_comm_->pt.buffer, &net_chunk_size, 1,
                                 &recv_comm_->pt.ureq[chunk_id]) == 0);

      recv_comm_->pt.done[chunk_id] = false;
      offset += net_chunk_size;
      chunk_id++;
    }

    recv_comm_->pt.nb_net_chunk = chunk_id;
  }
  // MPI_LOG(INFO) << "p2p receive from " << target_rank << ", " << chunk_id;
}

void test_all_send(int local_rank, int target_rank, bool incast) {
  uint32_t nb_net_chunk;
  if (incast) {
    nb_net_chunk =
        nodes[target_rank].send_comm_.incast[local_rank].nb_net_chunk;
  } else {
    nb_net_chunk = nodes[target_rank].send_comm_.pt.nb_net_chunk;
  }
  for (int i = 0; i < nb_net_chunk; i++) {
    while (!test_p2p_send_done(local_rank, target_rank, i, incast)) {
    }
  }
}

void test_all_recv(int local_rank, int target_rank, bool incast) {
  uint32_t nb_net_chunk;
  if (incast) {
    nb_net_chunk =
        nodes[target_rank].recv_comm_.incast[local_rank].nb_net_chunk;
  } else {
    nb_net_chunk = nodes[target_rank].recv_comm_.pt.nb_net_chunk;
  }
  for (int i = 0; i < nb_net_chunk; i++) {
    while (!test_p2p_recv_done(local_rank, target_rank, i, incast)) {
    }
  }
}

// Non-blocking version.
bool nb_test_all_send(int local_rank, int target_rank, bool incast) {
  uint32_t nb_net_chunk;
  if (incast) {
    nb_net_chunk =
        nodes[target_rank].send_comm_.incast[local_rank].nb_net_chunk;
  } else {
    nb_net_chunk = nodes[target_rank].send_comm_.pt.nb_net_chunk;
  }
  uint32_t ready = 0;
  for (int i = 0; i < nb_net_chunk; i++) {
    if (test_p2p_send_done(local_rank, target_rank, i, incast)) ready++;
  }
  return ready == nb_net_chunk;
}

// Non-blocking version.
bool nb_test_all_recv(int local_rank, int target_rank, bool incast) {
  uint32_t nb_net_chunk;
  if (incast) {
    nb_net_chunk =
        nodes[target_rank].recv_comm_.incast[local_rank].nb_net_chunk;
  } else {
    nb_net_chunk = nodes[target_rank].recv_comm_.pt.nb_net_chunk;
  }
  uint32_t ready = 0;
  for (int i = 0; i < nb_net_chunk; i++) {
    if (test_p2p_recv_done(local_rank, target_rank, i, incast)) ready++;
  }
  return ready == nb_net_chunk;
}

void net_sync(int local_rank, int target_rank, bool incast) {
  test_all_send(local_rank, target_rank, incast);
  test_all_recv(local_rank, target_rank, incast);
}

void net_send_sync(int local_rank, int target_rank, bool incast) {
  test_all_send(local_rank, target_rank, incast);
}

void net_recv_sync(int local_rank, int target_rank, bool incast) {
  test_all_recv(local_rank, target_rank, incast);
}

void launch_stats_thread() {
  stats_thread = std::thread([&] {
    while (!quit) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
#ifdef PT
      for (int i = 0; i < NB_THREADS; i++) {
        int local_rank = i + NODE_ID * NB_THREADS;
        printf(
            "Rank %d (Permutation): TX Tput: %.4f Gbps, RX Tput: %.4f Gbps, "
            "iterations: %d\n",
            local_rank,
            (pt_tx_cur_sec_bytes[local_rank].load() -
             pt_tx_prev_sec_bytes[local_rank]) *
                8.0 / 1e9,
            (pt_rx_cur_sec_bytes[local_rank].load() -
             pt_rx_prev_sec_bytes[local_rank]) *
                8.0 / 1e9,
            pt_cur_iteration[local_rank]);

        pt_tx_prev_sec_bytes[local_rank] =
            pt_tx_cur_sec_bytes[local_rank].load();
        pt_rx_prev_sec_bytes[local_rank] =
            pt_rx_cur_sec_bytes[local_rank].load();
      }
#endif

#ifdef INCAST
      for (int i = 0; i < NB_THREADS; i++) {
        int local_rank = i + NODE_ID * NB_THREADS;
        printf(
            "Rank %d (Incast): TX Tput: %.4f Gbps, RX Tput: %.4f Gbps, "
            "iterations: %d\n",
            local_rank,
            (incast_tx_cur_sec_bytes[local_rank].load() -
             incast_tx_prev_sec_bytes[local_rank]) *
                8.0 / 1e9,
            (incast_rx_cur_sec_bytes[local_rank].load() -
             incast_rx_prev_sec_bytes[local_rank]) *
                8.0 / 1e9,
            incast_cur_iteration[local_rank]);

        incast_tx_prev_sec_bytes[local_rank] =
            incast_tx_cur_sec_bytes[local_rank].load();
        incast_rx_prev_sec_bytes[local_rank] =
            incast_rx_cur_sec_bytes[local_rank].load();
      }
#endif
    }
    return 0;
  });
}

void verify_params() { CHECK(FLAGS_pt_size <= MAX_BUFFER_SIZE); }

void incast_send(int local_rank) {
  while (incast_cur_iteration[local_rank]++ < 50 * FLAGS_iterations && !quit) {
    p2p_send(local_rank, INCAST_RANK, FLAGS_incast_size, true);
    // MPI_LOG(INFO) << local_rank << " send to incast rank.";
    net_send_sync(local_rank, INCAST_RANK, true);
    // MPI_LOG(INFO) << local_rank << " send to incast rank done.";
  }
}

void incast_recv(int local_rank) {
  bool first_run = true;

  while (!quit) {
    if (first_run) {
      first_run = false;
      // First run, post buffers to all nodes.
      for (int r = 0; r < NRANKS; r++) {
        if (r == local_rank) continue;
        p2p_receive(local_rank, r, FLAGS_incast_size, true);
        incast_cur_iteration[local_rank]++;
      }
    } else {
      // Not first run, post buffers when last recv is done.
      for (int r = 0; r < NRANKS; r++) {
        if (r == local_rank) continue;
        if (nb_test_all_recv(local_rank, r, true)) {
          p2p_receive(local_rank, r, FLAGS_incast_size, true);
          incast_cur_iteration[local_rank]++;
        }
      }
    }

    if (incast_cur_iteration[local_rank] >= 50 * FLAGS_iterations) break;
  }
}

void incast() {
  std::vector<std::thread> ts;
  // Span NB_THREADS threads.
  for (int i = 0; i < NB_THREADS; i++) {
    int local_rank = i + NODE_ID * NB_THREADS;
    if (local_rank == INCAST_RANK) {
      ts.push_back(std::move(std::thread(incast_recv, local_rank)));
    } else {
      ts.push_back(std::move(std::thread(incast_send, local_rank)));
    }
  }
  for (auto& t : ts) {
    t.join();
  }
}

int find_target_rank(std::string const& filePath, int sourceNode) {
  std::ifstream file(filePath);
  std::string line;

  while (std::getline(file, line)) {
    size_t arrowPos = line.find("->");
    if (arrowPos == std::string::npos) continue;

    int from = std::stoi(line.substr(0, arrowPos));
    if (from != sourceNode) continue;

    return std::stoi(line.substr(arrowPos + 2));
  }

  return -1;
}

void permutation_traffic_rank_thread(int local_rank) {
  int target_rank = find_target_rank("matrix.txt", local_rank);
  MPI_LOG(INFO) << local_rank << "'s target is " << target_rank;

  while (pt_cur_iteration[local_rank]++ < FLAGS_iterations && !quit) {
    p2p_receive(local_rank, target_rank, FLAGS_pt_size, false);
    p2p_send(local_rank, target_rank, FLAGS_pt_size, false);
    net_sync(local_rank, target_rank, false);
  }

  if (local_rank % NB_THREADS == 0) {
    dump_rtts();
  }
}

void permutation_traffic() {
  std::vector<std::thread> ts;
  // Span NB_THREADS threads.
  for (int i = 0; i < NB_THREADS; i++) {
    int local_rank = i + NODE_ID * NB_THREADS;
    ts.push_back(
        std::move(std::thread(permutation_traffic_rank_thread, local_rank)));
  }

  for (auto& t : ts) {
    t.join();
  }
}

void get_ips() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string local_ip;
  struct ifaddrs *ifaddr, *ifa;
  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET)
      continue;

    if (strcmp(ifa->ifa_name, SINGLE_CTRL_NIC.c_str()) == 0) {
      void* tmpAddrPtr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
      char addressBuffer[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
      local_ip = addressBuffer;
      break;
    }
  }
  freeifaddrs(ifaddr);

  if (rank == 0) {
    ips.resize(size);
    ips[0] = local_ip;

    for (int i = 1; i < size; i++) {
      char ip_buffer[INET_ADDRSTRLEN];
      MPI_Recv(ip_buffer, INET_ADDRSTRLEN, MPI_CHAR, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      ips[i] = ip_buffer;
    }

    for (int i = 0; i < size; i++) {
      MPI_Bcast(const_cast<char*>(ips[i].c_str()), ips[i].size() + 1, MPI_CHAR,
                0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(local_ip.c_str(), local_ip.size() + 1, MPI_CHAR, 0, 0,
             MPI_COMM_WORLD);

    ips.resize(size);
    for (int i = 0; i < size; i++) {
      char ip_buffer[INET_ADDRSTRLEN];
      MPI_Bcast(ip_buffer, INET_ADDRSTRLEN, MPI_CHAR, 0, MPI_COMM_WORLD);
      ips[i] = ip_buffer;
    }
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  signal(SIGINT, interrupt_handler);

  verify_params();

  MPI_Init(&argc, &argv);

  // Get ips from hostname.txt
  get_ips();
  DCHECK(ips.size() == 2);

  MPI_Comm_rank(MPI_COMM_WORLD, &NODE_ID);

  MPI_Comm_size(MPI_COMM_WORLD, &NRANKS);

  NRANKS *= NB_THREADS;
  DCHECK(NRANKS == 16);

  pt_rtts.resize(NRANKS);
  incast_rtts.resize(NRANKS);

  for (int i = 0; i < NRANKS; i++) {
    nodes.push_back(NodeInfo(ips[i / NB_THREADS], i));
  }

  ep.emplace(DEVNAME_SUFFIX_LIST, NUM_DEVICES, NUM_ENGINES);
  DCHECK(NUM_DEVICES == 8);

  // Initialize connections, allocate buffers, etc.
  for (int i = 0; i < NB_THREADS; i++) {
    int local_rank = i + NODE_ID * NB_THREADS;
#ifdef PT
    pt_listen_ports(local_rank);
#endif
  }

#ifdef INCAST
  incast_listen_ports();
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < NB_THREADS; i++) {
    int local_rank = i + NODE_ID * NB_THREADS;
    init_benchmark_per_rank(local_rank);
  }

  for (auto& t : ts) t.join();

  for (int i = 0; i < NB_THREADS; i++) {
    int local_rank = i + NODE_ID * NB_THREADS;
    int target_rank = find_target_rank("matrix.txt", local_rank);
    allocate_buffers_per_rank(local_rank, target_rank);
  }

  // Wait until all nodes are ready.
  MPI_Barrier(MPI_COMM_WORLD);

  // Launch stats thread.
  launch_stats_thread();

  std::thread incast_thread;
  std::thread pt_thread;

// Run incast benchmark first and then run permutation traffic benchmark.
#ifdef INCAST
  incast_thread = std::thread(incast);
#endif

#ifdef PT
  pt_thread = std::thread(permutation_traffic);
#endif

#ifdef PT
  pt_thread.join();
#endif

#ifdef INCAST
  incast_thread.join();
#endif

  // Destroy connections, free buffers, etc.
  for (int i = 0; i < NB_THREADS; i++) {
    int local_rank = i + NODE_ID * NB_THREADS;
    exit_benchmark_per_rank(local_rank);
  }

  MPI_Finalize();

  return 0;
}