#include "transport.h"
#include "transport_config.h"
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
#include <mpi.h>
#include <netdb.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_NODES 512
#define BASE_LISTEN_PORT 6666

#define MPI_LOG(level) LOG(level) << "Rank:" << LOCAL_RANK << " "

#define MAX_BUFFER_SIZE (32 * 1024 * 1024)  // 32MB
#define NET_CHUNK_SIZE (1024 * 1024)        // 1MB
#define MAX_CHUNK (MAX_BUFFER_SIZE / NET_CHUNK_SIZE)

DEFINE_uint32(size, 32 * 1024 * 1024, "Message size.");
DEFINE_uint32(iterations, 100000, "Number of iterations to run.");
DEFINE_string(benchtype, "PT",
              "Benchmark type. PT: Permutation Traffic, SA: Sequential "
              "AlltoAll, AA: AlltoAll");

using namespace uccl;

static constexpr uint32_t DEV = 0;
static constexpr uint32_t REMOTE_DEV = 0;

static int LOCAL_RANK;
static int NRANKS;

struct CommHandle {
  ConnID conn_id;
  void* buffer = nullptr;
  struct Mhandle* mhandle = nullptr;
  int nb_net_chunk = 0;
  struct ucclRequest ureq[MAX_CHUNK] = {};
};
std::optional<RDMAEndpoint> ep;

std::vector<std::string> ips;
class NodeInfo;
std::vector<NodeInfo> nodes;

std::thread stats_thread;
std::atomic<uint64_t> tx_cur_sec_bytes = 0;
uint64_t tx_prev_sec_bytes = 0;
std::atomic<uint64_t> rx_cur_sec_bytes = 0;
uint64_t rx_prev_sec_bytes = 0;

static bool volatile quit = false;

static uint32_t volatile cur_iteration = 0;

void interrupt_handler(int signal) {
  (void)signal;
  quit = true;
}

static void server_setup_agent(int target_rank, std::string ip,
                               struct CommHandle* recv_comm);

static void client_setup_agent(int target_rank, std::string ip,
                               struct CommHandle* send_comm);

class NodeInfo {
 public:
  NodeInfo(std::string& ip, int target_rank)
      : ip_(ip), target_rank_(target_rank), listen_fd_(0) {}
  ~NodeInfo() = default;

  void server_setup() {
    server_thread_ =
        std::thread(server_setup_agent, target_rank_, ip_, &recv_comm_);
  }

  void client_setup() {
    client_thread_ =
        std::thread(client_setup_agent, target_rank_, ip_, &send_comm_);
  }

  void wait_connect_done() { client_thread_.join(); }

  void wait_accept_done() { server_thread_.join(); }

  NodeInfo(NodeInfo const&) = delete;
  NodeInfo& operator=(NodeInfo const&) = delete;

  NodeInfo(NodeInfo&&) = default;
  NodeInfo& operator=(NodeInfo&&) = default;

  // FD of listening socket.
  int listen_fd_;
  // Target rank.
  int target_rank_;
  // IP address.
  std::string ip_;
  // Communication handle for sending.
  struct CommHandle send_comm_;
  // Communication handle for receiving.
  struct CommHandle recv_comm_;
  // Thread for client stuff.
  std::thread client_thread_;
  // Thread for server stuff.
  std::thread server_thread_;
};

static void server_setup_agent(int target_rank, std::string ip,
                               struct CommHandle* recv_comm) {
  int remote_dev;
  int remote_gpuidx;
  auto conn_id = ep->uccl_accept(DEV, nodes[target_rank].listen_fd_, DEV, ip,
                                 &remote_dev, &remote_gpuidx);

  recv_comm->conn_id = conn_id;

  MPI_LOG(INFO) << "Accepted from " << target_rank << " succesfully";
}

static void client_setup_agent(int target_rank, std::string ip,
                               struct CommHandle* send_comm) {
  auto conn_id = ep->uccl_connect(DEV, DEV, REMOTE_DEV, REMOTE_DEV, ip,
                                  BASE_LISTEN_PORT + LOCAL_RANK);

  send_comm->conn_id = conn_id;

  MPI_LOG(INFO) << "Connected to " << target_rank << " succesfully";
}

void setup_connections() {
  for (int r = 0; r < NRANKS; r++) {
    if (r == LOCAL_RANK) continue;
    nodes[r].client_setup();
    nodes[r].server_setup();
  }
  MPI_LOG(INFO) << "Start to setup connections.";
}

void wait_setup_connections() {
  for (int r = 0; r < NRANKS; r++) {
    if (r == LOCAL_RANK) continue;
    nodes[r].wait_connect_done();
    nodes[r].wait_accept_done();
  }
  MPI_LOG(INFO) << "Connections setup done.";
}

void allocate_buffers() {
  for (int r = 0; r < NRANKS; r++) {
    if (r == LOCAL_RANK) continue;

    nodes[r].send_comm_.buffer =
        mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    DCHECK(nodes[r].send_comm_.buffer != MAP_FAILED);

    ep->uccl_regmr((UcclFlow*)nodes[r].send_comm_.conn_id.context,
                   nodes[r].send_comm_.buffer, MAX_BUFFER_SIZE, 0,
                   &nodes[r].send_comm_.mhandle);

    nodes[r].recv_comm_.buffer =
        mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    DCHECK(nodes[r].recv_comm_.buffer != MAP_FAILED);

    ep->uccl_regmr((UcclFlow*)nodes[r].recv_comm_.conn_id.context,
                   nodes[r].recv_comm_.buffer, MAX_BUFFER_SIZE, 0,
                   &nodes[r].recv_comm_.mhandle);
  }
}

void free_buffers() {
  for (int r = 0; r < NRANKS; r++) {
    if (r == LOCAL_RANK) continue;

    ep->uccl_deregmr(nodes[r].send_comm_.mhandle);
    munmap(nodes[r].send_comm_.buffer, MAX_BUFFER_SIZE);

    ep->uccl_deregmr(nodes[r].recv_comm_.mhandle);
    munmap(nodes[r].recv_comm_.buffer, MAX_BUFFER_SIZE);
  }
}

void listen_ports() {
  for (int r = 0; r < NRANKS; r++) {
    if (r == LOCAL_RANK) continue;
    nodes[r].listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(nodes[r].listen_fd_ >= 0);
    int flag = 1;
    DCHECK(setsockopt(nodes[r].listen_fd_, SOL_SOCKET, SO_REUSEADDR, &flag,
                      sizeof(int)) >= 0);
    struct sockaddr_in serv_addr;
    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(BASE_LISTEN_PORT + r);
    int ret = bind(nodes[r].listen_fd_, (struct sockaddr*)&serv_addr,
                   sizeof(serv_addr));
    DCHECK(ret >= 0) << ret;
    ret = listen(nodes[r].listen_fd_, MAX_NODES);
    DCHECK(ret == 0) << ret;
  }

  MPI_LOG(INFO) << "Listen ports done.";
}

static void init_benchmark() {
  ep.emplace(DEVNAME_SUFFIX_LIST, NUM_DEVICES, NUM_ENGINES);

  listen_ports();

  if (FLAGS_benchtype != "PT") {
    setup_connections();

    wait_setup_connections();

    allocate_buffers();
  }
}

static void exit_benchmark() {
  free_buffers();

  stats_thread.join();
}

bool test_p2p_send_done(int target_rank, int chunk_id) {
  auto* send_comm_ = &nodes[target_rank].send_comm_;
  bool done = ep->uccl_poll_ureq_once(&send_comm_->ureq[chunk_id]);
  if (done)
    tx_cur_sec_bytes.fetch_add(send_comm_->ureq[chunk_id].send.data_len);
  return done;
}

bool test_p2p_recv_done(int target_rank, int chunk_id) {
  auto* recv_comm_ = &nodes[target_rank].recv_comm_;
  bool done = ep->uccl_poll_ureq_once(&recv_comm_->ureq[chunk_id]);
  if (done)
    rx_cur_sec_bytes.fetch_add(recv_comm_->ureq[chunk_id].recv.data_len[0]);
  return done;
}

void p2p_send(int target_rank, int size) {
  // MPI_LOG(INFO) << "p2p send to " << target_rank;

  auto* send_comm_ = &nodes[target_rank].send_comm_;

  uint32_t offset = 0;
  uint32_t chunk_id = 0;

  while (offset < size) {
    int net_chunk_size = std::min(size - offset, (uint32_t)NET_CHUNK_SIZE);
    while (ep->uccl_send_async((UcclFlow*)send_comm_->conn_id.context,
                               send_comm_->mhandle, send_comm_->buffer,
                               net_chunk_size, &send_comm_->ureq[chunk_id])) {
    }

    offset += net_chunk_size;
    chunk_id++;
  }

  send_comm_->nb_net_chunk = chunk_id;
}

void p2p_receive(int target_rank, int size) {
  // MPI_LOG(INFO) << "p2p receive from " << target_rank;

  auto* recv_comm_ = &nodes[target_rank].recv_comm_;

  uint32_t offset = 0;
  uint32_t chunk_id = 0;

  while (offset < size) {
    int net_chunk_size = std::min(size - offset, (uint32_t)NET_CHUNK_SIZE);
    DCHECK(ep->uccl_recv_async((UcclFlow*)recv_comm_->conn_id.context,
                               &recv_comm_->mhandle, &recv_comm_->buffer,
                               &net_chunk_size, 1,
                               &recv_comm_->ureq[chunk_id]) == 0);

    offset += net_chunk_size;
    chunk_id++;
  }

  recv_comm_->nb_net_chunk = chunk_id;
}

void test_all_send(int target_rank) {
  auto nb_net_chunk = nodes[target_rank].send_comm_.nb_net_chunk;
  for (int i = 0; i < nb_net_chunk; i++) {
    while (!test_p2p_send_done(target_rank, i)) {
    }
  }
}

void test_all_recv(int target_rank) {
  auto nb_net_chunk = nodes[target_rank].recv_comm_.nb_net_chunk;
  for (int i = 0; i < nb_net_chunk; i++) {
    while (!test_p2p_recv_done(target_rank, i)) {
    }
  }
}

void net_sync(int target_rank) {
  test_all_send(target_rank);
  test_all_recv(target_rank);
}

void send_sync(int target_rank) { test_all_send(target_rank); }

void recv_sync(int target_rank) { test_all_recv(target_rank); }

void launch_stats_thread() {
  stats_thread = std::thread([&] {
    while (!quit) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      printf(
          "Rank %d: TX Tput: %.4f Gbps, RX Tput: %.4f Gbps, iterations: %d\n",
          LOCAL_RANK, (tx_cur_sec_bytes.load() - tx_prev_sec_bytes) * 8.0 / 1e9,
          (rx_cur_sec_bytes.load() - rx_prev_sec_bytes) * 8.0 / 1e9,
          cur_iteration);

      tx_prev_sec_bytes = tx_cur_sec_bytes.load();
      rx_prev_sec_bytes = rx_cur_sec_bytes.load();
    }
    return 0;
  });
}

void verify_params() { CHECK(FLAGS_size <= MAX_BUFFER_SIZE); }

void seq_alltoall() {
  while (cur_iteration++ < FLAGS_iterations && !quit) {
    for (int r = 1; r < NRANKS; r++) {
      int send_target = (LOCAL_RANK + r) % NRANKS;
      int recv_target = (LOCAL_RANK + NRANKS - r) % NRANKS;
      p2p_receive(recv_target, FLAGS_size);
      p2p_send(send_target, FLAGS_size);
      recv_sync(recv_target);
      send_sync(send_target);
    }
  }

  MPI_LOG(INFO) << "Sequential alltoall done.";
}

void alltoall() {
  while (cur_iteration++ < FLAGS_iterations && !quit) {
    for (int r = 0; r < NRANKS; r++) {
      if (r == LOCAL_RANK) continue;
      p2p_receive(r, FLAGS_size);
      p2p_send(r, FLAGS_size);
      net_sync(r);
    }
  }

  MPI_LOG(INFO) << "Alltoall done.";
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

void permutation_traffic() {
  int target_rank = find_target_rank("matrix.txt", LOCAL_RANK);

  MPI_LOG(INFO) << LOCAL_RANK << "'s target is " << target_rank;

  nodes[target_rank].client_setup();
  nodes[target_rank].server_setup();

  nodes[target_rank].wait_connect_done();
  nodes[target_rank].wait_accept_done();

  nodes[target_rank].send_comm_.buffer =
      mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  DCHECK(nodes[target_rank].send_comm_.buffer != MAP_FAILED);

  ep->uccl_regmr((UcclFlow*)nodes[target_rank].send_comm_.conn_id.context,
                 nodes[target_rank].send_comm_.buffer, MAX_BUFFER_SIZE, 0,
                 &nodes[target_rank].send_comm_.mhandle);

  nodes[target_rank].recv_comm_.buffer =
      mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  DCHECK(nodes[target_rank].recv_comm_.buffer != MAP_FAILED);

  ep->uccl_regmr((UcclFlow*)nodes[target_rank].recv_comm_.conn_id.context,
                 nodes[target_rank].recv_comm_.buffer, MAX_BUFFER_SIZE, 0,
                 &nodes[target_rank].recv_comm_.mhandle);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_LOG(INFO) << LOCAL_RANK << "'setup connection done.";

  while (cur_iteration++ < FLAGS_iterations && !quit) {
    p2p_receive(target_rank, FLAGS_size);
    p2p_send(target_rank, FLAGS_size);
    net_sync(target_rank);
  }

  MPI_LOG(INFO) << "Permutation traffic done.";
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

    if (strcmp(ifa->ifa_name, "eth0") == 0) {
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

  MPI_Comm_rank(MPI_COMM_WORLD, &LOCAL_RANK);

  MPI_Comm_size(MPI_COMM_WORLD, &NRANKS);

  get_ips();

  for (int i = 0; i < NRANKS; i++) nodes.push_back(NodeInfo(ips[i], i));

  // Initialize connections, allocate buffers, etc.
  init_benchmark();

  // Wait until all nodes are ready.
  MPI_Barrier(MPI_COMM_WORLD);

  // Launch stats thread.
  launch_stats_thread();

  // Run benchmark.
  if (FLAGS_benchtype == "SA")
    seq_alltoall();
  else if (FLAGS_benchtype == "PT")
    permutation_traffic();
  else if (FLAGS_benchtype == "AA")
    alltoall();

  // Destroy connections, free buffers, etc.
  exit_benchmark();

  MPI_Finalize();
  return 0;
}