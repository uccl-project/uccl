#include "engine.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

int const kMaxNumGPUs = 8;
// Assume the local and remote GPUs have the same GPU-NIC mapping.
uint8_t gpu_to_dev[kMaxNumGPUs] = {0};

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus)
    : local_gpu_idx_(local_gpu_idx), num_cpus_(num_cpus) {
  py::gil_scoped_release release;
  std::cout << "Creating Engine with GPU index: " << local_gpu_idx
            << ", CPUs: " << num_cpus << std::endl;

  google::InitGoogleLogging("uccl_p2p");
  google::InstallFailureSignalHandler();

  // Initialize the RDMA endpoint with lazy creation.
  ep_ = new uccl::RDMAEndpoint(ucclParamNUM_ENGINES());

  auto gpu_cards = uccl::get_gpu_cards();
  DCHECK(local_gpu_idx_ < gpu_cards.size() && gpu_cards.size() <= kMaxNumGPUs)
      << "Local GPU index out of range";

  auto ib_nics = uccl::get_rdma_nics();
  // Find the RDMA NIC that is closest to each of the GPUs.
  for (int i = 0; i < kMaxNumGPUs; i++) {
    auto gpu_device_path = gpu_cards[i];
    auto ib_nic_it = std::min_element(
        ib_nics.begin(), ib_nics.end(), [&](auto const& a, auto const& b) {
          return uccl::cal_pcie_distance(gpu_device_path, a.second) <
                 uccl::cal_pcie_distance(gpu_device_path, b.second);
        });
    gpu_to_dev[i] = ib_nic_it - ib_nics.begin();
  }
  std::cout << "Detected best GPU-NIC mapping: " << std::endl;
  for (int i = 0; i < kMaxNumGPUs; i++) {
    std::cout << "\tGPU " << i << " -> NIC " << gpu_to_dev[i] << " ("
              << ib_nics[gpu_to_dev[i]].first << ")" << std::endl;
  }
  std::cout << std::endl;

  // Initialize the engine based on the GPU index.
#ifdef LAZY_CREATE_ENGINE
  ep_->initialize_engine_by_dev(gpu_to_dev[local_gpu_idx_]);
#endif

  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::~Endpoint() {
  py::gil_scoped_release release;
  std::cout << "Destroying Engine..." << std::endl;
  delete ep_;

  for (auto& [conn_id, conn] : conn_id_to_conn_) {
    delete conn;
  }
  for (auto& [mr_id, mr] : mr_id_to_mr_) {
    delete mr;
  }

  std::cout << "Engine destroyed" << std::endl;
}

bool Endpoint::connect(std::string const& ip_addr, int const& remote_gpu_idx,
                       uint64_t& conn_id) {
  py::gil_scoped_release release;
  std::cout << "Attempting to connect to " << ip_addr << ":" << remote_gpu_idx
            << std::endl;

  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  uccl::ConnID uccl_conn_id = ep_->test_uccl_connect(
      gpu_to_dev[local_gpu_idx_], local_gpu_idx_, gpu_to_dev[remote_gpu_idx],
      remote_gpu_idx, ip_addr);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};

  return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  py::gil_scoped_release release;
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // For demo purposes, simulate accepted connection
  conn_id = next_conn_id_.fetch_add(1);

  uccl::ConnID uccl_conn_id = ep_->test_uccl_accept(
      gpu_to_dev[local_gpu_idx_], local_gpu_idx_, ip_addr, &remote_gpu_idx);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};

  return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  py::gil_scoped_release release;

  mr_id = next_mr_id_.fetch_add(1);

  uccl::Mhandle* mhandle;
  ep_->uccl_regmr(gpu_to_dev[local_gpu_idx_], const_cast<void*>(data), size, 0,
                  &mhandle);

  mr_id_to_mr_[mr_id] = new MR{mr_id, mhandle};

  return true;
}

bool Endpoint::send(uint64_t conn_id, uint64_t mr_id, void const* data,
                    size_t size) {
  py::gil_scoped_release release;
  DCHECK(size <= 0xffffffff) << "size must be less than 4GB";
  uccl::ucclRequest ureq;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  int rc;
  do {
    rc = ep_->uccl_send_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
        data, size, &ureq);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  ep_->uccl_poll_ureq(&ureq);

  return true;
}

bool Endpoint::recv(uint64_t conn_id, uint64_t mr_id, void* data,
                    size_t max_size, size_t& recv_size) {
  py::gil_scoped_release release;
  uccl::ucclRequest ureq;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  int max_size_int = static_cast<int>(max_size);

  int rc;
  do {
    rc = ep_->uccl_recv_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
        &data, &max_size_int, 1, &ureq);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  ep_->uccl_poll_ureq(&ureq);

  recv_size = ureq.recv.data_len[0];
  return true;
}

bool Endpoint::publish_peer(std::string const& discovery_uri,
                            std::string const& group_name, int rank,
                            PeerInfo const& info) {
  if (discovery_uri.rfind("redis://", 0) == 0) {
#ifdef USE_REDIS
    std::string key = group_name + ":" + std::to_string(rank);
    return publish_redis(discovery_uri, key, info);
#else
    std::cerr << "[publish_peer] Redis support not compiled in\n";
    return false;
#endif
  } else {
    std::cerr << "[publish_peer] Unsupported discovery backend: "
              << discovery_uri << "\n";
    return false;
  }
}

bool Endpoint::collect_peers(std::string const& discovery_uri,
                             std::string const& group_name, int world_size,
                             std::vector<PeerInfo>& out) {
  if (discovery_uri.rfind("redis://", 0) == 0) {
#ifdef USE_REDIS
    std::string key_prefix = group_name + ":";
    return fetch_all_redis(discovery_uri, key_prefix, world_size, out);
#else
    std::cerr << "[collect_peers] Redis support not compiled in\n";
    return false;
#endif
  } else {
    std::cerr << "[collect_peers] Unsupported discovery backend: "
              << discovery_uri << "\n";
    return false;
  }
}

uint64_t Endpoint::conn_id_of_rank(int rank) const {
  auto it = rank2conn_.find(rank);
  return it != rank2conn_.end() ? it->second : UINT64_MAX;
}

bool Endpoint::join_group(std::string const& discovery_uri,
                          std::string const& group_name, int world_size,
                          int my_rank, uint16_t listen_port) {
  std::string local_ip;
  {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_addr.s_addr = inet_addr("8.8.8.8");
    serv.sin_port = htons(53);
    ::connect(sock, (sockaddr*)&serv, sizeof(serv));
    sockaddr_in name{};
    socklen_t namelen = sizeof(name);
    getsockname(sock, (sockaddr*)&name, &namelen);
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &name.sin_addr, buf, sizeof(buf));
    local_ip = buf;
    close(sock);
  }

  PeerInfo me{local_ip, static_cast<int>(local_gpu_idx_)};
  if (!publish_peer(discovery_uri, group_name, my_rank, me)) {
    std::cerr << "[join_group] failed to publish peer info\n";
    return false;
  }

  std::vector<PeerInfo> peers;
  if (!collect_peers(discovery_uri, group_name, world_size, peers)) {
    std::cerr << "[join_group] failed to collect peers\n";
    return false;
  }

  /* Low errank connect, higher rank accept. */
  for (int r = 0; r < world_size; ++r) {
    if (r == my_rank) continue;
    uint64_t cid;
    if (my_rank < r) {
      if (!connect(peers[r].ip_addr, peers[r].gpu_idx, cid)) {
        std::cerr << "[join_group] connect to rank " << r << " failed\n";
        return false;
      }
    } else {
      std::string peer_ip;
      int peer_gpu;
      if (!accept(peer_ip, peer_gpu, cid)) {
        std::cerr << "[join_group] accept from rank " << r << " failed\n";
        return false;
      }
    }
    rank2conn_[r] = cid;
  }
  return true;
}

std::unique_ptr<Endpoint> Endpoint::CreateAndJoin(
    std::string const& discovery_uri, std::string const& group_name,
    int world_size, int my_rank, uint32_t local_gpu_idx, uint32_t num_cpus) {
  auto ep = std::make_unique<Endpoint>(local_gpu_idx, num_cpus);
  uint16_t dummy_listen_port = 0;  // not used in this stub
  if (!ep->join_group(discovery_uri, group_name, world_size, my_rank,
                      dummy_listen_port)) {
    throw std::runtime_error("Endpoint::CreateAndJoin() failed");
  }
  return ep;
}

bool Endpoint::publish_redis(std::string const& redis_uri,
                             std::string const& key, PeerInfo const& info) {
  try {
    auto redis = sw::redis::Redis(redis_uri);
    std::ostringstream oss;
    oss << info.ip_addr << "," << info.gpu_idx;
    redis.set(key, oss.str());
    return true;
  } catch (sw::redis::Error const& e) {
    std::cerr << "[publish_redis] Redis error: " << e.what() << "\n";
    return false;
  }
}

bool Endpoint::fetch_all_redis(std::string const& redis_uri,
                               std::string const& key_prefix, int world_size,
                               std::vector<PeerInfo>& out) {
  try {
    auto redis = sw::redis::Redis(redis_uri);
    out.clear();
    out.reserve(world_size);

    for (int rank = 0; rank < world_size; ++rank) {
      std::string key = key_prefix + std::to_string(rank);
      std::optional<std::string> val;

      while (true) {
        val = redis.get(key);
        if (val) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      auto const& s = *val;
      auto comma = s.find(',');
      if (comma == std::string::npos) {
        std::cerr << "[fetch_all_redis] bad format for key " << key << "\n";
        return false;
      }
      PeerInfo p;
      p.ip_addr = s.substr(0, comma);
      p.gpu_idx = std::stoi(s.substr(comma + 1));
      out.push_back(p);
    }
    return true;
  } catch (sw::redis::Error const& e) {
    std::cerr << "[fetch_all_redis] Redis error: " << e.what() << "\n";
    return false;
  }
}