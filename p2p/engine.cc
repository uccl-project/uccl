#include "engine.h"
#include "util/net.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

int const kMaxNumGPUs = 8;
// Assume the local and remote GPUs have the same GPU-NIC mapping.
uint8_t gpu_to_dev[kMaxNumGPUs] = {0};
std::once_flag glog_init_once;

inline void check_python_signals() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  if (PyErr_CheckSignals() != 0) {
    std::cerr << "Python signal caught, exiting..." << std::endl;
    std::abort();
  }
  PyGILState_Release(gstate);
}

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus)
    : local_gpu_idx_(local_gpu_idx), num_cpus_(num_cpus) {
  py::gil_scoped_release release;
  std::cout << "Creating Engine with GPU index: " << local_gpu_idx
            << ", CPUs: " << num_cpus << std::endl;
  // Py_Initialize();

  std::call_once(glog_init_once,
                 []() { google::InitGoogleLogging("uccl_p2p"); });

  google::InstallFailureSignalHandler();

  // Initialize the RDMA endpoint with lazy creation.
  ep_ = new uccl::RDMAEndpoint(ucclParamNUM_ENGINES());

  auto gpu_cards = uccl::get_gpu_cards();
  DCHECK(local_gpu_idx_ < gpu_cards.size() && gpu_cards.size() <= kMaxNumGPUs)
      << "Local GPU index out of range";

  auto ib_nics = uccl::get_rdma_nics();
  // Find the RDMA NIC that is closest to each of the GPUs,
  // ensuring fair NIC allocation.
  std::vector<bool> nic_allocated(ib_nics.size(), false);
  for (int i = 0; i < gpu_cards.size(); i++) {
    auto gpu_device_path = gpu_cards[i];
    int best_nic = -1;
    int best_distance = std::numeric_limits<int>::max();
    for (int j = 0; j < ib_nics.size(); ++j) {
      if (nic_allocated[j]) continue;
      int dist = uccl::cal_pcie_distance(gpu_device_path, ib_nics[j].second);
      if (dist < best_distance) {
        best_distance = dist;
        best_nic = j;
      }
    }
    if (best_nic != -1) {
      gpu_to_dev[i] = best_nic;
      nic_allocated[best_nic] = true;
    } else {
      // If all NICs are allocated, fallback to the closest
      auto ib_nic_it = std::min_element(
          ib_nics.begin(), ib_nics.end(), [&](auto const& a, auto const& b) {
            return uccl::cal_pcie_distance(gpu_device_path, a.second) <
                   uccl::cal_pcie_distance(gpu_device_path, b.second);
          });
      gpu_to_dev[i] = ib_nic_it - ib_nics.begin();
    }
  }
  std::cout << "Detected best GPU-NIC mapping: " << std::endl;
  for (int i = 0; i < gpu_cards.size(); i++) {
    std::cout << "\tGPU " << i << " -> NIC " << gpu_to_dev[i] << " ("
              << ib_nics[gpu_to_dev[i]].first << ")" << std::endl;
  }
  std::cout << std::endl;

  // Initialize the engine based on the GPU index.
#ifdef LAZY_CREATE_ENGINE
  printf("Lazy creation of engine, GPU index: %d\n", local_gpu_idx_);
  ep_->initialize_engine_by_dev(gpu_to_dev[local_gpu_idx_], false);
  printf("Engine initialized for GPU %d\n", local_gpu_idx_);
#endif

  {
    transfer_metadata_ = new TransferMetaData();
    py::gil_scoped_acquire acquire;
    bool rc = reg(transfer_metadata_->chunk_size_v, sizeof(TransferMetaData),
                  transfer_metadata_mr_id_);
    DCHECK(rc) << "Failed to register large KV meta data";
  }
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
  delete transfer_metadata_;
  std::cout << "Engine destroyed" << std::endl;
}

bool Endpoint::connect(std::string const& ip_addr, int const& remote_gpu_idx,
                       uint64_t& conn_id, int remote_port) {
  py::gil_scoped_release release;
  std::cout << "Attempting to connect to " << ip_addr << ":" << remote_gpu_idx
            << std::endl;

  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  std::future<uccl::ConnID> uccl_conn_id_future = std::async(
      std::launch::async, [this, remote_gpu_idx, &ip_addr, remote_port]() {
        if (remote_port == -1) {
          throw std::runtime_error("Error: remote_port is not set.");
          return ep_->test_uccl_connect(
              gpu_to_dev[local_gpu_idx_], local_gpu_idx_,
              gpu_to_dev[remote_gpu_idx], remote_gpu_idx, ip_addr);
        } else {
          return ep_->uccl_connect(gpu_to_dev[local_gpu_idx_], local_gpu_idx_,
                                   gpu_to_dev[remote_gpu_idx], remote_gpu_idx,
                                   ip_addr, remote_port);
        }
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (uccl_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    check_python_signals();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  uccl::ConnID uccl_conn_id = uccl_conn_id_future.get();

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};

  return true;
}

bool Endpoint::connect(py::bytes const& meta_bytes, uint64_t& conn_id) {
  std::string buf = meta_bytes;
  uint8_t const* data = reinterpret_cast<uint8_t const*>(buf.data());
  const size_t n = buf.size();

  std::string ip;
  int port = -1;
  int remote_gpu_idx = -1;

  char ipstr[INET6_ADDRSTRLEN] = {0};

  if (n == 10) {  // IPv4 format
    std::memcpy(ipstr, data, 4);
    inet_ntop(AF_INET, ipstr, ipstr, sizeof(ipstr));
    ip = ipstr;
    port = (data[4] << 8) | data[5];
    std::memcpy(&remote_gpu_idx, data + 6, 4);  // host byte-order
  } else if (n == 22) {                         // IPv6 format
    std::memcpy(ipstr, data, 16);
    inet_ntop(AF_INET6, ipstr, ipstr, sizeof(ipstr));
    ip = ipstr;
    port = (data[16] << 8) | data[17];
    std::memcpy(&remote_gpu_idx, data + 18, 4);
  } else {
    throw std::runtime_error("Endpoint::connect(metadata): invalid blob size");
  }

  return this->connect(ip, remote_gpu_idx, conn_id, port);
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  py::gil_scoped_release release;
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // For demo purposes, simulate accepted connection
  conn_id = next_conn_id_.fetch_add(1);

  std::future<uccl::ConnID> uccl_conn_id_future =
      std::async(std::launch::async, [this, &ip_addr, &remote_gpu_idx]() {
        return ep_->test_uccl_accept(gpu_to_dev[local_gpu_idx_], local_gpu_idx_,
                                     ip_addr, &remote_gpu_idx);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (uccl_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    check_python_signals();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  uccl::ConnID uccl_conn_id = uccl_conn_id_future.get();

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
  if (mhandle->mr == nullptr) {
    std::cerr << "[Endpoint::reg] Failed to register memory region, "
              << "mhandle->mr is null\n";
    std::abort();
  }
  mr_id_to_mr_[mr_id] = new MR{mr_id, mhandle};

  return true;
}

bool Endpoint::send(uint64_t conn_id, uint64_t mr_id, void const* data,
                    size_t size) {
  py::gil_scoped_release release;
  DCHECK(size <= 0xffffffff) << "size must be less than 4GB";

  // printf(
  //     "[Endpoint::send] conn_id: %lu, mr_id: %lu, data: %p, "
  //     "size: %zu\n",
  //     conn_id, mr_id, data, size);

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  uccl::ucclRequest ureq[kMaxInflightChunks];
  bool done[kMaxInflightChunks] = {false};
  transfer_metadata_->chunk_size_v[0] = size;
  auto mhandle_meta = mr_id_to_mr_[transfer_metadata_mr_id_]->mhandle_;

  // To avoid wasting the first RTT, we call uccl_send_async to try to send as
  // much as possible.
  int rc;
  int chunk_size = std::min(size, kRTTBytes);
  do {
    rc = ep_->uccl_send_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
        (void*)data, chunk_size, &ureq[0]);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  do {
    rc = ep_->uccl_send_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle_meta,
        (void*)transfer_metadata_->chunk_size_v, sizeof(uint64_t), &ureq[1]);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  while (!ep_->uccl_poll_ureq_once(&ureq[1])) {
    check_python_signals();
  }
  while (!ep_->uccl_poll_ureq_once(&ureq[0])) {
    check_python_signals();
  }

  data += chunk_size;
  size -= chunk_size;

  if (size == 0) return true;

  void* cur_data = const_cast<void*>(data);
  size_t size_sent = 0;
  int ureq_max = (size + kChunkSize - 1) / kChunkSize;
  int ureq_issued = 0, ureq_finished = 0;

  while (ureq_finished < ureq_max) {
    while (ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_sent < size) {
      size_t chunk_size = std::min(size - size_sent, kChunkSize);
      auto rc = ep_->uccl_send_async(
          static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
          cur_data, chunk_size, &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      cur_data += chunk_size;
      size_sent += chunk_size;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    check_python_signals();

    // First, poll all outstanding requests and mark which ones are done.
    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        // Just mark it as completed, DO NOT increment ureq_finished here.
        done[i % kMaxInflightChunks] = true;
      }
    }

    // Now, advance the ureq_finished counter as far as possible.
    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

std::vector<uint8_t> Endpoint::get_endpoint_metadata() {
  char uccl_ifname[MAX_IF_NAME_SIZE + 1];
  uccl::socketAddress uccl_ifaddr;
  int num_ifs =
      uccl::find_interfaces(uccl_ifname, &uccl_ifaddr, MAX_IF_NAME_SIZE, 1);
  if (num_ifs != 1) UCCL_INIT_CHECK(false, "No IP interface found");

  std::string ip_str = uccl::get_dev_ip(uccl_ifname);
  uint16_t port = ep_->get_port();
  // int listen_fd = uccl::open_ephemeral_port(port);
  // if (listen_fd < 0) {
  //   throw std::runtime_error("Failed to open ephemeral port");
  // }

  bool is_ipv6 = ip_str.find(':') != std::string::npos;
  size_t ip_len = is_ipv6 ? 16 : 4;

  // Additional 2 bytes for port and 4 bytes for local_gpu_idx_
  size_t total_len = ip_len + 2 + sizeof(int);
  std::vector<uint8_t> metadata(total_len);

  // Copy IP
  if (is_ipv6) {
    struct in6_addr ip6_bin;
    if (inet_pton(AF_INET6, ip_str.c_str(), &ip6_bin) != 1)
      throw std::runtime_error("Invalid IPv6 address: " + ip_str);
    std::memcpy(metadata.data(), &ip6_bin, 16);
  } else {
    struct in_addr ip4_bin;
    if (inet_pton(AF_INET, ip_str.c_str(), &ip4_bin) != 1)
      throw std::runtime_error("Invalid IPv4 address: " + ip_str);
    std::memcpy(metadata.data(), &ip4_bin, 4);
  }

  // Copy port in network byte order
  uint16_t net_port = htons(port);
  std::memcpy(metadata.data() + ip_len, &net_port, 2);

  // Copy local_gpu_idx_ in host byte order
  std::memcpy(metadata.data() + ip_len + 2, &local_gpu_idx_, sizeof(int));

  return metadata;
}

bool Endpoint::recv(uint64_t conn_id, uint64_t mr_id, void* data,
                    size_t max_size, size_t* recv_size) {
  py::gil_scoped_release release;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  int max_size_int = static_cast<int>(max_size);

  uccl::ucclRequest ureq[kMaxInflightChunks];
  bool done[kMaxInflightChunks] = {false};
  auto mhandle_meta = mr_id_to_mr_[transfer_metadata_mr_id_]->mhandle_;

  // To avoid wasting the first RTT, we call uccl_recv_async to try to receive
  // as much as possible.
  int rc;
  do {
    int chunk_size = kRTTBytes;
    rc = ep_->uccl_recv_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
        &data, &chunk_size, 1, &ureq[0]);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  void* meta_addr = (void*)transfer_metadata_->chunk_size_v;
  int meta_size = sizeof(uint64_t);
  do {
    rc = ep_->uccl_recv_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
        &mhandle_meta, &meta_addr, &meta_size, 1, &ureq[1]);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  while (!ep_->uccl_poll_ureq_once(&ureq[1])) {
    check_python_signals();
  }
  while (!ep_->uccl_poll_ureq_once(&ureq[0])) {
    check_python_signals();
  }

  auto first_actual_size = ureq[0].recv.data_len[0];
  size_t size_expected = transfer_metadata_->chunk_size_v[0];
  size_expected -= first_actual_size;
  data += first_actual_size;

  if (!size_expected) {
    *recv_size = first_actual_size;
    return true;
  }

  void* cur_data = data;
  size_t size_post_recv = 0;
  int ureq_max = (size_expected + kChunkSize - 1) / kChunkSize;
  int ureq_issued = 0, ureq_finished = 0;

  while (ureq_finished < ureq_max) {
    while (ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_post_recv < size_expected) {
      int chunk_size = std::min(size_expected - size_post_recv, kChunkSize);
      auto rc = ep_->uccl_recv_async(
          static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
          &cur_data, &chunk_size, 1, &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      cur_data += chunk_size;
      size_post_recv += chunk_size;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    check_python_signals();

    // First, poll all outstanding requests and mark which ones are done.
    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        // Just mark it as completed, DO NOT increment ureq_finished here.
        done[i % kMaxInflightChunks] = true;
      }
    }

    // Now, advance the ureq_finished counter as far as possible.
    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  *recv_size = size_expected + first_actual_size;
  return true;
}

bool Endpoint::sendv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void const*> data_v,
                     std::vector<size_t> size_v, size_t num_iovs) {
  py::gil_scoped_release release;
  auto conn = conn_id_to_conn_[conn_id];
  auto uccl_flow = static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context);

  uccl::ucclRequest ureq[kMaxInflightChunks];
  bool done[kMaxInflightChunks] = {false};

  int estimated_ureq_max = 0;
  for (int i = 0; i < num_iovs; i++) {
    transfer_metadata_->chunk_size_v[i] = size_v[i];
    estimated_ureq_max += (size_v[i] + kChunkSize - 1) / kChunkSize;
  }

  auto mhandle = mr_id_to_mr_[mr_id_v[0]]->mhandle_;
  auto mhandle_meta = mr_id_to_mr_[transfer_metadata_mr_id_]->mhandle_;

  int rc;
  int chunk_size = std::min(size_v[0], kRTTBytes);
  do {
    rc = ep_->uccl_send_async(uccl_flow, mhandle, (void*)data_v[0], chunk_size,
                              &ureq[0]);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  do {
    rc = ep_->uccl_send_async(uccl_flow, mhandle_meta,
                              (void*)transfer_metadata_->chunk_size_v,
                              sizeof(uint64_t) * num_iovs, &ureq[1]);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  while (!ep_->uccl_poll_ureq_once(&ureq[1])) {
    check_python_signals();
  }

  while (!ep_->uccl_poll_ureq_once(&ureq[0])) {
    check_python_signals();
  }

  std::vector<void*> data_send_vec;
  std::vector<size_t> size_send_vec;
  std::vector<uccl::Mhandle*> mhandle_send_vec;
  // Avoid reallocations.
  data_send_vec.reserve(estimated_ureq_max);
  size_send_vec.reserve(estimated_ureq_max);
  mhandle_send_vec.reserve(estimated_ureq_max);

  auto first_actual_size = chunk_size;
  void* cur_data = (void*)data_v[0] + first_actual_size;
  size_t cur_size_expected = size_v[0] - first_actual_size;
  size_t cur_size_post_send = 0;
  for (int i = 0; i < num_iovs; i++) {
    if (i != 0) {
      cur_data = (void*)data_v[i];
      cur_size_expected = size_v[i];
      cur_size_post_send = 0;
    }
    while (cur_size_post_send < cur_size_expected) {
      int chunk_size =
          std::min(cur_size_expected - cur_size_post_send, kChunkSize);
      data_send_vec.push_back(cur_data);
      size_send_vec.push_back(chunk_size);
      mhandle_send_vec.push_back(mr_id_to_mr_[mr_id_v[i]]->mhandle_);
      cur_data += chunk_size;
      cur_size_post_send += chunk_size;
    }
  }

  int ureq_max = data_send_vec.size();
  int ureq_issued = 0, ureq_finished = 0;

  while (ureq_finished < ureq_max) {
    while (ureq_issued < ureq_max &&
           ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_send_vec[ureq_issued] > 0) {
      auto rc = ep_->uccl_send_async(
          uccl_flow, mhandle_send_vec[ureq_issued], data_send_vec[ureq_issued],
          size_send_vec[ureq_issued], &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    check_python_signals();

    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        done[i % kMaxInflightChunks] = true;
      }
    }

    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

bool Endpoint::recvv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void*> data_v, std::vector<size_t> max_size_v,
                     std::vector<size_t>& recv_size_v, size_t num_iovs) {
  py::gil_scoped_release release;
  auto conn = conn_id_to_conn_[conn_id];
  auto uccl_flow = static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context);

  uccl::ucclRequest ureq[kMaxInflightChunks];
  bool done[kMaxInflightChunks] = {false};

  auto mhandle = mr_id_to_mr_[mr_id_v[0]]->mhandle_;
  auto mhandle_meta = mr_id_to_mr_[transfer_metadata_mr_id_]->mhandle_;

  int rc;
  do {
    int chunk_size = kRTTBytes;
    rc = ep_->uccl_recv_async(uccl_flow, &mhandle, &data_v[0], &chunk_size, 1,
                              &ureq[0]);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  void* meta_addr = (void*)transfer_metadata_->chunk_size_v;
  int meta_size = sizeof(uint64_t) * num_iovs;
  do {
    rc = ep_->uccl_recv_async(uccl_flow, &mhandle_meta, &meta_addr, &meta_size,
                              1, &ureq[1]);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  while (!ep_->uccl_poll_ureq_once(&ureq[1])) {
    check_python_signals();
  }

  while (!ep_->uccl_poll_ureq_once(&ureq[0])) {
    check_python_signals();
  }

  // Prepare the data, size, and mhandle vectors for the rest of the chunks.
  std::vector<void*> data_recv_vec;
  std::vector<void**> data_recv_ptr_vec;
  std::vector<int> size_recv_vec;
  std::vector<uccl::Mhandle*> mhandle_recv_vec;
  std::vector<uccl::Mhandle**> mhandle_recv_ptr_vec;

  int estimated_ureq_max = 0;
  for (int i = 0; i < num_iovs; i++) {
    estimated_ureq_max += (max_size_v[i] + kChunkSize - 1) / kChunkSize;
  }

  data_recv_vec.reserve(estimated_ureq_max);
  data_recv_ptr_vec.reserve(estimated_ureq_max);
  size_recv_vec.reserve(estimated_ureq_max);
  mhandle_recv_vec.reserve(estimated_ureq_max);
  mhandle_recv_ptr_vec.reserve(estimated_ureq_max);

  auto first_actual_size = ureq[0].recv.data_len[0];
  void* cur_data = data_v[0] + first_actual_size;
  size_t cur_size_expected =
      transfer_metadata_->chunk_size_v[0] - first_actual_size;
  size_t cur_size_post_recv = 0;
  for (int i = 0; i < num_iovs; i++) {
    if (i != 0) {
      cur_data = data_v[i];
      cur_size_expected = transfer_metadata_->chunk_size_v[i];
      cur_size_post_recv = 0;
    }
    while (cur_size_post_recv < cur_size_expected) {
      int chunk_size =
          std::min(cur_size_expected - cur_size_post_recv, kChunkSize);
      data_recv_vec.push_back(cur_data);
      data_recv_ptr_vec.push_back(&data_recv_vec.back());
      size_recv_vec.push_back(chunk_size);
      mhandle_recv_vec.push_back(mr_id_to_mr_[mr_id_v[i]]->mhandle_);
      mhandle_recv_ptr_vec.push_back(&mhandle_recv_vec.back());
      cur_data += chunk_size;
      cur_size_post_recv += chunk_size;
    }
  }

  // Handle receiving the rest of the sub-chunks.
  int ureq_max = data_recv_vec.size();
  int ureq_issued = 0, ureq_finished = 0;

  while (ureq_finished < ureq_max) {
    while (ureq_issued < ureq_max &&
           ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_recv_vec[ureq_issued] > 0) {
      auto rc = ep_->uccl_recv_async(
          uccl_flow, mhandle_recv_ptr_vec[ureq_issued],
          data_recv_ptr_vec[ureq_issued], &size_recv_vec[ureq_issued], 1,
          &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    check_python_signals();

    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        done[i % kMaxInflightChunks] = true;
      }
    }

    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  // Return the sizes of the received chunks.
  for (int i = 0; i < num_iovs; i++) {
    recv_size_v[i] = transfer_metadata_->chunk_size_v[i];
    DCHECK(recv_size_v[i] <= max_size_v[i])
        << "recv_size_v > max_size_v for chunk " << i;
  }

  return true;
}

bool Endpoint::send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                          size_t size, uint64_t* transfer_id) {
  py::gil_scoped_release release;
  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  auto _transfer_id = next_transfer_id_.fetch_add(1);
  auto* ureq = new uccl::ucclRequest();

  *transfer_id = _transfer_id;
  transfer_id_to_ureq_[_transfer_id] = ureq;

  int rc;
  do {
    rc = ep_->uccl_send_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
        (void*)data, size, ureq);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  return true;
}

bool Endpoint::recv_async(uint64_t conn_id, uint64_t mr_id, void* data,
                          size_t size, uint64_t* transfer_id) {
  py::gil_scoped_release release;
  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  auto _transfer_id = next_transfer_id_.fetch_add(1);
  auto* ureq = new uccl::ucclRequest();

  *transfer_id = _transfer_id;
  transfer_id_to_ureq_[_transfer_id] = ureq;
  int size_int = static_cast<int>(size);

  int rc;
  do {
    rc = ep_->uccl_recv_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
        &data, &size_int, 1, ureq);
    if (rc == -1) {
      check_python_signals();
      std::this_thread::yield();
    }
  } while (rc == -1);

  return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  py::gil_scoped_release release;
  auto* ureq = transfer_id_to_ureq_.at(transfer_id);
  *is_done = ep_->uccl_poll_ureq_once(ureq);
  if (*is_done) {
    delete ureq;
    transfer_id_to_ureq_.erase(transfer_id);
  }
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
                          int my_rank, int remote_gpu_idx) {
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
    std::cout << "[join_group] peer " << r << ": " << peers[r].ip_addr << ":"
              << peers[r].gpu_idx << "\n";
    std::cout << "[join_group] my rank: " << my_rank << ", peer rank: " << r
              << ", world_size: " << world_size << "\n";
    if (r == my_rank) continue;
    uint64_t cid;
    if (my_rank < r) {
      if (!accept(peers[r].ip_addr, peers[r].gpu_idx, cid)) {
        std::cerr << "[join_group] accept to rank " << r << " failed\n";
        return false;
      } else {
        std::cout << "[join_group] accepted to rank " << r << " with conn_id "
                  << cid << "\n";
      }
    } else {
      if (!connect(peers[r].ip_addr, remote_gpu_idx, cid, -1)) {
        std::cerr << "[join_group] connect from rank " << r << " failed\n";
        return false;
      }
    }
    rank2conn_[r] = cid;
  }
  return true;
}

std::unique_ptr<Endpoint> Endpoint::CreateAndJoin(
    std::string const& discovery_uri, std::string const& group_name,
    int world_size, int my_rank, uint32_t local_gpu_idx, uint32_t num_cpus,
    int remote_gpu_idx) {
  auto ep = std::make_unique<Endpoint>(local_gpu_idx, num_cpus);
  if (!ep->join_group(discovery_uri, group_name, world_size, my_rank,
                      remote_gpu_idx)) {
    throw std::runtime_error("Endpoint::CreateAndJoin() failed");
  }
  return ep;
}

#ifdef USE_REDIS
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
      std::cout << "[fetch_all_redis] Peer " << rank << ": " << s << std::endl;

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
#endif