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

thread_local bool large_kv_meta_data_registered_ = false;
thread_local LargeKVMetaData large_kv_meta_data_;
thread_local uint64_t large_kv_meta_data_mr_id_ = 0;

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
  if (!large_kv_meta_data_registered_) {
    py::gil_scoped_acquire acquire;
    bool rc = reg(&large_kv_meta_data_, sizeof(LargeKVMetaData),
                  large_kv_meta_data_mr_id_);
    DCHECK(rc) << "Failed to register large KV meta data";
    large_kv_meta_data_registered_ = true;
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

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  uccl::ucclRequest ureq[kMaxInflightChunks];
  bool done[kMaxInflightChunks] = {false};
  large_kv_meta_data_.total_size = size;
  auto mhandle_meta = mr_id_to_mr_[large_kv_meta_data_mr_id_]->mhandle_;

  // To avoid wasting the first RTT, we call uccl_send_async to try to send as
  // much as possible.
  int rc;
  int chunk_size = std::min(size, kRTTBytes);
  do {
    rc = ep_->uccl_send_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
        (void*)data, chunk_size, &ureq[0]);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  do {
    rc = ep_->uccl_send_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle_meta,
        (void*)&large_kv_meta_data_, sizeof(LargeKVMetaData), &ureq[1]);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  ep_->uccl_poll_ureq(&ureq[1]);
  ep_->uccl_poll_ureq(&ureq[0]);

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

bool Endpoint::recv(uint64_t conn_id, uint64_t mr_id, void* data,
                    size_t max_size, size_t& recv_size) {
  py::gil_scoped_release release;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  int max_size_int = static_cast<int>(max_size);

  uccl::ucclRequest ureq[kMaxInflightChunks];
  bool done[kMaxInflightChunks] = {false};
  auto mhandle_meta = mr_id_to_mr_[large_kv_meta_data_mr_id_]->mhandle_;

  // To avoid wasting the first RTT, we call uccl_recv_async to try to receive
  // as much as possible.
  int rc;
  do {
    int chunk_size = kRTTBytes;
    rc = ep_->uccl_recv_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
        &data, &chunk_size, 1, &ureq[0]);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  void* meta_addr = (void*)&large_kv_meta_data_;
  int meta_size = sizeof(LargeKVMetaData);
  do {
    rc = ep_->uccl_recv_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
        &mhandle_meta, &meta_addr, &meta_size, 1, &ureq[1]);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  ep_->uccl_poll_ureq(&ureq[1]);
  ep_->uccl_poll_ureq(&ureq[0]);

  auto first_actual_size = ureq[0].recv.data_len[0];
  size_t size_expected = large_kv_meta_data_.total_size;
  size_expected -= first_actual_size;
  data += first_actual_size;

  if (!size_expected) {
    recv_size = first_actual_size;
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

  recv_size = size_expected + first_actual_size;
  return true;
}
