#include "engine.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

Endpoint::Endpoint(const uint32_t local_gpu_idx, const uint32_t num_cpus)
    : local_gpu_idx_(local_gpu_idx), num_cpus_(num_cpus) {
  py::gil_scoped_release release;
  std::cout << "Creating Engine with GPU index: " << local_gpu_idx
            << ", CPUs: " << num_cpus << std::endl;

  // Initialize the RDMA endpoint with lazy creation.
  ep_ = new uccl::RDMAEndpoint(NUM_DEVICES, num_cpus);

  // Initialize the engine based on the GPU index.
  ep_->initialize_engine_by_dev(local_gpu_idx_);

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

  uccl::ConnID uccl_conn_id =
      ep_->test_uccl_connect(local_gpu_idx_, ip_addr, remote_gpu_idx);

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

  uccl::ConnID uccl_conn_id =
      ep_->test_uccl_accept(local_gpu_idx_, ip_addr, &remote_gpu_idx);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};

  return true;
}

bool Endpoint::reg_kv(void const* data, size_t size, uint64_t& mr_id) {
  py::gil_scoped_release release;
  std::cout << "Registering KV, size: " << size << " bytes" << std::endl;

  mr_id = next_mr_id_.fetch_add(1);

  uccl::Mhandle* mhandle;
  ep_->uccl_regmr(local_gpu_idx_, const_cast<void*>(data), size, 0, &mhandle);

  mr_id_to_mr_[mr_id] = new MR{mr_id, mhandle};

  return true;
}

bool Endpoint::send_kv(uint64_t conn_id, uint64_t mr_id, void const* data,
                       size_t size) {
  py::gil_scoped_release release;
  DCHECK(size <= 0xffffffff) << "size must be less than 4GB";
  std::cout << "Sending KV with mr_id: " << mr_id << ", size: " << size
            << " bytes" << std::endl;
  uccl::ucclRequest ureq;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  ep_->uccl_send_async(
      static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle, data,
      size, &ureq);

  ep_->uccl_poll_ureq(&ureq);

  std::cout << "KV sent successfully" << std::endl;
  return true;
}

bool Endpoint::recv_kv(uint64_t conn_id, uint64_t mr_id, void* data,
                       size_t max_size, size_t& recv_size) {
  py::gil_scoped_release release;

  std::cout << "Receiving KV with mr_id: " << mr_id << std::endl;
  uccl::ucclRequest ureq;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  int max_size_int = static_cast<int>(max_size);

  ep_->uccl_recv_async(
      static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
      &data, &max_size_int, 1, &ureq);

  ep_->uccl_poll_ureq(&ureq);

  recv_size = ureq.recv.data_len[0];

  std::cout << "KV received successfully, size: " << recv_size << " bytes"
            << std::endl;
  return true;
}
