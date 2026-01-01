#include "test.h"
#include "transport.h"
#include "util/gpu_rt.h"
#include <iostream>
#include <vector>

void cqpoller_client_thread(std::shared_ptr<Communicator> comm, int peer_rank) {
  if (comm->connect_to(peer_rank)) {
    std::cout << "[CLIENT] Successfully connected to rank " << peer_rank
              << std::endl;
  } else {
    std::cerr << "[CLIENT] Failed to connect to rank " << peer_rank
              << std::endl;
    return;
  }

  auto [ep, epok] = comm->get_endpoint_by_rank(peer_rank);

  float* d_data;
  size_t count = 1024;
  size_t size = count * sizeof(float);
  GPU_RT_CHECK(gpuMalloc((void**)&d_data, size));

  // Make host buffer
  std::vector<float> h_data(count);
  for (size_t i = 0; i < count; ++i) h_data[i] = static_cast<float>(i);

  // Copy to GPU
  GPU_RT_CHECK(gpuMemcpy(d_data, h_data.data(), size, gpuMemcpyHostToDevice));

  auto mr = comm->reg_mr(d_data, size);
  auto remote_mr = comm->wait_mr_notify(peer_rank);

  std::cout << "[CLIENT] Got remote MR id=" << remote_mr.id << " addr=0x"
            << std::hex << remote_mr.address << " len=" << std::dec
            << remote_mr.length << "\n";

  unsigned rid =
      comm->isend(peer_rank, d_data, 0, size, mr.id, remote_mr.id, true);
  if (rid == 0) {
    std::cerr << "[CLIENT] isend failed\n";
  } else {
    std::cout << "[CLIENT] isend posted\n";
  }

  comm->wait_finish(rid);
  std::cout << "[CLIENT] send completed\n";

  GPU_RT_CHECK(gpuFree(d_data));
}

void cqpoller_server_thread(std::shared_ptr<Communicator> comm, int peer_rank) {
  if (comm->accept_from(peer_rank)) {
    std::cout << "[SERVER] Successfully accepted connection from rank "
              << peer_rank << std::endl;
  } else {
    std::cerr << "[SERVER] Failed to accept connection from rank " << peer_rank
              << std::endl;
    return;
  }

  float* d_data;
  size_t count = 1024;
  size_t size = count * sizeof(float);
  GPU_RT_CHECK(gpuMalloc((void**)&d_data, size));

  auto mr = comm->reg_mr(d_data, size);
  comm->notify_mr(peer_rank, mr);
  sleep(1);  // test recv ceq before post irecv
  unsigned rid = comm->irecv(peer_rank, d_data, 0, size, true);
  if (rid == 0) {
    std::cerr << "[SERVER] irecv failed\n";
    GPU_RT_CHECK(gpuFree(d_data));
    return;
  } else {
    std::cout << "[SERVER] irecv posted\n";
  }

  comm->wait_finish(rid);

  // Copy and check data
  std::vector<float> h_recv(count);
  GPU_RT_CHECK(gpuMemcpy(h_recv.data(), d_data, size, gpuMemcpyDeviceToHost));

  bool valid = true;
  for (size_t i = 0; i < count; ++i) {
    if (h_recv[i] != static_cast<float>(i)) {
      std::cerr << "[CHECK] mismatch at idx=" << i << ", expected=" << i
                << " got=" << h_recv[i] << "\n";
      valid = false;
      break;
    }
  }

  if (valid) {
    std::cout << "[SERVER] recv completed correctly, first 32 bytes="
              << h_recv[0] << h_recv[1] << h_recv[2] << h_recv[3] << "\n";
  } else {
    std::cerr << "[SERVER] data mismatch detected!\n";
  }

  GPU_RT_CHECK(gpuFree(d_data));
}

void test_cq_poller() {
  auto comm0 =
      std::make_shared<Communicator>(0, 0, 2);  // gpu_id=0, local_rank=0
  auto comm1 =
      std::make_shared<Communicator>(0, 1, 2);  // gpu_id=0, local_rank=1

  std::thread t_client(cqpoller_client_thread, comm0, 1);
  std::thread t_server(cqpoller_server_thread, comm1, 0);

  t_client.join();
  t_server.join();

  std::cout << "[TEST] RDMA QP connection test finished" << std::endl;
}
