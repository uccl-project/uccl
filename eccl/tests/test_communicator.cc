#include "test.h"
#include "transport.h"
#include <iostream>
#include <thread>

void communicator_client_thread() {
  {
    auto comm = std::make_shared<Communicator>(
        0, 0, 2,
        std::make_shared<Config>());  // gpu_id=0, local_rank=0, world_size=2
    std::cout << "[CLIENT] Communicator for rank 0 created." << std::endl;

    int peer_rank = 1;
    if (comm->connect_to(peer_rank)) {
      std::cout << "[CLIENT] Successfully connected to rank " << peer_rank
                << std::endl;
    } else {
      std::cerr << "[CLIENT] Failed to connect to rank " << peer_rank
                << std::endl;
    }

    comm.reset();
    std::cout << "[CLIENT] Communicator destroyed and resources released."
              << std::endl;
  }
}

void communicator_server_thread() {
  {
    auto comm = std::make_shared<Communicator>(
        0, 1, 2,
        std::make_shared<Config>());  // gpu_id=0, local_rank=1, world_size=2
    std::cout << "[SERVER] Communicator for rank 1 created." << std::endl;

    int peer_rank = 0;
    if (comm->accept_from(peer_rank)) {
      std::cout << "[SERVER] Successfully accepted connection from rank "
                << peer_rank << std::endl;
    } else {
      std::cerr << "[SERVER] Failed to accept connection from rank "
                << peer_rank << std::endl;
    }

    comm.reset();
    std::cout << "[SERVER] Communicator destroyed and resources released."
              << std::endl;
  }
}

void test_communicator() {
  std::thread t_client(communicator_client_thread);
  std::thread t_server(communicator_server_thread);

  t_client.join();
  t_server.join();

  std::cout
      << "[TEST] RDMA QP connection test finished, all threads exited cleanly."
      << std::endl;
}
