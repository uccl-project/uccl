#include "tcpx_plugin_api.h"
#include <iostream>
#include <vector>
#include <cstring>  // for memset
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
  using namespace tcpx_plugin;

  InitOptions opts;
  Status st = init(opts);
  if (st != Status::kOk) {
    std::cerr << "init error\n";
    return -1;
  }

  Session server_sess(
      /*gpu_id=*/0,
      /*num_channels=*/1,
      /*bootstrap_info=*/"server_cfg_string",
      /*nic_device_id=*/0
  );

  Session client_sess(
      /*gpu_id=*/0,
      /*num_channels=*/1,
      /*bootstrap_info=*/"client_cfg_string",
      /*nic_device_id=*/0
  );

  std::string srv_json = server_sess.listen_json();

  st = client_sess.load_remote_json("srv", srv_json);
  if (st != Status::kOk) {
    std::cerr << "load_remote_json error\n";
    return -1;
  }

  st = server_sess.accept("cli");
  if (st != Status::kOk) {
    std::cerr << "server accept error\n";
    return -1;
  }
  st = client_sess.connect("srv");
  if (st != Status::kOk) {
    std::cerr << "client connect error\n";
    return -1;
  }

  constexpr size_t SIZE = 4096;
  void* buf = nullptr;
  cudaMalloc(&buf, SIZE);
  cudaMemset(buf, 0, SIZE);

  uint64_t mem_id_s = server_sess.register_memory(buf, SIZE, /*is_recv=*/false);
  uint64_t mem_id_c = client_sess.register_memory(buf, SIZE, /*is_recv=*/true);

  if (!mem_id_s || !mem_id_c) {
    std::cerr << "register_memory failed\n";
    return -1;
  }

  ConnID ci; 
  ci.remote = "srv"; // client->server
  Transfer* client_xfer = client_sess.create_transfer(ci);
  ConnID si;
  si.remote = "cli"; // server->client
  Transfer* server_xfer = server_sess.create_transfer(si);

  if (!client_xfer || !server_xfer) {
    std::cerr << "create_transfer error\n";
    return -1;
  }

  st = client_xfer->post_send(mem_id_c, 0, SIZE, /*tag=*/0);
  if (st != Status::kOk) {
    std::cerr << "client post_send error\n";
    return -1;
  }

  st = server_xfer->post_recv(mem_id_s, 0, SIZE, /*tag=*/0);
  if (st != Status::kOk) {
    std::cerr << "server post_recv error\n";
    return -1;
  }

  st = client_xfer->wait();
  if (st != Status::kOk) {
    std::cerr << "client wait error\n";
    return -1;
  }

  st = server_xfer->wait();
  if (st != Status::kOk) {
    std::cerr << "server wait error\n";
    return -1;
  }

  client_xfer->release();
  server_xfer->release();
  delete client_xfer;
  delete server_xfer;

  server_sess.deregister_memory(mem_id_s);
  client_sess.deregister_memory(mem_id_c);

  cudaFree(buf);

  std::cout << "Done.\n";
  return 0;
}
