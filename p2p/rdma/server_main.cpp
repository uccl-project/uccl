#include <iostream>
#include <thread>
#include <chrono>
#include <string>

#include "epoll_server.h"

int main(int argc, char** argv) {
  int port = 9000;
  if (argc >= 2) {
    port = std::stoi(argv[1]);
  }

  std::cout << "[Server] Starting EpollServer on port " << port << "\n";

  EpollServer server(
      port,
      [](std::string const& input,
         std::string& output,
         std::string const& client_ip,
         int client_port) {
        std::cout << "[Server] Got request from "
                  << client_ip << ":" << client_port
                  << " payload=\"" << input << "\"\n";

        // 模拟 RDMA metadata 处理
        output = "ACK_FROM_SERVER: " + input;
      });

  if (!server.start()) {
    std::cerr << "[Server] Failed to start\n";
    return 1;
  }

  std::cout << "[Server] Listening on port " << server.get_port() << "\n";
  std::cout << "[Server] Press Ctrl+C to exit\n";

  // 阻塞主线程（server 在内部 worker_thread 里跑）
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  return 0;
}
