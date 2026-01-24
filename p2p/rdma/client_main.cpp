#include <iostream>
#include <thread>
#include <chrono>
#include <string>

#include "epoll_client.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <server_ip> <server_port>\n";
    return 1;
  }

  std::string server_ip = argv[1];
  int server_port = std::stoi(argv[2]);

  EpollClient client;
  if (!client.start()) {
    std::cerr << "[Client] Failed to start EpollClient\n";
    return 1;
  }

  std::cout << "[Client] Connecting to "
            << server_ip << ":" << server_port << "\n";

  std::string conn_key =
      client.connect_to_server(server_ip, server_port);

  if (conn_key.empty()) {
    std::cerr << "[Client] Connection failed\n";
    return 1;
  }

  std::string msg = "hello from remote client";
  std::cout << "[Client] Sending payload: \"" << msg << "\"\n";

  client.send_meta(
      conn_key,
      msg,
      [](std::string const& response) {
        std::cout << "[Client] Got response: \"" << response << "\"\n";
      });

  // 等待 server 回包
  std::this_thread::sleep_for(std::chrono::seconds(5));

  std::cout << "[Client] Done\n";
  return 0;
}
