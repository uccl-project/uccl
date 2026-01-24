#include <iostream>
#include <thread>
#include <chrono>
#include <string>

#include "epoll_client.h"   // 假设你把上面那坨代码放在这个头文件里

int main() {
  // -------------------------
  // 1. 创建 server
  // -------------------------
  EpollServer server(
      0,  // port = 0 -> auto assign
      [](std::string const& input,
         std::string& output,
         std::string const& client_ip,
         int client_port) {
        std::cout << "[Server] Got request from "
                  << client_ip << ":" << client_port
                  << " payload=\"" << input << "\"\n";

        // 简单 echo + tag
        output = "ACK: " + input;
      });

  if (!server.start()) {
    std::cerr << "Failed to start EpollServer\n";
    return 1;
  }

  int port = server.get_port();
  std::cout << "[Main] Server listening on port " << port << "\n";

  // -------------------------
  // 2. 创建 client
  // -------------------------
  EpollClient client;
  if (!client.start()) {
    std::cerr << "Failed to start EpollClient\n";
    return 1;
  }

  // 给 server 一点时间起来（真实项目用 condition / barrier）
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // -------------------------
  // 3. 连接 server
  // -------------------------
  std::string conn_key = client.connect_to_server("127.0.0.1", port);
  if (conn_key.empty()) {
    std::cerr << "Client connect failed\n";
    return 1;
  }

  // -------------------------
  // 4. 发送一条消息
  // -------------------------
  std::string msg = "hello epoll meta";
  std::cout << "[Client] Sending: \"" << msg << "\"\n";

  client.send_meta(
      conn_key,
      msg,
      [](std::string const& response) {
        std::cout << "[Client] Got response: \"" << response << "\"\n";
      });

  // -------------------------
  // 5. 等待一会儿看回包
  // -------------------------
  std::this_thread::sleep_for(std::chrono::seconds(2));

  std::cout << "[Main] Test finished, shutting down\n";
  return 0;
}
