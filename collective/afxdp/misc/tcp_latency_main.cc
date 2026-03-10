#include "util.h"
#include "util/debug.h"
#include "util_tcp.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

int const PORT = 40001;
int const BUFFER_SIZE = 128 * 1024;  // 128KB
int const NUM_MESSAGES = 20000;
int const NUM_ITERATIONS = 1000;

void net_barrier(int sockfd) {
  bool sync = true;
  int ret = write(sockfd, &sync, sizeof(bool));
  ret = read(sockfd, &sync, sizeof(bool));
  UCCL_DCHECK(ret == sizeof(bool) && sync);
}

// Server function
void runServer() {
  int serverSocket, clientSocket;
  sockaddr_in serverAddr{}, clientAddr{};
  socklen_t clientLen = sizeof(clientAddr);

  serverSocket = socket(AF_INET, SOCK_STREAM, 0);
  if (serverSocket == -1) {
    UCCL_LOG(FATAL, UCCL_AFXDP)
        << "Socket creation failed: " << strerror(errno);
    exit(EXIT_FAILURE);
  }
  int flag = 1;
  if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int)) <
      0)
    UCCL_LOG(FATAL, UCCL_AFXDP) << "setsockopt(SO_REUSEADDR) failed";

  bzero((char*)&serverAddr, sizeof(serverAddr));
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_addr.s_addr = INADDR_ANY;
  serverAddr.sin_port = htons(PORT);

  if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) <
      0) {
    UCCL_LOG(FATAL, UCCL_AFXDP) << "Bind failed: " << strerror(errno);
    close(serverSocket);
    exit(EXIT_FAILURE);
  }

  if (listen(serverSocket, 5) < 0) {
    UCCL_LOG(FATAL, UCCL_AFXDP) << "Listen failed: " << strerror(errno);
    close(serverSocket);
    exit(EXIT_FAILURE);
  }

  UCCL_LOG(INFO, UCCL_AFXDP) << "Server listening on port " << PORT;
  char* buffer = (char*)malloc(BUFFER_SIZE);
  char const confirmation[] = "ACK";

  while (true) {
    clientSocket =
        accept(serverSocket, (struct sockaddr*)&clientAddr, &clientLen);
    if (clientSocket < 0) {
      UCCL_LOG(WARN, UCCL_AFXDP) << "Accept failed: " << strerror(errno);
      continue;
    }
    setsockopt(clientSocket, IPPROTO_TCP, TCP_NODELAY, (void*)&flag,
               sizeof(int));

    UCCL_LOG(INFO, UCCL_AFXDP)
        << "Connection accepted from " << inet_ntoa(clientAddr.sin_addr) << ":"
        << htons(clientAddr.sin_port);

    for (int j = 0; j < NUM_MESSAGES; ++j) {
      receive_message(BUFFER_SIZE, clientSocket, (uint8_t*)buffer, &quit);
      send_message(sizeof(confirmation), clientSocket, (uint8_t*)confirmation,
                   &quit);
    }

    net_barrier(clientSocket);
    close(clientSocket);
    UCCL_LOG(INFO, UCCL_AFXDP) << "Client disconnected.";
  }

  free(buffer);
  close(serverSocket);
}

// Client function
void runClient(std::string const& serverIP) {
  sockaddr_in serverAddr{};
  struct hostent* server;
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(PORT);
  if (inet_pton(AF_INET, serverIP.c_str(), &serverAddr.sin_addr) <= 0) {
    UCCL_LOG(FATAL, UCCL_AFXDP) << "Invalid server IP address: " << serverIP;
    exit(EXIT_FAILURE);
  }

  std::vector<double> medLatencies, tailLatencies;

  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    std::vector<double> latencies;

    char* message = (char*)malloc(BUFFER_SIZE);
    char response[4];

    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket < 0) {
      UCCL_LOG(FATAL, UCCL_AFXDP)
          << "Socket creation failed: " << strerror(errno);
      exit(EXIT_FAILURE);
    }

    if (connect(clientSocket, (struct sockaddr*)&serverAddr,
                sizeof(serverAddr)) < 0) {
      UCCL_LOG(FATAL, UCCL_AFXDP)
          << "Connection to server failed: " << strerror(errno);
      close(clientSocket);
      exit(EXIT_FAILURE);
    }

    int flag = 1;
    setsockopt(clientSocket, IPPROTO_TCP, TCP_NODELAY, (void*)&flag,
               sizeof(int));

    for (int j = 0; j < NUM_MESSAGES; ++j) {
      auto start = std::chrono::high_resolution_clock::now();

      int ret =
          send_message(BUFFER_SIZE, clientSocket, (uint8_t*)message, &quit);
      ret = receive_message(sizeof(response), clientSocket, (uint8_t*)response,
                            &quit);

      auto end = std::chrono::high_resolution_clock::now();
      latencies.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());
    }
    net_barrier(clientSocket);

    close(clientSocket);
    free(message);

    medLatencies.push_back(uccl::Percentile(latencies, 50));
    tailLatencies.push_back(uccl::Percentile(latencies, 99));

    UCCL_LOG(INFO, UCCL_AFXDP)
        << "Iteration " << i + 1 << ": Median Latency = " << medLatencies.back()
        << " µs, Tail Latency = " << tailLatencies.back() << " µs";
  }

  UCCL_LOG(INFO, UCCL_AFXDP)
      << "Completed " << NUM_ITERATIONS << " iterations.";
  UCCL_LOG(INFO, UCCL_AFXDP)
      << "Median of median Latencies: " << uccl::Percentile(medLatencies, 50)
      << " µs";
  UCCL_LOG(INFO, UCCL_AFXDP)
      << "Tail of median Latencies: " << uccl::Percentile(medLatencies, 99)
      << " µs";
  UCCL_LOG(INFO, UCCL_AFXDP)
      << "Median of tail Latencies: " << uccl::Percentile(tailLatencies, 50)
      << " µs";
  UCCL_LOG(INFO, UCCL_AFXDP)
      << "Tail of tail Latencies: " << uccl::Percentile(tailLatencies, 99)
      << " µs";
}

DEFINE_string(role, "server", "server or client.");
DEFINE_string(serverip, "127.0.0.1",
              "Server IP address the client tries to connect.");

/**
 * Usage:
 *   ./tcp_latency_main --logtostderr=1 --role=server
 *   ./tcp_latency_main --logtostderr=1 --role=client --serverip=192.168.6.1
 */

int main(int argc, char* argv[]) {
  // google::InitGoogleLogging(argv[0]);
  // google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_role == "server") {
    runServer();
  } else if (FLAGS_role == "client") {
    std::string serverIP = FLAGS_serverip;
    runClient(serverIP);
  } else {
    UCCL_LOG(FATAL, UCCL_AFXDP)
        << "Unknown role: " << FLAGS_role << ". Use 'server' or 'client'.";
    return EXIT_FAILURE;
  }

  return 0;
}
