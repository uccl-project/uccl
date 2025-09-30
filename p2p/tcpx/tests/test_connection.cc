#include "../tcpx_interface.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/mman.h>
#include <vector>
#ifndef NO_CUDA
#include <cuda.h>
#endif

// NCCL network handle - used to exchange connection details
// According to the NCCL spec, the handle is typically 128 bytes (extra space
// avoids overflow)
#define NCCL_NET_HANDLE_MAXSIZE 128
struct ncclNetHandle_v7 {
  char data[NCCL_NET_HANDLE_MAXSIZE];
};

// TCP port for handle exchange (similar to RDMA's bootstrap)
#define TCPX_BOOTSTRAP_PORT 12345

// Server: create bootstrap socket and wait for client to connect
int create_bootstrap_server() {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "Failed to create bootstrap socket" << std::endl;
    return -1;
  }

  int opt = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(TCPX_BOOTSTRAP_PORT);

  if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "Failed to bind bootstrap socket" << std::endl;
    close(listen_fd);
    return -1;
  }

  if (listen(listen_fd, 1) < 0) {
    std::cerr << "Failed to listen on bootstrap socket" << std::endl;
    close(listen_fd);
    return -1;
  }

  std::cout << "Bootstrap server listening on port " << TCPX_BOOTSTRAP_PORT
            << std::endl;

  int client_fd = accept(listen_fd, nullptr, nullptr);
  close(listen_fd);

  if (client_fd < 0) {
    std::cerr << "Failed to accept bootstrap connection" << std::endl;
    return -1;
  }

  return client_fd;
}

// Client: connect to server's bootstrap socket
int connect_to_bootstrap_server(char const* server_ip) {
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    std::cerr << "Failed to create bootstrap socket" << std::endl;
    return -1;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(TCPX_BOOTSTRAP_PORT);
  inet_aton(server_ip, &addr.sin_addr);

  // Retry connection with backoff
  int retry = 0;
  while (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    if (++retry > 10) {
      std::cerr << "Failed to connect to bootstrap server after " << retry
                << " retries" << std::endl;
      close(sock_fd);
      return -1;
    }
    std::cout << "Retrying bootstrap connection... (" << retry << "/10)"
              << std::endl;
    sleep(1);
  }

  std::cout << "Connected to bootstrap server at " << server_ip << std::endl;
  return sock_fd;
}

int main(int argc, char* argv[]) {
  std::cout << "=== TCPX Connection Test ===" << std::endl;
  std::cout << "Note: this is a simplified connectivity test" << std::endl;
  std::cout << "A real TCPX setup must exchange handles via out-of-band "
               "communication"
            << std::endl;
  std::cout << "This run only verifies that the API calls behave as expected"
            << std::endl;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <server|client> [remote_ip]"
              << std::endl;
    std::cout << "  server: Start as server (listener)" << std::endl;
    std::cout << "  client <ip>: Connect to server at <ip>" << std::endl;
    return 1;
  }

  // Enable debug output
  setenv("UCCL_TCPX_DEBUG", "1", 1);
  // Respect external env for RX memory import; do not override here.

  // Initialize TCPX
  std::cout << "\n[Step 1] Initializing TCPX..." << std::endl;
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "�?FAILED: No TCPX devices found" << std::endl;
    return 1;
  }
  std::cout << "�?SUCCESS: Found " << device_count << " TCPX devices"
            << std::endl;

  // Use device 0 for testing
  int dev_id = 0;
  std::cout << "Using TCPX device " << dev_id << std::endl;

  bool is_server = (strcmp(argv[1], "server") == 0);

  if (is_server) {
    std::cout << "\n[Step 2] Starting as SERVER..." << std::endl;

    // Create connection handle for listening
    ncclNetHandle_v7 handle;
    memset(&handle, 0, sizeof(handle));

    void* listen_comm = nullptr;
    std::cout << "Attempting to listen on device " << dev_id << "..."
              << std::endl;

    int rc = tcpx_listen(dev_id, &handle, &listen_comm);
    if (rc != 0) {
      std::cout << "�?FAILED: tcpx_listen returned " << rc << std::endl;
      return 1;
    }
    std::cout << "�?SUCCESS: Listening on device " << dev_id << std::endl;
    std::cout << "Listen comm: " << listen_comm << std::endl;

    // Keep TCPX handle opaque - don't parse or modify it
    std::cout << "\n[Step 3] TCPX handle ready for transmission..."
              << std::endl;

    // Print handle data for debugging only
    std::cout << "TCPX handle data (first 32 bytes):" << std::endl;
    for (int i = 0; i < 32; i++) {
      printf("%02x ", (unsigned char)handle.data[i]);
      if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // Don't modify the handle - pass it as-is to the client

    // Create bootstrap server to send handle to client
    std::cout << "Creating bootstrap server for handle exchange..."
              << std::endl;
    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      std::cout << "�?FAILED: Cannot create bootstrap server" << std::endl;
      return 1;
    }

    // Send handle to client via bootstrap connection
    std::cout << "Sending TCPX handle to client..." << std::endl;

    // Small delay to ensure TCPX is fully ready
    usleep(100000);  // 100ms

    // Loop to guarantee full 128B handle is sent
    size_t total_sent = 0;
    while (total_sent < NCCL_NET_HANDLE_MAXSIZE) {
      ssize_t s = send(bootstrap_fd, handle.data + total_sent,
                       NCCL_NET_HANDLE_MAXSIZE - total_sent, 0);
      if (s <= 0) {
        std::cout << "�?FAILED: Cannot send handle to client (sent " << total_sent
                  << " bytes)" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_sent += static_cast<size_t>(s);
    }
    std::cout << "�?SUCCESS: Handle sent to client (" << total_sent << " bytes)"
              << std::endl;

    close(bootstrap_fd);

    // Wait for client to process handle and initiate connection
    std::cout << "Waiting for client to connect..." << std::endl;
    sleep(3);  // Give client more time to process handle and connect

    // Accept connection with retry loop
    void* recv_comm = nullptr;
    // TCPX plugin expects caller-provided storage for devNetDeviceHandle
    // (see tcpxAccept_v5 -> tcpxGetDeviceHandle). Pre-allocate a buffer.
    alignas(16) unsigned char recv_dev_handle_storage[512] = {0};
    void* recv_dev_handle = recv_dev_handle_storage;

    std::cout << "Calling tcpx_accept_v5..." << std::endl;

    // Retry accept until we get a valid connection
    int accept_retries = 0;
    int const max_accept_retries = 10;

    while (accept_retries < max_accept_retries) {
      rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
      if (rc != 0) {
        std::cout << "�?FAILED: tcpx_accept_v5 returned " << rc << std::endl;
        tcpx_close_listen(listen_comm);
        return 1;
      }

      if (recv_comm != nullptr) {
        std::cout << "�?SUCCESS: Connection accepted!" << std::endl;
        std::cout << "Recv comm: " << recv_comm << std::endl;
        std::cout << "Recv dev handle: " << recv_dev_handle << std::endl;
        break;
      }

      accept_retries++;
      std::cout << "Accept returned null comm, retrying... (" << accept_retries
                << "/" << max_accept_retries << ")" << std::endl;
      sleep(1);
    }

    if (recv_comm == nullptr) {
      std::cout << "�?FAILED: No valid connection after " << max_accept_retries
                << " retries" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }

    std::cout << "\nConnection established successfully. Data path validation is covered by"
              << " tests/test_tcpx_transfer." << std::endl;
    if (recv_comm) tcpx_close_recv(recv_comm);
    if (listen_comm) tcpx_close_listen(listen_comm);

  } else if (strcmp(argv[1], "client") == 0) {
    if (argc < 3) {
      std::cout << "�?ERROR: Client mode requires remote IP" << std::endl;
      return 1;
    }

    std::cout << "\n[Step 2] Starting as CLIENT..." << std::endl;
    std::cout << "Connecting to server at " << argv[2] << std::endl;

    // Connect to server's bootstrap socket to receive handle
    std::cout << "\n[Step 3] Connecting to server for handle exchange..."
              << std::endl;
    int bootstrap_fd = connect_to_bootstrap_server(argv[2]);
    if (bootstrap_fd < 0) {
      std::cout << "�?FAILED: Cannot connect to bootstrap server" << std::endl;
      return 1;
    }

    // Receive handle from server via bootstrap connection
    std::cout << "Receiving TCPX handle from server..." << std::endl;
    ncclNetHandle_v7 handle;
    size_t total_received = 0;
    while (total_received < NCCL_NET_HANDLE_MAXSIZE) {
      ssize_t r = recv(bootstrap_fd, handle.data + total_received,
                       NCCL_NET_HANDLE_MAXSIZE - total_received, 0);
      if (r <= 0) {
        std::cout << "�?FAILED: Cannot receive handle from server (received "
                  << total_received << " bytes)" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_received += static_cast<size_t>(r);
    }
    std::cout << "�?SUCCESS: Handle received from server (" << total_received
              << " bytes)" << std::endl;

    // Debug: Print received handle data
    std::cout << "Received TCPX handle data (first 32 bytes):" << std::endl;
    for (int i = 0; i < 32; i++) {
      printf("%02x ", (unsigned char)handle.data[i]);
      if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    close(bootstrap_fd);

    // Use TCPX handle as-is (opaque data)
    std::cout << "Using TCPX handle for connection (opaque data)..."
              << std::endl;

    // Print handle for debugging
    std::cout << "Received handle (first 32 bytes):" << std::endl;
    for (int i = 0; i < 32; i++) {
      printf("%02x ", (unsigned char)handle.data[i]);
      if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // Small delay to ensure server is ready for accept
    std::cout << "Preparing to connect..." << std::endl;
    sleep(2);

    void* send_comm = nullptr;
    void* send_dev_handle = nullptr;

    std::cout << "Attempting TCPX connection..." << std::endl;
    int rc = tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle);
    if (rc != 0) {
      std::cout << "�?FAILED: tcpx_connect_v5 returned " << rc << std::endl;
      return 1;
    }

    std::cout << "�?SUCCESS: Connected to server!" << std::endl;
    std::cout << "Send comm: " << send_comm << std::endl;
    std::cout << "Send dev handle: " << send_dev_handle << std::endl;

    // Wait a bit to ensure connection is fully established
    std::cout << "Waiting for connection to stabilize..." << std::endl;
    sleep(3);  // Give more time for server to be ready for receive

    std::cout << "Connection established successfully. Data transfer test is"
              << " available in tests/test_tcpx_transfer." << std::endl;
    if (send_comm) tcpx_close_send(send_comm);

  } else {
    std::cout << "�?ERROR: Invalid mode. Use 'server' or 'client'" << std::endl;
    return 1;
  }

  std::cout << "\n=== TCPX Connection Test COMPLETED ===" << std::endl;
  return 0;
}
