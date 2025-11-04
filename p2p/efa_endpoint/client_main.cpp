// client_main.cpp
// Main function for epoll-based client with connection reuse
// Compile: g++ -std=c++17 client_main.cpp -o epoll_client -pthread

#include "epoll_client.h"
#include <atomic>

// ---------------------------
// Test client with connection reuse
// ---------------------------
int main(int argc, char* argv[]) {
    std::string server_ip = "127.0.0.1";  // Default server IP
    int server_port = 9000;               // Default port
    int num_messages = 10;                // Default number of messages
    int client_id_start = 0;              // Default starting client ID

    // Parse command line arguments
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [server_ip] [port] [num_messages] [client_id_start]\n";
        std::cout << "  server_ip: IP address of the server (default: 127.0.0.1)\n";
        std::cout << "  port: Port of the server (default: 9000)\n";
        std::cout << "  num_messages: Number of messages to send (default: 10)\n";
        std::cout << "  client_id_start: Starting client ID (default: 0)\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " 192.168.1.100 9000 20 0\n";
        return 0;
    }

    if (argc > 1) {
        server_ip = argv[1];
    }
    if (argc > 2) {
        server_port = std::atoi(argv[2]);
        if (server_port <= 0 || server_port > 65535) {
            std::cerr << "Invalid port number. Using default: 9000\n";
            server_port = 9000;
        }
    }
    if (argc > 3) {
        num_messages = std::atoi(argv[3]);
        if (num_messages <= 0) {
            std::cerr << "Invalid number of messages. Using default: 10\n";
            num_messages = 10;
        }
    }
    if (argc > 4) {
        client_id_start = std::atoi(argv[4]);
        if (client_id_start < 0) {
            std::cerr << "Invalid client ID start. Using default: 0\n";
            client_id_start = 0;
        }
    }

    EpollClient client;
    if (!client.start()) {
        std::cerr << "Failed to start client\n";
        return 1;
    }

    std::cout << "Client started, connecting to " << server_ip << ":" << server_port << "\n";

    // Connect once
    std::string conn_key = client.connect_to_server(server_ip, server_port);
    if (conn_key.empty()) {
        std::cerr << "Failed to connect to server at " << server_ip << ":" << server_port << "\n";
        std::cerr << "Please check:\n";
        std::cerr << "  1. Server is running\n";
        std::cerr << "  2. IP address and port are correct\n";
        std::cerr << "  3. No firewall blocking the connection\n";
        return 1;
    }

    std::cout << "Connected successfully, sending " << num_messages
              << " messages on the same connection...\n";

    // Send multiple messages on the same connection (connection reuse)
    std::atomic<int> responses_received{0};

    for (int i = 0; i < num_messages; ++i) {
        MetaInfo meta{};
        meta.client_id = client_id_start + i;
        meta.value = 3.14 + i;
        snprintf(meta.message, sizeof(meta.message), "Message %d from client (id=%d)",
                 i, meta.client_id);

        // Send with callback
        int msg_id = i;
        bool sent = client.send_meta(conn_key, meta, [msg_id, &responses_received](const std::string& response) {
            std::cout << "Message " << msg_id << " received response: " << response;
            responses_received++;
        });

        if (!sent) {
            std::cerr << "Failed to send message " << i << " (client_id=" << meta.client_id << ")\n";
        } else {
            std::cout << "Sent message " << i << " (client_id=" << meta.client_id << ")\n";
        }

        // Small delay between sends
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Wait for all responses
    std::cout << "Waiting for responses...\n";
    int wait_count = 0;
    while (responses_received < num_messages && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    std::cout << "Received " << responses_received << " / " << num_messages << " responses\n";
    std::cout << "Active connections: " << client.get_connection_count() << "\n";

    // Demonstrate connecting to multiple servers (if you had more servers)
    // You can call connect_to_server with different IPs/ports and reuse those connections too

    std::cout << "\nTest completed. Stopping client...\n";
    client.stop();

    return 0;
}
