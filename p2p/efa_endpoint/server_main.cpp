// server_main.cpp
// Main function for epoll-based server
// Compile: g++ -std=c++17 server_main.cpp -o epoll_server -pthread

#include "epoll_server.h"

// ---------------------------
// Example user handler (function B)
// ---------------------------
void my_process_meta(const MetaInfo& meta, int client_fd) {
    // This runs in threadpool; do any heavier work here.
    std::cout << "Processing in thread: client_id=" << meta.client_id
              << " value=" << meta.value
              << " message=" << meta.message
              << " (fd=" << client_fd << ")\n";
    // Example: pretend to do work
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

// ---------------------------
// main: start server
// ---------------------------
int main(int argc, char* argv[]) {
    int port = 9000;  // Default port
    int thread_count = 8;  // Default thread pool size

    // Parse command line arguments
    if (argc > 1) {
        port = std::atoi(argv[1]);
        if (port <= 0 || port > 65535) {
            std::cerr << "Invalid port number. Using default: 9000\n";
            port = 9000;
        }
    }
    if (argc > 2) {
        thread_count = std::atoi(argv[2]);
        if (thread_count <= 0) {
            std::cerr << "Invalid thread count. Using default: 8\n";
            thread_count = 8;
        }
    }

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [port] [thread_count]\n";
        std::cout << "  port: Port to listen on (default: 9000)\n";
        std::cout << "  thread_count: Number of worker threads (default: 8)\n";
        return 0;
    }

    EpollServer server(port, my_process_meta, thread_count);

    if (!server.start()) {
        std::cerr << "Failed to start server on port " << port << "\n";
        return 1;
    }
    std::cout << "Server started on 0.0.0.0:" << port << " (epoll, non-blocking)\n";
    std::cout << "Thread pool size: " << thread_count << "\n";
    std::cout << "Press Ctrl+C to stop the server\n";

    // Block forever (or until interrupted)
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    server.stop();
    std::cout << "Server stopped\n";
    return 0;
}
