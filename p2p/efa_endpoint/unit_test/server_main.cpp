// server_main.cpp
// Main function for epoll-based server
// Compile: g++ -std=c++17 server_main.cpp -o epoll_server -pthread

#include "define.h"
#include "epoll_server.h"

// ---------------------------
// Example processor class with member function handler
// ---------------------------
class MetaProcessor {
 public:
  MetaProcessor(std::string const& name) : name_(name), count_(0) {}

  // Member function that will be used as callback
  void process_meta(std::string const& input, std::string& output) {
    // Deserialize input to MetaInfoToExchange
    MetaInfoToExchange meta = deserialize<MetaInfoToExchange>(input);

    count_++;
    std::cout << "[" << name_ << "] Processing #" << count_
              << ": rank_id=" << meta.rank_id
              << " context_id=" << meta.context_id
              << " qpn=" << meta.channel_meta.qpn << " mem_addr=0x" << std::hex
              << meta.mem_meta.addr << std::dec
              << " rkey=" << meta.mem_meta.rkey << "\n";

    // Example: pretend to do work
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Create response (echo back the same data)
    MetaInfoToExchange response = meta;
    output = serialize(response);
  }

  int get_count() const { return count_; }

 private:
  std::string name_;
  std::atomic<int> count_;
};

// ---------------------------
// Example standalone handler function (alternative approach)
// ---------------------------
void my_process_meta(std::string const& input, std::string& output) {
  // Deserialize input to MetaInfoToExchange
  MetaInfoToExchange meta = deserialize<MetaInfoToExchange>(input);

  std::cout << "Processing: rank_id=" << meta.rank_id
            << " context_id=" << meta.context_id
            << " qpn=" << meta.channel_meta.qpn << " mem_addr=0x" << std::hex
            << meta.mem_meta.addr << std::dec << " rkey=" << meta.mem_meta.rkey
            << "\n";

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Create response (echo back the same data)
  MetaInfoToExchange response = meta;
  output = serialize(response);
}

// ---------------------------
// main: start server
// ---------------------------
int main(int argc, char* argv[]) {
  int port = 0;                 // Default: 0 = auto-assign port
  bool use_member_func = true;  // Use member function callback by default

  // Parse command line arguments
  if (argc > 1 &&
      (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    std::cout << "Usage: " << argv[0] << " [port] [callback_type]\n";
    std::cout << "  port: Port to listen on (default: 0 = auto-assign)\n";
    std::cout << "  callback_type: 'member' or 'function' (default: member)\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << argv[0]
              << " 0 member    # Auto-assign port, use member function\n";
    std::cout << "  " << argv[0] << " 9000        # Use port 9000\n";
    return 0;
  }

  if (argc > 1) {
    port = std::atoi(argv[1]);
    if (port < 0 || port > 65535) {
      std::cerr << "Invalid port number. Using default: 0 (auto-assign)\n";
      port = 0;
    }
  }
  if (argc > 2) {
    std::string callback_type = argv[2];
    if (callback_type == "function") {
      use_member_func = false;
    }
  }

  // Create server with appropriate callback
  if (use_member_func) {
    // Approach 1: Use object member function as callback
    MetaProcessor processor("MyProcessor");

    // Create server with lambda that captures processor and calls member
    // function
    EpollServer server(
        port, [&processor](std::string const& input, std::string& output) {
          processor.process_meta(input, output);
        });

    if (!server.start()) {
      std::cerr << "Failed to start server\n";
      return 1;
    }

    int actual_port = server.get_port();
    std::cout << "Server started on 0.0.0.0:" << actual_port
              << " (epoll, non-blocking)\n";
    std::cout << "Processing mode: Synchronous (blocking callback)\n";
    std::cout
        << "Callback type: Member function (MetaProcessor::process_meta)\n";
    std::cout << "Press Ctrl+C to stop the server\n";

    // Block forever (or until interrupted)
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    server.stop();
    std::cout << "Server stopped. Total requests processed: "
              << processor.get_count() << "\n";
  } else {
    // Approach 2: Use standalone function as callback
    EpollServer server(port, my_process_meta);

    if (!server.start()) {
      std::cerr << "Failed to start server\n";
      return 1;
    }

    int actual_port = server.get_port();
    std::cout << "Server started on 0.0.0.0:" << actual_port
              << " (epoll, non-blocking)\n";
    std::cout << "Processing mode: Synchronous (blocking callback)\n";
    std::cout << "Callback type: Standalone function (my_process_meta)\n";
    std::cout << "Press Ctrl+C to stop the server\n";

    // Block forever (or until interrupted)
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    server.stop();
    std::cout << "Server stopped\n";
  }

  return 0;
}
