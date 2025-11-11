#include "rdma_device.h"
#include "efa_endpoint.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <thread>
#include <chrono>
// ./efa_endpoint_example 0 1 19999 0 10.1.82.97 19999
// 
// ./efa_endpoint_example 0 0 19999 1 10.1.219.155 19999  这个
void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " <gpu_index> <rank_id> <port> <remote_rank> <remote_ip> <remote_port>\n"
              << "\n"
              << "Parameters:\n"
              << "  gpu_index    : GPU index to use\n"
              << "  rank_id      : Local rank ID\n"
              << "  port         : Local port for OOB server\n"
              << "  remote_rank  : Remote rank ID to connect to\n"
              << "  remote_ip    : Remote IP address\n"
              << "  remote_port  : Remote port number\n"
              << "\n"
              << "Example:\n"
              << "  " << program_name << " 0 0 8888 1 192.168.1.100 8889\n";
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 7) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse command line arguments
    int gpu_index = std::atoi(argv[1]);
    uint64_t rank_id = std::stoull(argv[2]);
    uint64_t port = std::stoull(argv[3]);
    uint64_t remote_rank = std::stoull(argv[4]);
    std::string remote_ip = argv[5];
    uint64_t remote_port = std::stoull(argv[6]);

    std::cout << "=== EFAEndpoint Usage Example ===\n";
    std::cout << "GPU Index: " << gpu_index << "\n";
    std::cout << "Rank ID: " << rank_id << "\n";
    std::cout << "Port: " << port << "\n";
    std::cout << "Remote Rank: " << remote_rank << "\n";
    std::cout << "Remote IP: " << remote_ip << "\n";
    std::cout << "Remote Port: " << remote_port << "\n";
    std::cout << "================================\n\n";

    try {
        // Initialize RDMA device manager
        std::cout << "Initializing RDMA device manager...\n";
        auto& device_manager = RdmaDeviceManager::instance();
        std::cout << "Found " << device_manager.deviceCount() << " RDMA device(s)\n\n";

        // Create EFAEndpoint with device_ids = {0}
        std::cout << "Creating EFAEndpoint...\n";
        std::vector<size_t> device_ids = {0,1,2};
        EFAEndpoint endpoint(gpu_index, rank_id, port, device_ids);
        std::cout << "EFAEndpoint created successfully\n\n";

        // Create OOBMetaData for remote rank
        std::cout << "Setting up remote rank metadata...\n";
        auto remote_meta = std::make_shared<OOBMetaData>();
        remote_meta->server_ip = remote_ip;
        remote_meta->server_port = remote_port;

        // Add remote rank metadata
        std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> rank_meta_map;
        rank_meta_map[remote_rank] = remote_meta;
        endpoint.add_rank_oob_meta(rank_meta_map);
        std::cout << "Added remote rank " << remote_rank
                  << " metadata (IP: " << remote_ip
                  << ", Port: " << remote_port << ")\n\n";

        // Wait a bit to ensure both endpoints are ready
        std::cout << "Waiting for 2 seconds to ensure both endpoints are ready...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Connect to remote rank
        std::cout << "Connecting to remote rank " << remote_rank << "...\n";
        bool connect_result = endpoint.build_connect(remote_rank);

        if (connect_result) {
            std::cout << "Successfully connected to remote rank " << remote_rank << "\n";
        } else {
            std::cerr << "Failed to connect to remote rank " << remote_rank << "\n";
            return 1;
        }

        std::cout << "\n=== Connection established successfully ===\n";
        std::cout << "Press Ctrl+C to exit...\n";

        // Keep the program running to maintain the connection
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}