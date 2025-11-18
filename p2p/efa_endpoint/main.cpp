#include "rdma_device.h"
#include "efa_endpoint.h"
#include "memory_allocator.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
// ./efa_endpoint_example 0 1 19997 0 10.1.82.97 19997
// 
// ./efa_endpoint_example 0 0 19997 1 10.1.219.155 19997  这个
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

    //     size_t gpu_size = 1024 * 1024;
    // auto gpu_mem =
    //     allocator.allocate(gpu_size, MemoryType::HOST, contexts_[0]);
    // std::cout << "Allocated " << gpu_mem->size << " bytes of GPU memory at "
    //           << gpu_mem->addr << std::endl;
    // RemoteMemInfo info(gpu_mem);
    // recv_test_ = std::make_shared<EFARecvRequest>(gpu_mem);
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

        std::this_thread::sleep_for(std::chrono::seconds(5));

        // Test send/recv functionality
        std::cout << "\n=== Testing send/recv functionality ===\n";

        // Allocate memory for send and recv
        MemoryAllocator allocator;
        size_t buffer_size = 1024 * 1024;  // 1MB
        auto send_mem = allocator.allocate(buffer_size, MemoryType::HOST, endpoint.getContext(0));
        auto recv_mem = allocator.allocate(buffer_size, MemoryType::HOST, endpoint.getContext(0));

        std::cout << "Allocated " << send_mem->size << " bytes for send buffer at " << send_mem->addr << "\n";
        std::cout << "Allocated " << recv_mem->size << " bytes for recv buffer at " << recv_mem->addr << "\n";

        // Prepare test data
        std::string test_message = "Hello from rank " + std::to_string(rank_id) + "!";
        if (send_mem->type == MemoryType::GPU) {
            char* h_data = (char*)malloc(buffer_size);
            strcpy(h_data, test_message.c_str());
            cudaMemcpy(send_mem->addr, h_data, buffer_size, cudaMemcpyHostToDevice);
            free(h_data);
        } else {
            strcpy((char*)send_mem->addr, test_message.c_str());
        }

        std::cout << "Prepared send data: \"" << test_message << "\"\n";

        // Create send and recv requests
        // For send request, create a placeholder remote_mem (will be filled by control channel)
        auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
        auto send_req = std::make_shared<EFASendRequest>(send_mem, remote_mem_placeholder);
        auto recv_req = std::make_shared<EFARecvRequest>(recv_mem);

        uint32_t channel_id = 0;  // Use channel 0 for testing

        // Test based on rank_id
        if (rank_id == 0) {
            // Rank 0: First recv, then send
            std::cout << "\n[Rank 0] Posting recv request...\n";
            int64_t recv_index = endpoint.recv(remote_rank, channel_id, recv_req);
            std::cout << "[Rank 0] Recv posted with index: " << recv_index << "\n";

            std::cout << "[Rank 0] Checking recv completion...\n";
            endpoint.checkRecvComplete(remote_rank, recv_index);
            std::cout << "[Rank 0] Recv completed!\n";

            // Read received data
            char* h_data = (char*)malloc(buffer_size);
            if (recv_mem->type == MemoryType::GPU) {
                cudaMemcpy(h_data, recv_mem->addr, buffer_size, cudaMemcpyDeviceToHost);
            } else {
                memcpy(h_data, recv_mem->addr, buffer_size);
            }
            std::cout << "[Rank 0] Received message: \"" << h_data << "\"\n";
            free(h_data);

            std::this_thread::sleep_for(std::chrono::seconds(1));

            std::cout << "\n[Rank 0] Posting send request...\n";
            int64_t send_wr_id = endpoint.send(remote_rank, channel_id, send_req);
            std::cout << "[Rank 0] Send posted with wr_id: " << send_wr_id << "\n";

            std::cout << "[Rank 0] Checking send completion...\n";
            endpoint.checkSendComplete(remote_rank, channel_id, send_wr_id);
            std::cout << "[Rank 0] Send completed!\n";

        } else {
            // Rank 1: First send, then recv
            std::cout << "\n[Rank 1] Posting send request...\n";
            int64_t send_wr_id = endpoint.send(remote_rank, channel_id, send_req);
            std::cout << "[Rank 1] Send posted with wr_id: " << send_wr_id << "\n";

            std::cout << "[Rank 1] Checking send completion...\n";
            endpoint.checkSendComplete(remote_rank, channel_id, send_wr_id);
            std::cout << "[Rank 1] Send completed!\n";

            std::this_thread::sleep_for(std::chrono::seconds(1));

            std::cout << "\n[Rank 1] Posting recv request...\n";
            int64_t recv_index = endpoint.recv(remote_rank, channel_id, recv_req);
            std::cout << "[Rank 1] Recv posted with index: " << recv_index << "\n";

            std::cout << "[Rank 1] Checking recv completion...\n";
            endpoint.checkRecvComplete(remote_rank, recv_index);
            std::cout << "[Rank 1] Recv completed!\n";

            // Read received data
            char* h_data = (char*)malloc(buffer_size);
            if (recv_mem->type == MemoryType::GPU) {
                cudaMemcpy(h_data, recv_mem->addr, buffer_size, cudaMemcpyDeviceToHost);
            } else {
                memcpy(h_data, recv_mem->addr, buffer_size);
            }
            std::cout << "[Rank 1] Received message: \"" << h_data << "\"\n";
            free(h_data);
        }

        std::cout << "\n=== Send/Recv test completed successfully ===\n";
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