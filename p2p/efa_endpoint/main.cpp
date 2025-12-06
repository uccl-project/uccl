#include "efa_endpoint.h"
#include "memory_allocator.h"
#include "rdma_device.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

// Define command line flags
DEFINE_int32(gpu_index, 0, "GPU index to use");
DEFINE_uint64(rank_id, 0, "Local rank ID");
DEFINE_uint64(port, 19997, "Local port for OOB server");
DEFINE_uint64(remote_rank, 1, "Remote rank ID to connect to");
DEFINE_string(remote_ip, "", "Remote IP address");
DEFINE_uint64(remote_port, 19997, "Remote port number");
DEFINE_string(test_mode, "correctness",
              "Test mode: 'correctness', 'bandwidth', or 'unidirectional'");
DEFINE_int32(iterations, 1000, "Number of iterations for bandwidth test");
DEFINE_uint64(buffer_size, 1024 * 1024, "Buffer size in bytes");

// Example usage:
// Correctness test (100 iterations with verification):
// ./efa_endpoint_example --gpu_index=0 --rank_id=0 --port=19997 --remote_rank=1 --remote_ip=10.1.219.155 --remote_port=19997 --test_mode=correctness --buffer_size=104857600
// ./efa_endpoint_example --gpu_index=0 --rank_id=1 --port=19997 --remote_rank=0 --remote_ip=10.1.82.97 --remote_port=19997 --test_mode=correctness --buffer_size=104857600
//
//
// Unidirectional test (rank 0 sends, rank 1 receives):
// ./efa_endpoint_example --gpu_index=0 --rank_id=0 --port=19997 --remote_rank=1 --remote_ip=10.1.219.155 --remote_port=19997 --test_mode=unidirectional --iterations=100 --buffer_size=104857600
// ./efa_endpoint_example --gpu_index=0 --rank_id=1 --port=19997 --remote_rank=0 --remote_ip=10.1.82.97 --remote_port=19997 --test_mode=unidirectional --iterations=100 --buffer_size=104857600

// Bandwidth test (bidirectional):
// ./efa_endpoint_example --gpu_index=0 --rank_id=0 --port=19997 --remote_rank=1 --remote_ip=10.1.219.155 --remote_port=19997 --test_mode=bandwidth --iterations=100 --buffer_size=104857600
// ./efa_endpoint_example --gpu_index=0 --rank_id=1 --port=19997 --remote_rank=0 --remote_ip=10.1.82.97 --remote_port=19997 --test_mode=bandwidth --iterations=100 --buffer_size=104857600

// Correctness test: perform 100 send/recv operations and verify results
void correctness_test(EFAEndpoint& endpoint, MemoryAllocator& allocator) {
  std::cout << "\n=== Starting Correctness Test (100 iterations) ===\n"
            << std::flush;

  int const num_iterations = 100;
  size_t test_buffer_size = FLAGS_buffer_size;

  // Allocate buffers
  auto send_mem = allocator.allocate(test_buffer_size, MemoryType::GPU,
                                     endpoint.getContext(0));
  auto recv_mem = allocator.allocate(test_buffer_size, MemoryType::GPU,
                                     endpoint.getContext(0));

  if (!endpoint.regMem(send_mem)) {
    throw std::runtime_error("Failed to register send_mem");
  }
  if (!endpoint.regMem(recv_mem)) {
    throw std::runtime_error("Failed to register recv_mem");
  }

  std::cout << "Allocated buffers of size " << test_buffer_size << " bytes\n"
            << std::flush;

  // Host buffers for verification
  char* h_send_data = (char*)malloc(test_buffer_size);
  char* h_recv_data = (char*)malloc(test_buffer_size);

  int passed = 0;
  int failed = 0;

  for (int i = 0; i < num_iterations; i++) {
    // Prepare unique test data for this iteration with header and footer
    std::string header = "START:Rank " + std::to_string(FLAGS_rank_id) +
                         " iteration " + std::to_string(i);
    std::string footer = "END:Rank " + std::to_string(FLAGS_rank_id) +
                         " iteration " + std::to_string(i);

    memset(h_send_data, 0, test_buffer_size);
    memset(h_recv_data, 0, test_buffer_size);

    // Write header at the beginning
    strcpy(h_send_data, header.c_str());

    // Write footer at the end (ensure it fits)
    if (test_buffer_size > footer.length() + 1) {
      strcpy(h_send_data + test_buffer_size - footer.length() - 1,
             footer.c_str());
    }

    // Copy to GPU
    cudaMemcpy(send_mem->addr, h_send_data, test_buffer_size,
               cudaMemcpyHostToDevice);
    cudaMemset(recv_mem->addr, 0, test_buffer_size);

    // Create requests
    auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
    auto send_req =
        std::make_shared<EFASendRequest>(send_mem, remote_mem_placeholder);
    auto recv_req = std::make_shared<EFARecvRequest>(recv_mem);

    // Post recv first
    int64_t recv_index = endpoint.recv(FLAGS_remote_rank, recv_req);
    std::cout << "recv_index:" << recv_index << std::endl << std::flush;
    if (recv_index < 0) {
      std::cerr << "Failed to post recv request\n" << std::flush;
      failed++;
      continue;
    }

    // Post send
    int64_t send_wr_id = endpoint.send(FLAGS_remote_rank, send_req);

    std::cout << "send_wr_id:" << send_wr_id << std::endl << std::flush;
    ;
    if (send_wr_id < 0) {
      std::cerr << "Failed to post send request\n";
      failed++;
      continue;
    }

    // Wait for completion
    endpoint.checkRecvComplete(FLAGS_remote_rank, recv_index);
    std::cout << "After checkRecvComplete\n" << std::flush;
    ;
    endpoint.checkSendComplete(FLAGS_remote_rank, send_wr_id);

    // Verify received data - check both header and footer
    cudaMemcpy(h_recv_data, recv_mem->addr, test_buffer_size,
               cudaMemcpyDeviceToHost);

    std::string expected_header = "START:Rank " +
                                  std::to_string(FLAGS_remote_rank) +
                                  " iteration " + std::to_string(i);
    std::string expected_footer = "END:Rank " +
                                  std::to_string(FLAGS_remote_rank) +
                                  " iteration " + std::to_string(i);

    // Extract received header and footer
    std::string recv_header(
        h_recv_data, std::min(expected_header.length(), test_buffer_size));
    std::string recv_footer;
    if (test_buffer_size > expected_footer.length() + 1) {
      recv_footer = std::string(h_recv_data + test_buffer_size -
                                expected_footer.length() - 1);
    }

    bool header_match = (recv_header == expected_header);
    bool footer_match = (recv_footer == expected_footer);

    if (header_match && footer_match) {
      passed++;
      if (i % 10 == 0) {
        std::cout << "Iteration " << i
                  << " PASSED (header and footer verified)\n"
                  << std::flush;
      }
    } else {
      failed++;
      std::cout << "Iteration " << i << " FAILED:\n";
      if (!header_match) {
        std::cout << "  Header mismatch - Expected: \"" << expected_header
                  << "\", Got: \"" << recv_header << "\"\n";
      }
      if (!footer_match) {
        std::cout << "  Footer mismatch - Expected: \"" << expected_footer
                  << "\", Got: \"" << recv_footer << "\"\n";
      }
      std::cout << std::flush;
    }
  }

  free(h_send_data);
  free(h_recv_data);

  std::cout << "\n=== Correctness Test Results ===\n";
  std::cout << "Total iterations: " << num_iterations << "\n";
  std::cout << "Passed: " << passed << "\n";
  std::cout << "Failed: " << failed << "\n";
  std::cout << "Success rate: " << (100.0 * passed / num_iterations) << "%\n";
}

// Unidirectional bandwidth test: rank 0 only sends, rank 1 only receives
void unidirectional_test(EFAEndpoint& endpoint, MemoryAllocator& allocator,
                         int iterations) {
  std::cout << "\n=== Starting Unidirectional Bandwidth Test (" << iterations
            << " iterations) ===\n";
  std::cout << "Rank " << FLAGS_rank_id
            << " role: " << (FLAGS_rank_id == 0 ? "SENDER" : "RECEIVER")
            << "\n";

  size_t test_buffer_size = FLAGS_buffer_size;

  // Allocate buffers
  auto send_mem = allocator.allocate(test_buffer_size, MemoryType::GPU,
                                     endpoint.getContext(0));
  auto recv_mem = allocator.allocate(test_buffer_size, MemoryType::GPU,
                                     endpoint.getContext(0));

  if (!endpoint.regMem(send_mem)) {
    throw std::runtime_error("Failed to register send_mem");
  }
  if (!endpoint.regMem(recv_mem)) {
    throw std::runtime_error("Failed to register recv_mem");
  }

  std::cout << "Buffer size: " << test_buffer_size << " bytes ("
            << (test_buffer_size / (1024.0 * 1024.0)) << " MB)\n";

  // Initialize send buffer
  char* h_data = (char*)malloc(test_buffer_size);
  memset(h_data, 'A', test_buffer_size);
  cudaMemcpy(send_mem->addr, h_data, test_buffer_size, cudaMemcpyHostToDevice);
  free(h_data);

  // Warmup
  std::cout << "Running warmup (10 iterations)...\n";
  for (int i = 0; i < 10; i++) {
    if (FLAGS_rank_id == 0) {
      // Rank 0: only send
      auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
      auto send_req =
          std::make_shared<EFASendRequest>(send_mem, remote_mem_placeholder);
      int64_t send_wr_id = endpoint.send(FLAGS_remote_rank, send_req);
      endpoint.checkSendComplete(FLAGS_remote_rank, send_wr_id);
    } else {
      // Rank 1: only receive
      auto recv_req = std::make_shared<EFARecvRequest>(recv_mem);
      int64_t recv_index = endpoint.recv(FLAGS_remote_rank, recv_req);
      endpoint.checkRecvComplete(FLAGS_remote_rank, recv_index);
    }
  }

  std::cout << "Starting benchmark...\n";

  // Benchmark
  auto start_time = std::chrono::high_resolution_clock::now();

  if (FLAGS_rank_id == 0) {
    // Rank 0: sender only
    std::vector<std::pair<int, int64_t>>
        send_infos;  // (channel_id, send_wr_id)
    send_infos.reserve(iterations);

    // First, send all messages
    for (int i = 0; i < iterations; i++) {
      auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
      auto send_req =
          std::make_shared<EFASendRequest>(send_mem, remote_mem_placeholder);

      int64_t send_wr_id = endpoint.send(FLAGS_remote_rank, send_req);
      send_infos.push_back({send_req->channel_id, send_wr_id});
      endpoint.checkSendComplete(FLAGS_remote_rank, send_wr_id);
      if ((i + 1) % 100 == 0) {
        std::cout << "Sent " << (i + 1) << " messages\n";
      }
    }

    // Then, check all send completions
    // for (auto const& [channel_id, send_wr_id] : send_infos) {
    //   endpoint.checkSendComplete(FLAGS_remote_rank, send_wr_id);
    // }
    std::cout << "All sends completed\n";
  } else {
    // Rank 1: receiver only
    std::vector<int64_t> recv_indices;
    recv_indices.reserve(iterations);

    // First, recv all messages
    for (int i = 0; i < iterations; i++) {
      auto recv_req = std::make_shared<EFARecvRequest>(recv_mem);

      int64_t recv_index = endpoint.recv(FLAGS_remote_rank, recv_req);
      recv_indices.push_back(recv_index);
      endpoint.checkRecvComplete(FLAGS_remote_rank, recv_index);
      if ((i + 1) % 100 == 0) {
        std::cout << "Received " << (i + 1) << " messages\n";
      }
    }

    // Then, check all recv completions
    // for (int64_t recv_index : recv_indices) {
    //   endpoint.checkRecvComplete(FLAGS_remote_rank, recv_index);
    // }
    std::cout << "All receives completed\n";
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  // Calculate statistics
  double elapsed_seconds =
      std::chrono::duration<double>(end_time - start_time).count();
  double total_bytes = static_cast<double>(test_buffer_size) * iterations;
  double bandwidth_gbps =
      (total_bytes / elapsed_seconds) / (1024.0 * 1024.0 * 1024.0);
  double latency_us = (elapsed_seconds / iterations) * 1000000.0;

  std::cout << "\n=== Unidirectional Bandwidth Test Results (Rank "
            << FLAGS_rank_id << ") ===\n";
  std::cout << "Role: " << (FLAGS_rank_id == 0 ? "SENDER" : "RECEIVER") << "\n";
  std::cout << "Iterations: " << iterations << "\n";
  std::cout << "Buffer size: " << test_buffer_size << " bytes\n";
  std::cout << "Total time: " << elapsed_seconds << " seconds\n";
  std::cout << "Total data transferred: "
            << (total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
  std::cout << "Bandwidth: " << bandwidth_gbps << " GB/s\n";
  std::cout << "Average latency: " << latency_us << " us\n";
  std::cout << "Operations per second: " << (iterations / elapsed_seconds)
            << "\n";
}

// Bandwidth test: perform N send/recv operations and measure bandwidth
void bandwidth_test(EFAEndpoint& endpoint, MemoryAllocator& allocator,
                    int iterations) {
  std::cout << "\n=== Starting Bandwidth Test (" << iterations
            << " iterations) ===\n"
            << std::flush;

  size_t test_buffer_size = FLAGS_buffer_size;

  // Allocate buffers
  auto send_mem = allocator.allocate(test_buffer_size, MemoryType::GPU,
                                     endpoint.getContext(0));
  auto recv_mem = allocator.allocate(test_buffer_size, MemoryType::GPU,
                                     endpoint.getContext(0));

  if (!endpoint.regMem(send_mem)) {
    throw std::runtime_error("Failed to register send_mem");
  }
  if (!endpoint.regMem(recv_mem)) {
    throw std::runtime_error("Failed to register recv_mem");
  }

  std::cout << "Buffer size: " << test_buffer_size << " bytes ("
            << (test_buffer_size / (1024.0 * 1024.0)) << " MB)\n"
            << std::flush;

  // Initialize send buffer
  char* h_data = (char*)malloc(test_buffer_size);
  memset(h_data, 'A', test_buffer_size);
  cudaMemcpy(send_mem->addr, h_data, test_buffer_size, cudaMemcpyHostToDevice);
  free(h_data);

  // Warmup
  std::cout << "Running warmup (10 iterations)...\n" << std::flush;
  for (int i = 0; i < 10; i++) {
    auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
    auto send_req =
        std::make_shared<EFASendRequest>(send_mem, remote_mem_placeholder);
    auto recv_req = std::make_shared<EFARecvRequest>(recv_mem);

    int64_t recv_index = endpoint.recv(FLAGS_remote_rank, recv_req);
    int64_t send_wr_id = endpoint.send(FLAGS_remote_rank, send_req);

    endpoint.checkSendComplete(FLAGS_remote_rank, send_wr_id);
    endpoint.checkRecvComplete(FLAGS_remote_rank, recv_index);
  }

  std::cout << "Starting benchmark...\n" << std::flush;

  // Benchmark

  std::vector<int64_t> recv_indices;
  std::vector<std::pair<int, int64_t>> send_infos;  // (channel_id, send_wr_id)
  recv_indices.reserve(iterations);
  send_infos.reserve(iterations);
  auto start_time = std::chrono::high_resolution_clock::now();

  // 统计第一段耗时：304-319行 (recv/send操作)
  auto phase1_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
    auto send_req =
        std::make_shared<EFASendRequest>(send_mem, remote_mem_placeholder);
    auto recv_req = std::make_shared<EFARecvRequest>(recv_mem);

    // 先进行recv/send操作，收集index
    int64_t recv_index = endpoint.recv(FLAGS_remote_rank, recv_req);
    recv_indices.push_back(recv_index);

    int64_t send_wr_id = endpoint.send(FLAGS_remote_rank, send_req);
    send_infos.push_back({send_req->channel_id, send_wr_id});
    // if ((i + 1) % 100 == 0) {
    //   std::cout << "Completed " << (i + 1) << " iterations\n";
    // }
  }
  auto phase1_end = std::chrono::high_resolution_clock::now();
  double phase1_time =
      std::chrono::duration<double>(phase1_end - phase1_start).count();
  std::cout << "[Timing] Phase 1 (recv/send loop): " << phase1_time
            << " seconds\n"
            << std::flush;

  // 统计第二段耗时：321-329行 (check complete操作)
  auto phase2_start = std::chrono::high_resolution_clock::now();
  // 统一遍历进行checkSendComplete
  for (auto const& [channel_id, send_wr_id] : send_infos) {
    // std::cout << "channel_id:" <<channel_id<<",
    // send_wr_id:"<<send_wr_id<<std::endl<<std::flush;
    endpoint.checkSendComplete(FLAGS_remote_rank, send_wr_id);
  }
  // 统一遍历进行checkRecvComplete
  for (int64_t recv_index : recv_indices) {
    // std::cout << "recv_index:" <<recv_index<<std::endl<<std::flush;
    endpoint.checkRecvComplete(FLAGS_remote_rank, recv_index);
  }
  auto phase2_end = std::chrono::high_resolution_clock::now();
  double phase2_time =
      std::chrono::duration<double>(phase2_end - phase2_start).count();
  std::cout << "[Timing] Phase 2 (check complete loop): " << phase2_time
            << " seconds\n"
            << std::flush;

  auto end_time = std::chrono::high_resolution_clock::now();

  // Calculate statistics
  double elapsed_seconds =
      std::chrono::duration<double>(end_time - start_time).count();
  double total_bytes =
      static_cast<double>(test_buffer_size) * iterations * 2;  // send + recv
  double bandwidth_gbps =
      (total_bytes / elapsed_seconds) / (1024.0 * 1024.0 * 1024.0);
  double latency_us = (elapsed_seconds / iterations) * 1000000.0;

  std::cout << "\n=== Bandwidth Test Results ===\n" << std::flush;
  std::cout << "Iterations: " << iterations << "\n" << std::flush;

  std::cout << "Total time: " << elapsed_seconds << " seconds\n" << std::flush;
  std::cout << "Total data transferred: "
            << (total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
            << std::flush;
  std::cout << "Message size: " << test_buffer_size << " bytes ("
            << (test_buffer_size / (1024.0 * 1024.0)) << " MB)\n"
            << std::flush;
  std::cout << "Bandwidth: " << bandwidth_gbps << " GB/s\n" << std::flush;
  std::cout << "Average latency (per round-trip): " << latency_us << " us\n"
            << std::flush;
  std::cout << "Operations per second: " << (iterations / elapsed_seconds)
            << "\n"
            << std::flush;
}

int main(int argc, char* argv[]) {
  // Initialize Google's logging library
  google::InitGoogleLogging(argv[0]);

  // Set logging level to INFO
  // FLAGS_minloglevel = google::INFO;
  FLAGS_minloglevel = google::WARNING;
  FLAGS_logtostderr = true;

  // Parse command line flags
  gflags::SetUsageMessage("EFAEndpoint usage example");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Validate required flags
  if (FLAGS_remote_ip.empty()) {
    std::cerr << "remote_ip is required!\n";
    gflags::ShowUsageWithFlagsRestrict(argv[0], "main.cpp");
    return 1;
  }

  std::cout << "=== EFAEndpoint Usage Example ===\n";
  std::cout << "GPU Index: " << FLAGS_gpu_index << "\n";
  std::cout << "Rank ID: " << FLAGS_rank_id << "\n";
  std::cout << "Port: " << FLAGS_port << "\n";
  std::cout << "Remote Rank: " << FLAGS_remote_rank << "\n";
  std::cout << "Remote IP: " << FLAGS_remote_ip << "\n";
  std::cout << "Remote Port: " << FLAGS_remote_port << "\n";
  std::cout << "================================\n\n";

  //     size_t gpu_size = 1024 * 1024;
  // auto gpu_mem =
  //     allocator.allocate(gpu_size, MemoryType::HOST, contexts_[0]);
  // std::cout << "Allocated " << gpu_mem->size << " bytes of GPU memory at "
  //           << gpu_mem->addr << std::endl;
  // RemoteMemInfo info(gpu_mem);
  // recv_test_ = std::make_shared<EFARecvRequest>(gpu_mem);
  try {
    // Set GPU device for the entire process
    cudaError_t cuda_err = cudaSetDevice(FLAGS_gpu_index);
    if (cuda_err != cudaSuccess) {
      std::cerr << "Failed to set GPU device " << FLAGS_gpu_index << ": "
                << cudaGetErrorString(cuda_err) << "\n";
      return 1;
    }
    std::cout << "Set GPU device to: " << FLAGS_gpu_index << "\n\n";

    // Initialize RDMA device manager
    std::cout << "Initializing RDMA device manager...\n";
    auto& device_manager = RdmaDeviceManager::instance();
    auto results = device_manager.get_best_dev_idx(FLAGS_gpu_index);

    // Print results
    std::cout << "Best device indices for GPU " << FLAGS_gpu_index << ": [";
    for (size_t i = 0; i < results.size(); i++) {
      std::cout << results[i];
      if (i < results.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "Found " << device_manager.deviceCount()
              << " RDMA device(s)\n\n";

    // Create EFAEndpoint with device_ids = {0}
    std::cout << "Creating EFAEndpoint...\n";
    std::vector<size_t> device_ids = {0, 1};
    EFAEndpoint endpoint(FLAGS_gpu_index, FLAGS_rank_id, FLAGS_port);
    std::cout << "EFAEndpoint created successfully\n\n";

    // Create OOBMetaData for remote rank
    std::cout << "Setting up remote rank metadata...\n";
    auto remote_meta = std::make_shared<OOBMetaData>();
    remote_meta->server_ip = FLAGS_remote_ip;
    remote_meta->server_port = FLAGS_remote_port;

    // Add remote rank metadata
    std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> rank_meta_map;
    rank_meta_map[FLAGS_remote_rank] = remote_meta;
    endpoint.add_rank_oob_meta(rank_meta_map);
    std::cout << "Added remote rank " << FLAGS_remote_rank
              << " metadata (IP: " << FLAGS_remote_ip
              << ", Port: " << FLAGS_remote_port << ")\n\n";

    // Wait a bit to ensure both endpoints are ready
    std::cout
        << "Waiting for 2 seconds to ensure both endpoints are ready...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Connect to remote rank
    std::cout << "Connecting to remote rank " << FLAGS_remote_rank << "...\n";
    // bool connect_result = endpoint.build_connect_sync(FLAGS_remote_rank);
    // // bool connect_result = endpoint.build_connect(FLAGS_remote_rank);
    // endpoint.connect_check(FLAGS_remote_rank);
    std::string remote_ip;
    int remote_dev;
    int remote_gpuidx;
    endpoint.uccl_connect(0,0,0,0,FLAGS_remote_ip,FLAGS_remote_port);
    endpoint.uccl_accept(0,0,0,remote_ip,&remote_dev,&remote_gpuidx);
    // if (connect_result>=0) {
    //   std::cout << "Successfully connected to remote rank " << FLAGS_remote_rank
    //             << "\n";
    // } else {
    //   std::cerr << "Failed to connect to remote rank " << FLAGS_remote_rank
    //             << "\n";
    //   return 1;
    // }

    // std::this_thread::sleep_for(std::chrono::seconds(5));

    // Create memory allocator
    MemoryAllocator allocator;

    // Run the selected test mode
    if (FLAGS_test_mode == "correctness") {
      std::cout << "?????????????????????" << std::endl;
      correctness_test(endpoint, allocator);
    } else if (FLAGS_test_mode == "bandwidth") {
      bandwidth_test(endpoint, allocator, FLAGS_iterations);
    } else if (FLAGS_test_mode == "unidirectional") {
      unidirectional_test(endpoint, allocator, FLAGS_iterations);
    } else {
      std::cerr << "Invalid test_mode: " << FLAGS_test_mode << "\n";
      std::cerr << "Valid options are: 'correctness', 'bandwidth', or "
                   "'unidirectional'\n";
      return 1;
    }

    std::cout << "\nTest completed. Press Ctrl+C to exit...\n";

    // Keep the program running to maintain the connection
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  google::ShutdownGoogleLogging();
  return 0;
}