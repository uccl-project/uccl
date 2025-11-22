
    // // Test based on rank_id
    // if (rank_id == 0) {
    //   // Rank 0: First recv, then send
    //   std::cout << "\n[Rank 0] Posting recv request...\n";
    //   int64_t recv_index = endpoint.recv(remote_rank, recv_req);
    //   std::cout << "[Rank 0] Recv posted with index: " << recv_index << "\n";

    //   std::cout << "[Rank 0] Checking recv completion...\n";
    //   endpoint.checkRecvComplete(remote_rank, recv_index);
    //   std::cout << "[Rank 0] Recv completed!\n";

    //   // Read received data
    //   char* h_data = (char*)malloc(buffer_size);
    //   if (recv_mem->type == MemoryType::GPU) {
    //     cudaMemcpy(h_data, recv_mem->addr, buffer_size, cudaMemcpyDeviceToHost);
    //   } else {
    //     memcpy(h_data, recv_mem->addr, buffer_size);
    //   }
    //   std::cout << "[Rank 0] Received message: \"" << h_data << "\"\n";
    //   free(h_data);

    //   // std::this_thread::sleep_for(std::chrono::seconds(1));

    //   std::cout << "\n[Rank 0] Posting send request...\n";
    //   int64_t send_wr_id = endpoint.send(remote_rank, send_req);
    //   std::cout << "[Rank 0] Send posted with wr_id: " << send_wr_id << "\n";

    //   std::cout << "[Rank 0] Checking send completion...\n";
    //   endpoint.checkSendComplete(remote_rank,send_req->channel_id, send_wr_id);
    //   std::cout << "[Rank 0] Send completed!\n";

    // } else {
    //   // Rank 1: First send, then recv
    //   std::cout << "\n[Rank 1] Posting send request...\n";
    //   int64_t send_wr_id = endpoint.send(remote_rank, send_req);
    //   std::cout << "[Rank 1] Send posted with wr_id: " << send_wr_id << "\n";

    //   std::cout << "[Rank 1] Checking send completion...\n";
    //   endpoint.checkSendComplete(remote_rank, send_req->channel_id, send_wr_id);
    //   std::cout << "[Rank 1] Send completed!\n";

    //   // std::this_thread::sleep_for(std::chrono::seconds(1));

    //   std::cout << "\n[Rank 1] Posting recv request...\n";
    //   int64_t recv_index = endpoint.recv(remote_rank, recv_req);
    //   std::cout << "[Rank 1] Recv posted with index: " << recv_index << "\n";

    //   std::cout << "[Rank 1] Checking recv completion...\n";
    //   endpoint.checkRecvComplete(remote_rank, recv_index);
    //   std::cout << "[Rank 1] Recv completed!\n";

    //   // Read received data
    //   char* h_data = (char*)malloc(buffer_size);
    //   if (recv_mem->type == MemoryType::GPU) {
    //     cudaMemcpy(h_data, recv_mem->addr, buffer_size, cudaMemcpyDeviceToHost);
    //   } else {
    //     memcpy(h_data, recv_mem->addr, buffer_size);
    //   }
    //   std::cout << "[Rank 1] Received message: \"" << h_data << "\"\n";
    //   free(h_data);
    // }
