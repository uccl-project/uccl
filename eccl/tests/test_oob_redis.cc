#include "oob.h"
#include "test.h"
#include "utils.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>
#include <chrono>

#include <iostream>
#include <thread>
#include <vector>
#include <memory>

void publisher() {
    RedisExchanger ex("127.0.0.1", 6379);
    if (!ex.valid()) {
        fprintf(stderr, "[ERROR] Publisher failed to connect to Redis\n");
        return;
    }

    RDMAConnectionInfo remote{};
    remote.qp_num = 4321;
    remote.lid   = 8765;
    for (int i = 0; i < 16; i++) remote.gid[i] = 15 - i;

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    if (!ex.publish("rdma:peer:1:0", remote)) {
        fprintf(stderr, "[ERROR] Publisher failed to publish remote RDMA info\n");
        return;
    }
    printf("[INFO] Publisher thread published remote RDMA info\n");
}

void test_redis_oob() {
    RedisExchanger ex("127.0.0.1", 6379);
    if (!ex.valid()) {
        fprintf(stderr, "[ERROR] Failed to connect to Redis\n");
    }
    printf("[INFO] Connected to Redis\n");

    RDMAConnectionInfo local{};
    local.qp_num = 1234;
    local.lid    = 5678;
    for (int i = 0; i < 16; i++) local.gid[i] = i;

    if (!ex.publish("rdma:peer:0:0", local)) {
        fprintf(stderr, "[ERROR] Failed to publish local RDMA info\n");
    }
    printf("[INFO] Published local RDMA info\n");

    std::thread pub_thread(publisher);

    RDMAConnectionInfo remote{};
    if (ex.wait_and_fetch("rdma:peer:1:0", remote)) {
        printf("[INFO] Got remote RDMA info:\n");
        printf("       qp_num=%u lid=%u\n",
               remote.qp_num, remote.lid);
        printf("       gid=");
        for (int i = 0; i < 16; i++) printf("%02x ", remote.gid[i]);
        printf("\n");
    } else {
        fprintf(stderr, "[WARN] Timeout waiting for remote RDMA info\n");
        pub_thread.join();
    }

    pub_thread.join();
    printf("[INFO] Redis OOB multithread test completed successfully\n");
}

void rank_thread(int local_rank, int world_size, const std::string& redis_ip, int redis_port) {
    auto ex = std::make_shared<RedisExchanger>(redis_ip, redis_port);
    if (!ex->valid()) {
        std::cerr << "[ERROR] Rank " << local_rank << " failed to connect to Redis" << std::endl;
        return;
    }

    CommunicatorMeta local;
    local.host_id = generate_host_id() + "_" + std::to_string(local_rank);
    local.is_ready = true;

    std::string key = "meta:" + std::to_string(local_rank);

    if (!ex->publish(key, local)) {
        std::cerr << "[ERROR] Rank " << local_rank
                  << " failed to publish meta to key " << key << std::endl;
        return;
    }
    std::cout << "[INFO] Rank " << local_rank
              << " published meta to key " << key << std::endl;

    for (int r = 0; r < world_size; ++r) {
        if (r == local_rank) continue;
        std::string remote_key = "meta:" + std::to_string(r);
        CommunicatorMeta remote;
        if (ex->wait_and_fetch(remote_key, remote, 50, 100)) {
            std::cout << "[INFO] Rank " << local_rank
                      << " fetched meta for rank " << r
                      << ", host_id=" << remote.host_id
                      << ", is_ready=" << remote.is_ready << std::endl;
        } else {
            std::cerr << "[WARN] Rank " << local_rank
                      << " timeout waiting for meta of rank " << r << std::endl;
        }
    }

    std::cout << "[INFO] Rank " << local_rank << " completed meta exchange" << std::endl;
}

void test_meta_exchange_multi_threads(int world_size) {
    std::vector<std::thread> threads;
    for (int rank = 0; rank < world_size; ++rank) {
        threads.emplace_back(rank_thread, rank, world_size, "127.0.0.1", 6379);
    }

    for (auto& t : threads) t.join();
    std::cout << "[INFO] CommunicatorMeta multithread Redis test completed" << std::endl;
}

