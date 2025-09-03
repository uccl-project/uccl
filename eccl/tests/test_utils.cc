#include "test.h"
#include "utils.h"

void test_find_best_rdma_for_gpu(int gpu_id) {
    // find best NIC for cur gpu
    auto [nic_id, nic_name] = find_best_rdma_for_gpu(gpu_id);
    struct ibv_device** dev_list = ibv_get_device_list(nullptr);
    if (!dev_list) {
        std::cerr << "Failed to get IB devices" << std::endl;
        std::abort();
    }
    struct ibv_device* nic_dev = nullptr;
    for (int i = 0; dev_list[i] != nullptr; ++i) {
        if (nic_name == ibv_get_device_name(dev_list[i])) {
            nic_dev = dev_list[i];
            break;
        }
    }
    if (!nic_dev) {
        std::cerr << "Could not match RDMA NIC: " << nic_name << std::endl;
        ibv_free_device_list(dev_list);
        std::abort();
    }
    struct ibv_context* nic_ibv_ctx_ = ibv_open_device(nic_dev);
    if (!nic_ibv_ctx_) {
        std::cerr << "Failed to open ibv context for NIC: " << nic_name << std::endl;
        ibv_free_device_list(dev_list);
        std::abort();
    }
    ibv_free_device_list(dev_list);

    std::cout << "Communicator initialized: GPU " << gpu_id
              << " -> RDMA NIC " << nic_name << std::endl;
}

void test_generate_host_id() {
    auto id = generate_host_id(false);
    auto id_with_ip = generate_host_id(true);

    std::cout << "Test Utils: generate id " << id
              << " with ip " << id_with_ip << std::endl;
}