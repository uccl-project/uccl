// Exercise C API IPC metadata across processes with different visible devices.
// Link against libuccl_p2p.so and libcudart, then run:
// UCCL_P2P_TRANSPORT=nccl ./scenario_ipc_import_device 0,1 1 0 0
// Arguments: exporter visibility, exporter ordinal, importer visibility,
// importer ordinal. NCCL establishes the connection; the explicit IPC
// descriptors select IPC for every data transfer. The parent launches both
// runtimes through exec.
#include "uccl_engine.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime_api.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static void cuda_ok(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}

static void send_bytes(int fd, void const* data, size_t size) {
  auto* cursor = static_cast<char const*>(data);
  while (size) {
    auto count = write(fd, cursor, size);
    if (count <= 0) throw std::runtime_error("rendezvous write failed");
    cursor += count;
    size -= count;
  }
}

static void receive_bytes(int fd, void* data, size_t size) {
  auto* cursor = static_cast<char*>(data);
  while (size) {
    auto count = read(fd, cursor, size);
    if (count <= 0) throw std::runtime_error("rendezvous read failed");
    cursor += count;
    size -= count;
  }
}

static void run_peer(bool exporter, int device, int fd) {
  cuda_ok(cudaSetDevice(device));
  int devices = 0;
  cuda_ok(cudaGetDeviceCount(&devices));
  auto* engine = uccl_engine_create(1, false);
  if (!engine) throw std::runtime_error("engine construction failed");

  std::vector<size_t> sizes{1, 65539, 2 * 1024 * 1024 + 17};
  std::vector<size_t> offsets;
  size_t slab_size = 64;
  for (auto size : sizes) {
    offsets.push_back(slab_size);
    slab_size += size + 64;
  }
  void* gpu = nullptr;
  void* pinned = nullptr;
  cuda_ok(cudaMalloc(&gpu, slab_size));
  cuda_ok(cudaMallocHost(&pinned, slab_size));
  uccl_mr_t gpu_mr = 0, host_mr = 0;
  if (uccl_engine_reg(engine, reinterpret_cast<uintptr_t>(gpu), slab_size,
                      gpu_mr) ||
      uccl_engine_reg(engine, reinterpret_cast<uintptr_t>(pinned), slab_size,
                      host_mr))
    throw std::runtime_error("buffer registration failed");

  std::string endpoint;
  uintptr_t remote_base = 0;
  std::array<char, IPC_INFO_SIZE> remote_info{};
  uccl_conn_t* connection = nullptr;
  if (exporter) {
    char* metadata = nullptr;
    if (uccl_engine_get_metadata(engine, &metadata))
      throw std::runtime_error("endpoint metadata failed");
    endpoint = metadata;
    delete[] metadata;
    bool has_ipc = false;
    if (uccl_engine_get_ipc_info(engine, reinterpret_cast<uintptr_t>(gpu),
                                 remote_info.data(), &has_ipc) ||
        !has_ipc)
      throw std::runtime_error("registered buffer has no IPC descriptor");
    uint32_t length = endpoint.size();
    remote_base = reinterpret_cast<uintptr_t>(gpu);
    send_bytes(fd, &length, sizeof(length));
    send_bytes(fd, endpoint.data(), length);
    send_bytes(fd, &remote_base, sizeof(remote_base));
    send_bytes(fd, remote_info.data(), remote_info.size());
    char remote_ip[64];
    int remote_device = -1;
    connection = uccl_engine_accept(engine, remote_ip, sizeof(remote_ip),
                                    &remote_device);
  } else {
    uint32_t length = 0;
    receive_bytes(fd, &length, sizeof(length));
    if (length > 1024) throw std::runtime_error("invalid endpoint metadata");
    endpoint.resize(length);
    receive_bytes(fd, endpoint.data(), length);
    receive_bytes(fd, &remote_base, sizeof(remote_base));
    receive_bytes(fd, remote_info.data(), remote_info.size());
    auto question = endpoint.rfind('?');
    auto colon = endpoint.rfind(':', question);
    if (question == std::string::npos || colon == std::string::npos)
      throw std::runtime_error("unexpected endpoint metadata");
    std::string ip = endpoint.substr(0, colon);
    if (!ip.empty() && ip.front() == '[') ip = ip.substr(1, ip.size() - 2);
    int port = std::stoi(endpoint.substr(colon + 1, question - colon - 1));
    connection = uccl_engine_connect(
        engine, ip.c_str(), endpoint.substr(question + 1).c_str(), port, false);
    if (connection && !uccl_engine_conn_is_local(connection))
      throw std::runtime_error("connection did not select local IPC");
  }
  if (!connection) throw std::runtime_error("connection failed");
  int exported_device = -1;
  std::memcpy(&exported_device,
              remote_info.data() + sizeof(cudaIpcMemHandle_t) +
                  sizeof(uintptr_t) + sizeof(size_t),
              sizeof(exported_device));
  char bdf[32];
  cuda_ok(cudaDeviceGetPCIBusId(bdf, sizeof(bdf), device));
  std::printf(
      "{\"peer\":\"%s\",\"local_bdf\":\"%s\",\"local_device\":%d,"
      "\"visible_devices\":%d,\"exported_ordinal\":%d}\n",
      exporter ? "exporter" : "importer", bdf, device, devices,
      exported_device);
  std::fflush(stdout);

  int case_id = 0;
  for (int round = 0; round < 2; ++round) {
    for (bool reading : {false, true}) {
      for (bool host : {false, true}) {
        for (size_t count : {size_t(1), sizes.size()}) {
          ++case_id;
          std::vector<unsigned char> expected(slab_size, 0xa5);
          std::vector<unsigned char> empty = expected;
          for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < sizes[i]; ++j) {
              expected[offsets[i] + j] = (j + 17 * i + 53 * case_id) % 251;
              empty[offsets[i] + j] = 0xff;
            }
          }
          bool source = exporter ? reading : !reading;
          auto const& initial = source ? expected : empty;
          void* local = !exporter && host ? pinned : gpu;
          cuda_ok(cudaSetDevice(device));
          if (!exporter && host)
            std::memcpy(local, initial.data(), slab_size);
          else
            cuda_ok(cudaMemcpy(local, initial.data(), slab_size,
                               cudaMemcpyHostToDevice));
          int reply = 0;
          int caller_device =
              !exporter && devices > 1 ? (device + 1) % devices : device;
          if (exporter) {
            send_bytes(fd, &reply, sizeof(reply));
            receive_bytes(fd, &reply, sizeof(reply));
          } else {
            receive_bytes(fd, &reply, sizeof(reply));
            std::vector<std::array<char, IPC_INFO_SIZE>> blobs(count,
                                                               remote_info);
            std::vector<char*> blob_ptrs;
            std::vector<void*> pointers;
            std::vector<FifoItem> fifo(count);
            std::vector<size_t> lengths(sizes.begin(), sizes.begin() + count);
            for (size_t i = 0; i < count; ++i) {
              if (uccl_engine_update_ipc_info(blobs[i].data(),
                                              remote_base + offsets[i],
                                              remote_base, sizes[i]))
                throw std::runtime_error("IPC subrange metadata failed");
              blob_ptrs.push_back(blobs[i].data());
              pointers.push_back(static_cast<char*>(local) + offsets[i]);
              fifo[i].addr = remote_base + offsets[i];
              fifo[i].size = sizes[i];
            }
            uint64_t transfer = 0;
            std::vector<uccl_mr_t> ids(count, host ? host_mr : gpu_mr);
            cuda_ok(cudaSetDevice(caller_device));
            int result = reading ? uccl_engine_read_vector(
                                       connection, ids, pointers, lengths, fifo,
                                       count, &transfer, blob_ptrs)
                                 : uccl_engine_write_vector(
                                       connection, ids, pointers, lengths, fifo,
                                       count, &transfer, blob_ptrs);
            if (result) throw std::runtime_error("IPC submission failed");
            int restored = -1;
            cuda_ok(cudaGetDevice(&restored));
            if (restored != caller_device)
              throw std::runtime_error("caller device was not restored");
            auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(20);
            while (!uccl_engine_xfer_status(connection, transfer)) {
              if (std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("IPC completion timed out");
              std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            send_bytes(fd, &reply, sizeof(reply));
          }
          std::vector<unsigned char> actual(slab_size);
          cuda_ok(cudaSetDevice(device));
          if (!exporter && host)
            std::memcpy(actual.data(), local, slab_size);
          else
            cuda_ok(cudaMemcpy(actual.data(), local, slab_size,
                               cudaMemcpyDeviceToHost));
          reply = actual == expected ? 0 : 1;
          int peer_reply = 0;
          send_bytes(fd, &reply, sizeof(reply));
          receive_bytes(fd, &peer_reply, sizeof(peer_reply));
          if (reply || peer_reply)
            throw std::runtime_error("payload or guard bytes differ");
          std::printf(
              "{\"peer\":\"%s\",\"case\":%d,\"round\":%d,\"read\":%s,"
              "\"pinned\":%s,\"iovs\":%zu,\"visible_devices\":%d,"
              "\"local_device\":%d,\"caller_device\":%d,"
              "\"payload_and_guards_correct\":true}\n",
              exporter ? "exporter" : "importer", case_id, round,
              reading ? "true" : "false", host ? "true" : "false", count,
              devices, device, caller_device);
          std::fflush(stdout);
        }
      }
    }
  }
  uccl_engine_mr_destroy(engine, host_mr);
  uccl_engine_mr_destroy(engine, gpu_mr);
  uccl_engine_conn_destroy(connection);
  uccl_engine_destroy(engine);
  cuda_ok(cudaSetDevice(device));
  cuda_ok(cudaFree(gpu));
  cuda_ok(cudaFreeHost(pinned));
  close(fd);
}

int main(int argc, char** argv) {
  try {
    if (argc == 5 && std::string(argv[1]) == "--peer") {
      run_peer(std::string(argv[2]) == "exporter", std::stoi(argv[3]),
               std::stoi(argv[4]));
      return 0;
    }
    if (argc != 5)
      throw std::runtime_error(
          "usage: scenario exporter_visible exporter_device importer_visible "
          "importer_device");
    char const* transport = std::getenv("UCCL_P2P_TRANSPORT");
    if (!transport || std::string(transport) != "nccl")
      throw std::runtime_error("set UCCL_P2P_TRANSPORT=nccl");
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets))
      throw std::runtime_error("socketpair failed");
    pid_t children[2] = {-1, -1};
    bool passed = true;
    for (int peer = 0; peer < 2; ++peer) {
      children[peer] = fork();
      if (children[peer] < 0) {
        std::perror("fork");
        passed = false;
        break;
      }
      if (children[peer] == 0) {
        close(sockets[1 - peer]);
        if (setenv("CUDA_VISIBLE_DEVICES", argv[1 + 2 * peer], 1) ||
            setenv("UCCL_P2P_COMPRESS_STRATEGY", "none", 1)) {
          std::perror("setenv");
          _exit(127);
        }
        std::string fd = std::to_string(sockets[peer]);
        execl(argv[0], argv[0], "--peer", peer == 0 ? "exporter" : "importer",
              argv[2 + 2 * peer], fd.c_str(), nullptr);
        std::perror("exec");
        _exit(127);
      }
    }
    close(sockets[0]);
    close(sockets[1]);
    // Bound connection setup and stalled peers, including a failed sibling.
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(100);
    while (passed) {
      bool running = false;
      for (auto& child : children) {
        if (child < 0) continue;
        int status = 0;
        pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) {
          child = -1;
          if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) passed = false;
        } else if (result < 0 && errno != EINTR) {
          if (errno == ECHILD) child = -1;
          std::perror("waitpid");
          passed = false;
        } else {
          running = true;
        }
      }
      if (!running || !passed) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        std::fprintf(stderr, "IPC peers timed out\n");
        passed = false;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for (auto child : children) {
      if (child < 0) continue;
      kill(child, SIGKILL);
      while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
    return passed ? 0 : 1;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "IPC import-device scenario failed: %s\n",
                 error.what());
    return 1;
  }
}
