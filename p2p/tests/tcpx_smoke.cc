#include "tcpx_engine.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

bool checkCuda(cudaError_t err, char const* where) {
  if (err != cudaSuccess) {
    std::cerr << "[CUDA] " << where << " failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

bool poll_transfer(tcpx::Endpoint& endpoint, uint64_t transfer_id) {
  constexpr int kSleepMicros = 100;
  bool done = false;
  while (!done) {
    if (!endpoint.poll_async(transfer_id, &done)) {
      std::cerr << "[TCPX] poll_async failed" << std::endl;
      return false;
    }
    if (!done) {
      std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicros));
    }
  }
  return true;
}

struct Options {
  std::string mode;
  std::string ip{"127.0.0.1"};
  int port = 9999;
  int gpu = 0;
  size_t size = 1 << 20;  // 1 MiB
};

bool parse_args(int argc, char** argv, Options& out) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto pos = arg.find('=');
    std::string key = arg.substr(0, pos);
    std::string value = pos == std::string::npos ? "" : arg.substr(pos + 1);
    if (key == "--mode") {
      out.mode = value;
    } else if (key == "--ip") {
      out.ip = value;
    } else if (key == "--port") {
      out.port = std::stoi(value);
    } else if (key == "--gpu") {
      out.gpu = std::stoi(value);
    } else if (key == "--size") {
      out.size = static_cast<size_t>(std::stoul(value));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return false;
    }
  }
  if (out.mode != "server" && out.mode != "client") {
    std::cerr << "Missing or invalid --mode=server|client" << std::endl;
    return false;
  }
  return true;
}

std::vector<uint8_t> make_pattern(size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<uint8_t>(i & 0xff);
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, opts)) {
    std::cerr
        << "Usage: " << argv[0]
        << " --mode=server|client [--ip=ADDR] [--port=NUM] [--gpu=ID] [--size=BYTES]\n";
    return 1;
  }

  // Ensure both sides use the same TCP control port.
  std::string port_str = std::to_string(opts.port);
  setenv("UCCL_TCPX_OOB_PORT", port_str.c_str(), /*overwrite=*/1);
  setenv("NCCL_GPUDIRECTTCPX_CTRL_DEV", "eth0", 0);
  setenv("NCCL_NSOCKS_PERTHREAD", "2", 0);
  setenv("NCCL_SOCKET_NTHREADS", "1", 0);
  setenv("NCCL_DYNAMIC_CHUNK_SIZE", "524288", 0);
  setenv("NCCL_P2P_NET_CHUNKSIZE", "524288", 0);
  setenv("NCCL_P2P_PCI_CHUNKSIZE", "524288", 0);
  setenv("NCCL_P2P_NVL_CHUNKSIZE", "1048576", 0);
  setenv("NCCL_BUFFSIZE", "8388608", 0);
  setenv("NCCL_GPUDIRECTTCPX_TX_BINDINGS",
         "eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177", 0);
  setenv("NCCL_GPUDIRECTTCPX_RX_BINDINGS",
         "eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191", 0);
  setenv("NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS", "50000", 0);
  setenv("NCCL_GPUDIRECTTCPX_FORCE_ACK", "0", 0);
  setenv("NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP", "100", 0);
  setenv("NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP", "100", 0);
  setenv("NCCL_SOCKET_IFNAME", "eth0", 0);
  setenv("NCCL_CROSS_NIC", "0", 0);
  setenv("NCCL_NET_GDR_LEVEL", "PIX", 0);
  setenv("NCCL_P2P_PXN_LEVEL", "0", 0);
  setenv("NCCL_ALGO", "Ring", 0);
  setenv("NCCL_PROTO", "Simple", 0);
  setenv("NCCL_MAX_NCHANNELS", "8", 0);
  setenv("NCCL_MIN_NCHANNELS", "8", 0);
  setenv("NCCL_DEBUG", "INFO", 0);
  setenv("NCCL_DEBUG_SUBSYS", "ENV", 0);

  if (!checkCuda(cudaSetDevice(opts.gpu), "cudaSetDevice")) return 1;

  tcpx::Endpoint endpoint(static_cast<uint32_t>(opts.gpu), /*num_cpus=*/1);

  uint8_t* dev_buf = nullptr;
  if (!checkCuda(cudaMalloc(&dev_buf, opts.size), "cudaMalloc")) return 1;

  uint64_t mr_id = 0;
  if (!endpoint.reg(dev_buf, opts.size, mr_id)) {
    std::cerr << "[TCPX] reg failed" << std::endl;
    return 1;
  }

  if (opts.mode == "server") {
    std::cout << "[TCPX Smoke] Waiting for client on port " << opts.port
              << "...\n";
    std::string peer_ip;
    int peer_gpu = -1;
    uint64_t conn_id = 0;
    if (!endpoint.accept(peer_ip, peer_gpu, conn_id)) {
      std::cerr << "[TCPX] accept failed" << std::endl;
      return 1;
    }
    std::cout << "[TCPX Smoke] Accepted connection from " << peer_ip
              << " (peer GPU " << peer_gpu << ")\n";

    if (!checkCuda(cudaMemset(dev_buf, 0, opts.size), "cudaMemset")) return 1;

    uint64_t transfer_id = 0;
    if (!endpoint.recv_async(conn_id, mr_id, dev_buf, opts.size,
                             &transfer_id)) {
      std::cerr << "[TCPX] recv_async failed" << std::endl;
      return 1;
    }
    if (!poll_transfer(endpoint, transfer_id)) return 1;

    std::vector<uint8_t> host_buf(opts.size);
    if (!checkCuda(cudaMemcpy(host_buf.data(), dev_buf, opts.size,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H"))
      return 1;

    auto expected = make_pattern(opts.size);
    if (host_buf != expected) {
      std::cerr << "[TCPX Smoke] Data mismatch!" << std::endl;
      for (size_t i = 0; i < std::min<size_t>(64, opts.size); ++i) {
        std::cerr << "  idx " << i << " recv=" << static_cast<int>(host_buf[i])
                  << " expect=" << static_cast<int>(expected[i]) << std::endl;
      }
      return 1;
    }
    std::cout << "[TCPX Smoke] Receive verification passed (" << opts.size
              << " bytes)" << std::endl;
  } else {  // client
    std::cout << "[TCPX Smoke] Connecting to " << opts.ip << ":" << opts.port
              << "...\n";
    uint64_t conn_id = 0;
    if (!endpoint.connect(opts.ip, /*remote_gpu_idx=*/0, opts.port, conn_id)) {
      std::cerr << "[TCPX] connect failed" << std::endl;
      return 1;
    }

    auto payload = make_pattern(opts.size);
    if (!checkCuda(cudaMemcpy(dev_buf, payload.data(), opts.size,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D"))
      return 1;

    uint64_t transfer_id = 0;
    if (!endpoint.send_async(conn_id, mr_id, dev_buf, opts.size,
                             &transfer_id)) {
      std::cerr << "[TCPX] send_async failed" << std::endl;
      return 1;
    }
    if (!poll_transfer(endpoint, transfer_id)) return 1;

    std::cout << "[TCPX Smoke] Payload sent (" << opts.size << " bytes)"
              << std::endl;
  }

  endpoint.dereg(mr_id);
  cudaFree(dev_buf);
  return 0;
}
