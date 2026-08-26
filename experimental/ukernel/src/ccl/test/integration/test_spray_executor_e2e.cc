#include "coll_config.h"
#include "executor.h"
#include "backend/device_backend.h"
#include "gpu_rt.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

using namespace UKernel::CCL;

static std::string get_arg(int argc, char** argv, std::string const& name,
                           std::string const& def) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == name && i + 1 < argc) return argv[i + 1];
    std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
  }
  return def;
}

static int get_int_arg(int argc, char** argv, std::string const& name,
                       int def) {
  std::string s = get_arg(argc, argv, name, std::to_string(def));
  try {
    return std::stoi(s);
  } catch (...) {
    return def;
  }
}

int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  std::string role = get_arg(argc, argv, "--role", "");
  if (role != "server" && role != "client") {
    std::fprintf(stderr,
                 "Usage: --role=server|client [--gpu GPU] [--exchanger-ip IP] "
                 "[--exchanger-port PORT]\n"
                 "NOTE: same-host IPC needs both ranks to see BOTH GPUs, e.g. "
                 "CUDA_VISIBLE_DEVICES=0,1 with --gpu=0 / --gpu=1 (restricting "
                 "each rank to one visible device breaks the peer device "
                 "numbering and the IPC path fails to open).\n");
    return 1;
  }

  int rank = (role == "server") ? 0 : 1;
  int gpu = get_int_arg(argc, argv, "--gpu", rank);
  int port = get_int_arg(argc, argv, "--exchanger-port", 16998);
  std::string xip = get_arg(argc, argv, "--exchanger-ip", "127.0.0.1");

  std::printf("[e2e] %s rank=%d gpu=%d\n", role.c_str(), rank, gpu);
  GPU_RT_CHECK(gpuSetDevice(gpu));

  SprayExecutorConfig cfg;
  cfg.gpu_id = gpu;
  cfg.rank = rank;
  cfg.world_size = 2;
  cfg.exchanger_ip = (rank == 0) ? "0.0.0.0" : xip;
  cfg.exchanger_port = port;
  cfg.local_id = gpu;
  auto ex = SprayExecutor::create(cfg);

  // 4MB in-place allreduce via IPC
  constexpr size_t kBufBytes = 4ULL * 1024 * 1024;  // 4 MB
  void *d_in = nullptr, *d_out = nullptr, *d_scr = nullptr;
  GPU_RT_CHECK(gpuMalloc(&d_in, kBufBytes));
  GPU_RT_CHECK(gpuMalloc(&d_out, kBufBytes));
  GPU_RT_CHECK(gpuMalloc(&d_scr, kBufBytes));

  std::vector<float> host_in(kBufBytes / sizeof(float), (float)(rank + 1));
  GPU_RT_CHECK(
      gpuMemcpy(d_in, host_in.data(), kBufBytes, gpuMemcpyHostToDevice));
  // Zero the output buffer with a kernel (not cudaMemset): the worker is
  // already resident, and the copy-engine memset's writes can still be
  // draining when the worker reduce read-modify-writes the buffer.
  UKernel::Device::zero_device_buffer(d_out, kBufBytes);

  CollectiveConfig ar;
  ar.nranks = 2;
  ar.rank = rank;
  ar.input_bytes = kBufBytes;
  ar.output_bytes = kBufBytes;
  ar.tile_bytes = 65536;
  ar.kind = CollKind::AllReduceRing;

  ex->prepare(ar, d_in, d_out);

  std::printf("[e2e] submit AllReduce 4MB non-inplace...\n");
  auto h = ex->submit(ar, d_in, d_out);

  bool passed = false;
  for (int p = 0; p < 120; ++p) {
    ex->wait(h, std::chrono::milliseconds(500));
    auto st = ex->status(h);
    if (st == CollectiveOpStatus::Completed) {
      float result_in, result_out;
      GPU_RT_CHECK(
          gpuMemcpy(&result_in, d_in, sizeof(float), gpuMemcpyDeviceToHost));
      GPU_RT_CHECK(
          gpuMemcpy(&result_out, d_out, sizeof(float), gpuMemcpyDeviceToHost));
      float expected = 3.0f;
      std::printf("[e2e] in=%.1f out=%.1f\n", result_in, result_out);
      if (std::abs(result_out - expected) < 1e-5f) {
        std::printf("[e2e] PASSED (%.1f)\n", result_out);
        passed = true;
      } else {
        std::printf("[e2e] FAILED (got %.1f want %.1f)\n", result_out,
                    expected);
      }
      break;
    }
  }
  if (!passed) std::printf("[e2e] not completed\n");
  ex->release(h);
  std::printf("[e2e] done\n");
  fflush(stdout);
  return passed ? 0 : 1;
}
