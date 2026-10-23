"""Exercise RDMA request descriptor ownership through submission and release."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

STUBS = r"""
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#define likely(value) (value)
#define UCCL_LOG(level) std::cerr
struct Keys { void copy_from(void const*) {} };
enum class MemoryType { GPU };
struct RegMemBlock { void* addr{}; size_t size{}; MemoryType type{}; int mr_array{}; };
struct RemoteMemInfo { uint64_t addr{}; size_t length{}; MemoryType type{}; Keys rkey_array; };
enum class SendType { Read, Write };
enum class ReqType { ReqRead, ReqWrite };
struct UcclRequest { ReqType type{}; int engine_idx{}; int peer_id{}; };
struct P2PMhandle {
  bool compress_only = false;
  void const* registered_addr = nullptr;
  size_t registered_len = 0;
  std::shared_ptr<int> compress_ctx;
  int mr_array = 0;
};
struct FifoItem { uint64_t addr = 32; size_t size = 64; char padding[4]{}; };
struct Conn { struct { int peer_id = 7; } uccl_conn_id_; };
struct RDMASendRequest {
  std::shared_ptr<RegMemBlock> local_mem;
  std::shared_ptr<RemoteMemInfo> remote_mem;
  std::shared_ptr<int> compress_ctx;
  bool compress_only = false;
  int to_peer_id = 0;
  SendType send_type{};
  bool need_signaled = false;
  RDMASendRequest(std::shared_ptr<RegMemBlock> local,
                  std::shared_ptr<RemoteMemInfo> remote)
      : local_mem(std::move(local)), remote_mem(std::move(remote)) {}
};
struct SendConnection {
  static constexpr int kPostError = -2;
  int mode = 0;
  int attempts = 0;
  std::weak_ptr<RDMASendRequest> request;
  std::weak_ptr<RegMemBlock> local;
  std::weak_ptr<RemoteMemInfo> remote;
  std::vector<std::shared_ptr<RDMASendRequest>> pending;
  int64_t post_write_or_read(std::shared_ptr<RDMASendRequest> const& req) {
    ++attempts;
    request = req;
    local = req->local_mem;
    remote = req->remote_mem;
    if (mode == 2) return kPostError;
    if (mode == 3 && attempts == 1) return -1;
    if (mode == 1 || mode == 3) pending.push_back(req);
    return 11;
  }
};
struct RDMAEndpoint : SendConnection {
  int64_t write_or_read(std::shared_ptr<RDMASendRequest> const& req) {
    return post_write_or_read(req);
  }
};
"""

SCENARIOS = r"""
void require(bool value, char const* reason) {
  if (!value) throw std::runtime_error(reason);
}
int main() {
  int cases = 0;
  for (bool grouped : {false, true}) {
    for (auto direction : {ReqType::ReqRead, ReqType::ReqWrite}) {
      for (int mode : {0, 1, 2, 3}) {
        if (!grouped && mode == 3) continue;
        for (int round = 0; round < 64; ++round) {
          auto endpoint = std::make_shared<RDMAEndpoint>();
          endpoint->mode = mode;
          Conn connection;
          P2PMhandle memory;
          memory.compress_ctx = std::make_shared<int>(23);
          char source[64]{};
          memory.registered_addr = source;
          memory.registered_len = sizeof(source);
          FifoItem remote;
          UcclRequest operation;
          operation.type = direction;
          int result = grouped
              ? set_request_on_group(endpoint.get(), &connection, &memory,
                                     source + 8, 32, remote, &operation)
              : set_request(endpoint, &connection, &memory,
                            source + 8, 32, remote, &operation);
          require(result == (mode == 2 ? SendConnection::kPostError : 11),
                  "submission result changed");
          if (!endpoint->pending.empty()) {
            auto request = endpoint->pending.front();
            require(request->local_mem->addr == source + 8 &&
                        request->local_mem->size == 32 &&
                        request->remote_mem->addr == remote.addr,
                    "delayed submission lost its descriptors");
            require(!request->compress_ctx, "subrange retained compression");
            auto retained_local = request->local_mem;
            auto retained_remote = request->remote_mem;
            request.reset();
            endpoint->pending.clear();
            require(endpoint->request.expired(), "request participates in an ownership cycle");
            require(!endpoint->local.expired() && !endpoint->remote.expired(),
                    "retained descriptors lost their owner");
            retained_local.reset();
            retained_remote.reset();
          }
          require(endpoint->request.expired() && endpoint->local.expired() &&
                      endpoint->remote.expired(),
                  "request descriptors were retained after the last consumer");
          ++cases;
        }
      }
    }
  }
  std::cout << cases << " request lifecycles passed\n";
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "endpoint_wrapper.h",
    )
    args = parser.parse_args()
    source = args.source.read_text()
    start = source.index("// RDMA-only:")
    end = source.index("// ── Dispatch functions", start)
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "request-lifetime"
        subprocess.run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++17",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                "-x",
                "c++",
                "-",
                "-o",
                str(executable),
            ],
            input=STUBS + source[start:end] + SCENARIOS,
            text=True,
            check=True,
            timeout=60,
        )
        subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    main()
