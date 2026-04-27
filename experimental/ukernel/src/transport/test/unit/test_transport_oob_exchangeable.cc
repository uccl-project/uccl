#include "oob/oob.h"
#include "test.h"
#include "test_utils.h"
#include <cstring>

namespace {

using UKernel::Transport::CommunicatorMeta;
using UKernel::Transport::IpcBufferInfo;
using UKernel::Transport::MR;
using UKernel::Transport::NamedMR;
using UKernel::Transport::NamedMRInfos;
using UKernel::Transport::TcpP2PInfo;
using UKernel::Transport::UCCLP2PInfo;
using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

void test_exchangeable_round_trip() {
  CommunicatorMeta meta{};
  meta.host_id = "host-a";
  meta.ip = "10.0.0.1";
  meta.local_id = 7;
  meta.rdma_capable = true;
  CommunicatorMeta meta_rt;
  std::string encoded;
  require(meta.serialize(encoded), "CommunicatorMeta serialize failed");
  require(meta_rt.deserialize(encoded), "CommunicatorMeta deserialize failed");
  require(meta_rt.host_id == meta.host_id, "CommunicatorMeta host_id mismatch");
  require(meta_rt.ip == meta.ip, "CommunicatorMeta ip mismatch");
  require(meta_rt.local_id == meta.local_id,
          "CommunicatorMeta local_id mismatch");
  require(meta_rt.rdma_capable == meta.rdma_capable,
          "CommunicatorMeta rdma_capable mismatch");

  NamedMRInfos infos{};
  infos.generation = 42;
  infos.entries.push_back(NamedMR{3, MR{1, 0x1000ULL, 256, 0, 11}});
  infos.entries.push_back(NamedMR{7, MR{2, 0x2000ULL, 512, 0, 22}});
  NamedMRInfos infos_rt;
  require(infos.serialize(encoded), "NamedMRInfos serialize failed");
  require(infos_rt.deserialize(encoded), "NamedMRInfos deserialize failed");
  require(infos_rt.generation == infos.generation,
          "NamedMRInfos generation mismatch");
  require(infos_rt.entries.size() == 2, "NamedMRInfos size mismatch");
  require(infos_rt.entries[0].buffer_id == infos.entries[0].buffer_id,
          "NamedMRInfos first buffer_id mismatch");
  require(infos_rt.entries[0].mr.address == infos.entries[0].mr.address,
          "NamedMRInfos first address mismatch");
  require(infos_rt.entries[1].mr.key == infos.entries[1].mr.key,
          "NamedMRInfos second key mismatch");

  UCCLP2PInfo uccl{"127.0.0.1", 12345, 6, 2};
  UCCLP2PInfo uccl_rt;
  require(uccl.serialize(encoded), "UCCLP2PInfo serialize failed");
  require(uccl_rt.deserialize(encoded), "UCCLP2PInfo deserialize failed");
  require(uccl_rt.ip == uccl.ip && uccl_rt.port == uccl.port,
          "UCCLP2PInfo endpoint mismatch");
  require(uccl_rt.dev_idx == uccl.dev_idx && uccl_rt.gpu_idx == uccl.gpu_idx,
          "UCCLP2PInfo device mapping mismatch");

  TcpP2PInfo tcp{"127.0.0.1", 23456};
  TcpP2PInfo tcp_rt;
  require(tcp.serialize(encoded), "TcpP2PInfo serialize failed");
  require(tcp_rt.deserialize(encoded), "TcpP2PInfo deserialize failed");
  require(tcp_rt.ip == tcp.ip && tcp_rt.port == tcp.port,
          "TcpP2PInfo round-trip mismatch");

  IpcBufferInfo ipc{};
  ipc.base_offset = 128;
  ipc.bytes = 4096;
  ipc.device_idx = 3;
  ipc.valid = true;
  auto* handle_bytes = reinterpret_cast<unsigned char*>(&ipc.handle);
  for (size_t i = 0; i < sizeof(gpuIpcMemHandle_t); ++i) {
    handle_bytes[i] = static_cast<unsigned char>(0x80 + (i & 0x0F));
  }

  IpcBufferInfo ipc_rt;
  require(ipc.serialize(encoded), "IpcBufferInfo serialize failed");
  require(ipc_rt.deserialize(encoded), "IpcBufferInfo deserialize failed");
  require(ipc_rt.base_offset == ipc.base_offset,
          "IpcBufferInfo base_offset mismatch");
  require(ipc_rt.bytes == ipc.bytes, "IpcBufferInfo bytes mismatch");
  require(ipc_rt.device_idx == ipc.device_idx,
          "IpcBufferInfo device_idx mismatch");
  require(ipc_rt.valid == ipc.valid, "IpcBufferInfo valid mismatch");
  require(
      std::memcmp(&ipc_rt.handle, &ipc.handle, sizeof(gpuIpcMemHandle_t)) == 0,
      "IpcBufferInfo handle mismatch");
}

}  // namespace

void test_transport_oob_exchangeables() {
  run_case("transport unit", "oob exchangeable round trip",
           test_exchangeable_round_trip);
}
