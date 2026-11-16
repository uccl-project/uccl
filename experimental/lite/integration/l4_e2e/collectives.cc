// Explicit benchmark adapter: four local GPUs, optionally two MPI processes.
#include "uccl_lite/collective_dispatch.hpp"
#include "uccl_lite/collectives/peer_collectives.hpp"
#include "uccl_lite/cpu_router/packet_pipeline.hpp"
#include <array>
#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
namespace {
void ck(cudaError_t x) {
  if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x));
}
void nk(ncclResult_t x) {
  if (x != ncclSuccess) throw std::runtime_error(ncclGetErrorString(x));
}
thread_local char error[1024];
struct Context {
  int node = 0, nodes = 1;
  bool usePeer;
  bool replicas;
  bool usePacket;
  std::unique_ptr<uccl_lite::cpu_router::PacketPipeline> packet;
  std::unique_ptr<uccl_lite::PeerCollectives> peer;
  std::array<ncclComm_t, 4> comm{}, cross{};
  std::array<void*, 4> temporary{}, local{};
  size_t capacity = 0;
  std::set<size_t> prepared;
  Context(int flags)
      : usePeer(flags & 1), replicas(flags & 2), usePacket(flags & 4) {
    bool p = usePeer;
    int init;
    MPI_Initialized(&init);
    if (!init) {
      int provided;
      MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
      if (provided < MPI_THREAD_MULTIPLE)
        throw std::runtime_error("MPI_THREAD_MULTIPLE required");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    if (nodes != 1 && nodes != 2)
      throw std::runtime_error("one or two four-GPU nodes required");
    if (usePacket && nodes == 2 && !replicas)
      throw std::runtime_error("packet AG requires local four-rank sharding");
    if (p) {
      uccl_lite::Topology t;
      t.devices = {0, 1, 2, 3};
      for (int r = 0; r < 4; ++r)
        for (int s = 0; s < 4; ++s) {
          int a = 1;
          if (r != s) ck(cudaDeviceCanAccessPeer(&a, r, s));
          t.peerAccess.push_back(a);
        }
      if (!uccl_lite::PeerCollectives::supported(t))
        throw std::runtime_error("full four-GPU P2P required");
      peer = std::make_unique<uccl_lite::PeerCollectives>(t);
    }
    if (!p || nodes == 2) {
      std::array<ncclUniqueId, 4> ids;
      if (node == 0)
        for (auto& id : ids) nk(ncclGetUniqueId(&id));
      MPI_Bcast(ids.data(), sizeof(ids), MPI_BYTE, 0, MPI_COMM_WORLD);
      nk(ncclGroupStart());
      for (int r = 0; r < 4; ++r) {
        ck(cudaSetDevice(r));
        nk(ncclCommInitRank(&comm[r],
                            replicas ? (p ? 2 : 4) : (p ? nodes : nodes * 4),
                            ids[replicas && !p ? node : (p ? r : 0)],
                            replicas && !p ? r : (p ? node : node * 4 + r)));
      }
      nk(ncclGroupEnd());
    }
    if (replicas && nodes == 2 && !p) {
      std::array<ncclUniqueId, 4> ids;
      if (node == 0)
        for (auto& id : ids) nk(ncclGetUniqueId(&id));
      MPI_Bcast(ids.data(), sizeof(ids), MPI_BYTE, 0, MPI_COMM_WORLD);
      nk(ncclGroupStart());
      for (int r = 0; r < 4; ++r) {
        ck(cudaSetDevice(r));
        nk(ncclCommInitRank(&cross[r], 2, ids[r], node));
      }
      nk(ncclGroupEnd());
    }
  }
  void replicaReduce(size_t n, void** out, void** streams) {
    if (!replicas || nodes != 2) return;
    nk(ncclGroupStart());
    for (int r = 0; r < 4; ++r) {
      ck(cudaSetDevice(r));
      nk(ncclAllReduce(out[r], out[r], n, ncclBfloat16, ncclSum,
                       usePeer ? comm[r] : cross[r], (cudaStream_t)streams[r]));
    }
    nk(ncclGroupEnd());
  }

  uccl_lite::CollectiveCall call(int op, size_t n, void** in, void** out,
                                 void** streams) {
    uccl_lite::CollectiveCall c;
    c.collective = op == 1 ? uccl_lite::Collective::AllGather
                           : uccl_lite::Collective::ReduceScatter;
    c.count = n;
    for (int r = 0; r < 4; ++r) {
      ck(cudaSetDevice(r));
      cudaStreamCaptureStatus capture;
      ck(cudaStreamIsCapturing((cudaStream_t)streams[r], &capture));
      if (capture != cudaStreamCaptureStatusNone)
        throw std::runtime_error("eager only");
      cudaPointerAttributes a{}, b{};
      ck(cudaPointerGetAttributes(&a, in[r]));
      ck(cudaPointerGetAttributes(&b, out[r]));
      if (a.type != cudaMemoryTypeDevice || b.type != cudaMemoryTypeDevice ||
          a.device != r || b.device != r || in[r] == out[r])
        throw std::runtime_error("out-of-place local device buffers required");
      c.ports.push_back({in[r], out[r], (cudaStream_t)streams[r], r});
    }
    return c;
  }
  ~Context() {
    for (int r = 0; r < 4; ++r) {
      ck(cudaSetDevice(r));
      if (comm[r]) ncclCommDestroy(comm[r]);
      if (cross[r]) ncclCommDestroy(cross[r]);
      cudaFree(temporary[r]);
      cudaFree(local[r]);
    }
  }
};
template <class F>
int guard(F f) {
  try {
    f();
    return 0;
  } catch (std::exception const& e) {
    snprintf(error, sizeof(error), "%s", e.what());
    return -1;
  }
}
}  // namespace
extern "C" {
char const* uccl_lite_e2e_error() { return error; }
void* uccl_lite_e2e_create(int p) {
  Context* c = nullptr;
  guard([&] { c = new Context(p); });
  return c;
}
void uccl_lite_e2e_destroy(void* p) { delete static_cast<Context*>(p); }
int uccl_lite_e2e_version() {
  int v;
  ncclGetVersion(&v);
  return v;
}
int uccl_lite_e2e_node(void* p) { return static_cast<Context*>(p)->node; }
int uccl_lite_e2e_nodes(void* p) { return static_cast<Context*>(p)->nodes; }
void uccl_lite_e2e_barrier() { MPI_Barrier(MPI_COMM_WORLD); }
double uccl_lite_e2e_max(double value) {
  double result;
  MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return result;
}
void uccl_lite_e2e_finalize() { MPI_Finalize(); }
int uccl_lite_e2e_prepare(void* p, size_t n, void** in, void** out,
                          void** streams) {
  return guard([&] {
    auto& c = *static_cast<Context*>(p);
    if (!n || n > (32UL << 20))
      throw std::runtime_error("positive shard <=64MiB required");
    if (c.usePacket && n * 8 > (64UL << 20))
      throw std::runtime_error("packet AG prepared extent exceeds 64 MiB");
    for (int r = 0; r < 4; ++r) {
      ck(cudaSetDevice(r));
      ck(cudaDeviceSynchronize());
    }
    auto call = c.call(2, n, in, out, streams);
    if (c.usePeer) {
      if (c.nodes == 2 && !c.replicas) {
        if (n > c.capacity) {
          for (int r = 0; r < 4; ++r) {
            ck(cudaSetDevice(r));
            ck(cudaFree(c.temporary[r]));
            ck(cudaFree(c.local[r]));
            ck(cudaMalloc(&c.temporary[r], n * 16));
            ck(cudaMalloc(&c.local[r], n * 4));
          }
          c.capacity = n;
        }
        call = c.call(2, n * 2, c.temporary.data(), c.local.data(), streams);
      }
      c.peer->prepare(call);
      if (c.usePacket && !c.packet) {
        uccl_lite::cpu_router::PacketPipelineOptions options;
        options.maxTensorBytes = 64UL << 20;
        options.packetBytes = 4UL << 20;
        options.workersPerNode = 8;
        options.transportBlocks = 248;
        options.reduction = uccl_lite::cpu_router::ReductionPlacement::Gpu;
        options.gpuProgress = true;
        options.agPerSourceProgress = true;
        c.packet =
            std::make_unique<uccl_lite::cpu_router::PacketPipeline>(options);
        std::array<uccl_lite::cpu_router::PacketPort, 4> ports;
        for (int r = 0; r < 4; ++r)
          ports[r] = {out[r], in[r], (cudaStream_t)streams[r], r};
        c.packet->prepare(ports);
      }
    }
    c.prepared.insert(n);
  });
}
int uccl_lite_e2e_enqueue(void* p, int op, size_t n, void** in, void** out,
                          void** streams) {
  return guard([&] {
    auto& c = *static_cast<Context*>(p);
    if ((op != 1 && op != 2) || !c.prepared.count(n))
      throw std::runtime_error("prepare exact AG/RS extent first");
    auto call = c.call(op, n, in, out, streams);
    if (!c.usePeer) {
      nk(ncclGroupStart());
      for (int r = 0; r < 4; ++r) {
        ck(cudaSetDevice(r));
        if (op == 1)
          nk(ncclAllGather(in[r], out[r], n, ncclBfloat16, c.comm[r],
                           call.ports[r].stream));
        else
          nk(ncclReduceScatter(in[r], out[r], n, ncclBfloat16, ncclSum,
                               c.comm[r], call.ports[r].stream));
      }
      nk(ncclGroupEnd());
      if (op == 2) c.replicaReduce(n, out, streams);
      return;
    }
    if (c.nodes == 1 || c.replicas) {
      if (c.usePacket && op == 1) {
        std::array<uccl_lite::cpu_router::PacketPort, 4> ports;
        for (int r = 0; r < 4; ++r)
          ports[r] = {in[r], out[r], (cudaStream_t)streams[r], r};
        c.packet->enqueue(uccl_lite::cpu_router::PacketProgram::allGather(n, 4),
                          ports);
      } else
        c.peer->enqueue(call);
      if (op == 2) c.replicaReduce(n, out, streams);
      return;
    }
    // Interleaved order [local-rank][node][shard] makes local AG/RS contiguous.
    if (op == 1) {
      nk(ncclGroupStart());
      for (int r = 0; r < 4; ++r) {
        ck(cudaSetDevice(r));
        nk(ncclAllGather(in[r], c.local[r], n, ncclBfloat16, c.comm[r],
                         call.ports[r].stream));
      }
      nk(ncclGroupEnd());
      c.peer->enqueue(
          c.call(1, n * 2, c.local.data(), c.temporary.data(), streams));
      for (int r = 0; r < 4; ++r) {
        ck(cudaSetDevice(r));
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 2; ++j)
            ck(cudaMemcpyAsync((char*)out[r] + (j * 4 + i) * n * 2,
                               (char*)c.temporary[r] + (i * 2 + j) * n * 2,
                               n * 2, cudaMemcpyDeviceToDevice,
                               call.ports[r].stream));
      }
    } else {
      for (int r = 0; r < 4; ++r) {
        ck(cudaSetDevice(r));
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 2; ++j)
            ck(cudaMemcpyAsync((char*)c.temporary[r] + (i * 2 + j) * n * 2,
                               (char*)in[r] + (j * 4 + i) * n * 2, n * 2,
                               cudaMemcpyDeviceToDevice, call.ports[r].stream));
      }
      c.peer->enqueue(
          c.call(2, n * 2, c.temporary.data(), c.local.data(), streams));
      nk(ncclGroupStart());
      for (int r = 0; r < 4; ++r) {
        ck(cudaSetDevice(r));
        nk(ncclReduceScatter(c.local[r], out[r], n, ncclBfloat16, ncclSum,
                             c.comm[r], call.ports[r].stream));
      }
      nk(ncclGroupEnd());
    }
  });
}
}
