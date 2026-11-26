#include "cpu_executor.hpp"
#include "packet_pipeline.hpp"
#include "signal.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <cuda_bf16.h>
#include <sched.h>
using namespace uccl_lite::cpu_router;
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void fill(uint16_t* p, size_t n) {
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += blockDim.x * gridDim.x)
    p[i] = 0x3f80;
}
__global__ void verify(uint16_t const* p, size_t n, uint16_t expected,
                       int* errors) {
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += blockDim.x * gridDim.x)
    if (p[i] != expected) atomicAdd(errors, 1);
}
int main() {
  try {
    int available = 0;
    check(cudaGetDeviceCount(&available));
    if (available < 2)
      throw std::runtime_error("primitive checks require at least two GPUs");
    int ranks = std::min(available, 8);
    int node0 = gpuNumaNode(0), node1 = gpuNumaNode(available / 2);
    int peerAvailable = 0;
    check(cudaDeviceCanAccessPeer(&peerAvailable, 0, 1));
    for (auto backend : {TransportBackend::GpuKernel, TransportBackend::Dma}) {
      auto input = RegisteredMemory::gpu(8194, 0),
           output = RegisteredMemory::gpu(8194, 0),
           peer = RegisteredMemory::gpu(8194, 1);
      auto host = RegisteredMemory::pinned(8194, node0);
      cudaStream_t stream, other;
      check(cudaSetDevice(0));
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      check(cudaSetDevice(1));
      check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
      auto error0 = RegisteredMemory::gpu(sizeof(int), 0),
           error1 = RegisteredMemory::gpu(sizeof(int), 1);
      check(cudaSetDevice(0));
      check(cudaMemset(error0.data(), 0, 4));
      Channel channel({backend, 0, 32, 256, 8});
      Channel peerChannel({backend, 1, 32, 256, 8});
      check(cudaSetDevice(0));
      fill<<<16, 256, 0, stream>>>(static_cast<uint16_t*>(input.data()) + 1,
                                   4096);
      auto put = channel.put(host.span().subspan(2, 8192),
                             input.span().subspan(2, 8192), stream);
      auto get = channel.get(output.span().subspan(2, 8192),
                             host.span().subspan(2, 8192), stream);
      verify<<<16, 256, 0, stream>>>(static_cast<uint16_t*>(output.data()) + 1,
                                     4096, 0x3f80,
                                     static_cast<int*>(error0.data()));
      ChannelCompletion direct;
      if (peerAvailable) {
        direct = channel.put(peer.span().subspan(2, 8192),
                             input.span().subspan(2, 8192), stream);
      } else {
        put.waitOn(1, other);
        direct = peerChannel.put(peer.span().subspan(2, 8192),
                                 host.span().subspan(2, 8192), other);
      }
      check(cudaSetDevice(1));
      check(cudaMemset(error1.data(), 0, 4));
      direct.waitOn(1, other);
      verify<<<16, 256, 0, other>>>(static_cast<uint16_t*>(peer.data()) + 1,
                                    4096, 0x3f80,
                                    static_cast<int*>(error1.data()));
      check(cudaStreamSynchronize(other));
      check(cudaSetDevice(0));
      check(cudaStreamSynchronize(stream));
      int wrong = 0, second = 0;
      check(cudaMemcpy(&wrong, error0.data(), 4, cudaMemcpyDeviceToHost));
      check(cudaMemcpy(&second, error1.data(), 4, cudaMemcpyDeviceToHost));
      if (wrong || second || !put.ready() || !get.ready())
        throw std::runtime_error("channel producer/copy/consumer failed");
      check(cudaStreamDestroy(stream));
      check(cudaSetDevice(1));
      check(cudaStreamDestroy(other));
    }
    auto a = RegisteredMemory::pinned(4096, node0),
         b = RegisteredMemory::pinned(4096, node1),
         local = RegisteredMemory::pinned(4096, node0),
         remote = RegisteredMemory::pinned(4096, node1),
         result = RegisteredMemory::pinned(4096, node0);
    for (int i = 0; i < 2048; ++i) {
      static_cast<uint16_t*>(a.data())[i] = 0x3f80;
      static_cast<uint16_t*>(b.data())[i] = 0x4000;
    }
    cpu_set_t permitted;
    if (sched_getaffinity(0, sizeof(permitted), &permitted))
      throw std::runtime_error("CPU affinity query failed");
    std::vector<int> cpus;
    for (int id = 0; id < CPU_SETSIZE && cpus.size() < 2; ++id)
      if (CPU_ISSET(id, &permitted)) cpus.push_back(id);
    if (cpus.empty()) throw std::runtime_error("no permitted CPU");
    if (cpus.size() == 1) cpus.push_back(cpus[0]);
    CpuExecutor cpu(cpus, 2);
    MemorySpan aa[] = {a.span(), a.span()}, bb[] = {b.span(), b.span()};
    auto l = cpu.enqueueSum(0, aa, 2, local.span());
    auto r = cpu.enqueueSum(1, bb, 2, remote.span());
    l.wait();
    r.wait();
    for (int i = 0; i < 257; ++i)
      cpu.enqueueMerge(0, local.span(), remote.span(), result.span()).wait();
    for (int i = 0; i < 2048; ++i)
      if (static_cast<uint16_t*>(result.data())[i] != 0x40c0)
        throw std::runtime_error("BF16 local/merge failed");
    PacketOwnership packets[2];
    uint32_t payload[2][16]{};
    std::atomic<int> wrong{0};
    std::vector<std::thread> readers;
    for (int reader = 0; reader < 3; ++reader)
      readers.emplace_back([&, reader] {
        for (Generation g = 1; g <= 300; ++g) {
          auto p = packets[g & 1].view(7);
          p.ready.wait(g);
          for (auto x : payload[g & 1])
            if (x != g) ++wrong;
          p.consume(reader, g);
        }
      });
    std::thread releaser([&] {
      for (Generation g = 1; g <= 300; ++g) {
        auto p = packets[g & 1].view(7);
        p.ready.wait(g);
        p.release(g);
      }
    });
    Generation last[2]{};
    for (Generation g = 1; g <= 300; ++g) {
      auto p = packets[g & 1].view(7);
      p.acquire(last[g & 1]);
      for (auto& x : payload[g & 1]) x = g;
      p.publish(g);
      last[g & 1] = g;
    }
    for (auto& t : readers) t.join();
    releaser.join();
    if (wrong.load())
      throw std::runtime_error("packet multiple-reader ownership failed");
    for (auto backend : {TransportBackend::GpuKernel, TransportBackend::Dma}) {
      PacketPipelineOptions o;
      o.backend = backend;
      o.maxTensorBytes = 65536;
      PacketPipeline pipeline(o);
      std::vector<RegisteredMemory> outputs(ranks), errors(ranks);
      std::vector<cudaStream_t> streams(ranks);
      for (int rank = 0; rank < ranks; ++rank) {
        outputs[rank] = RegisteredMemory::gpu(128, rank);
        errors[rank] = RegisteredMemory::gpu(4, rank);
        check(cudaSetDevice(rank));
        check(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
        check(cudaMemset(errors[rank].data(), 0, 4));
      }
      for (int call = 0; call < 150; ++call) {
        PacketPipeline::Binding bind(ranks);
        for (int rank = 0; rank < ranks; ++rank) {
          auto input = RegisteredMemory::gpu(128, rank);
          bind.inputs[rank] = input.span();
          bind.outputs[rank] = outputs[rank].span();
          bind.devices[rank] = rank;
          bind.streams[rank] = streams[rank];
          check(cudaSetDevice(rank));
          fill<<<1, 64, 0, streams[rank]>>>(
              static_cast<uint16_t*>(input.data()), 64);
        }
        pipeline.enqueue(PacketProgram::allReduce(64, ranks), bind);
        for (int rank = 0; rank < ranks; ++rank) {
          check(cudaSetDevice(rank));
          verify<<<1, 64, 0, streams[rank]>>>(
              static_cast<uint16_t*>(outputs[rank].data()), 64,
              __bfloat16_as_ushort(__float2bfloat16(float(ranks))),
              static_cast<int*>(errors[rank].data()));
        }
      }
      for (int rank = 0; rank < ranks; ++rank) {
        check(cudaSetDevice(rank));
        check(cudaStreamSynchronize(streams[rank]));
        int wrong;
        check(
            cudaMemcpy(&wrong, errors[rank].data(), 4, cudaMemcpyDeviceToHost));
        if (wrong)
          throw std::runtime_error("registered binding lifetime failed");
        check(cudaStreamDestroy(streams[rank]));
      }
    }
    std::puts(
        "PASS: kernel/DMA channels, peer transfer, BF16 CPU tasks, 300 "
        "multi-reader generations, "
        "and owned program bindings across ring reuse");
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
