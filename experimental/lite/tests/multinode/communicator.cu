#include "uccl_lite/collectives.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <mpi.h>
#ifdef UCCL_LITE_TEST_NCCL_ABI
#include <cstring>
#include <nccl.h>
class UcclLiteNcclAdapter {
  ncclComm_t comm_{};
  static void check(ncclResult_t r) {
    if (r != ncclSuccess) throw std::runtime_error(ncclGetErrorString(r));
  }

 public:
  static uccl_lite::MultinodeId createUniqueId() {
    ncclUniqueId id;
    check(ncclGetUniqueId(&id));
    uccl_lite::MultinodeId result{};
    static_assert(sizeof(id) == result.size());
    std::memcpy(result.data(), &id, sizeof(id));
    return result;
  }
  UcclLiteNcclAdapter(uccl_lite::MultinodeId bytes, int rank, int, int, int,
                      size_t) {
    ncclUniqueId id;
    std::memcpy(&id, bytes.data(), sizeof(id));
    check(ncclCommInitRank(&comm_, 8, id, rank));
  }
  ~UcclLiteNcclAdapter() { ncclCommDestroy(comm_); }
  void enqueue(uccl_lite::CollectiveCall const& call) {
    auto const& p = call.ports[0];
    if (call.collective == uccl_lite::Collective::AllGather)
      check(ncclAllGather(p.input, p.output, call.count, ncclFloat32, comm_,
                          p.stream));
    else if (call.collective == uccl_lite::Collective::ReduceScatter)
      check(ncclReduceScatter(p.input, p.output, call.count, ncclFloat32,
                              ncclSum, comm_, p.stream));
    else
      check(ncclAllReduce(p.input, p.output, call.count, ncclFloat32, ncclSum,
                          comm_, p.stream));
  }
};
using TestCommunicator = UcclLiteNcclAdapter;
#else
using TestCommunicator = uccl_lite::MultinodeCommunicator;
#endif
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
struct PlacementSnapshot {
  cpu_set_t cpus{};
  int mode = 0;
  unsigned long nodes[16]{};
  PlacementSnapshot() {
    if (sched_getaffinity(0, sizeof(cpus), &cpus) ||
        syscall(SYS_get_mempolicy, &mode, nodes, 1024, nullptr, 0))
      throw std::runtime_error("placement snapshot failed");
  }
  bool operator==(PlacementSnapshot const& other) const {
    return mode == other.mode &&
           !std::memcmp(&cpus, &other.cpus, sizeof(cpus)) &&
           !std::memcmp(nodes, other.nodes, sizeof(nodes));
  }
};
using namespace uccl_lite;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct Job {
  CollectiveCall call;
  float *a = nullptr, *b = nullptr;
  std::vector<float> input, output;
  size_t ni, no, guard, sourceOffset, targetOffset;
  int generation;
  ~Job() {
    if (a) cudaFree(a);
    if (b) cudaFree(b);
  }
};
__global__ void delay() {
  auto begin = clock64();
  while (clock64() - begin < 1000000) {
  }
}
float pattern(size_t i) { return float(int(i % 31) - 15) / 16; }
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  std::string phase = "setup";
  try {
    if (ranks != 8)
      throw std::runtime_error("requires eight processes on two hosts");
    int device = rank % 4;
    check(cudaSetDevice(device));
    cudaStream_t streams[2];
    for (auto& stream : streams)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    MultinodeId id{};
    std::vector<std::unique_ptr<Job>> jobs;
    for (int generation = 0; generation < 3; ++generation)
      for (size_t full : {128UL, 512UL, 16384UL, 131072UL, 1048576UL})
        for (auto op : {Collective::AllReduce, Collective::AllGather,
                        Collective::ReduceScatter}) {
          auto job = std::make_unique<Job>();
          auto& j = *job;
          j.generation = generation;
          j.call.collective = op;
          j.call.dtype = ScalarType::Float32;
          j.call.count = full / 4 / (op == Collective::AllReduce ? 1 : 8);
          j.call.inPlace = rank % 2;
          j.ni = j.call.count * (op == Collective::ReduceScatter ? 8 : 1);
          j.no = j.call.count * (op == Collective::AllGather ? 8 : 1);
          j.guard = 128 + rank + 1;
          j.sourceOffset = j.call.inPlace && op == Collective::AllGather
                               ? rank * j.call.count
                               : 0;
          j.targetOffset = j.call.inPlace && op == Collective::ReduceScatter
                               ? rank * j.call.count
                               : 0;
          size_t extent = std::max(j.ni, j.no) + 2 * j.guard;
          j.input.assign(extent, -12345);
          j.output.assign(extent, -12345);
          for (size_t i = 0; i < j.ni; ++i)
            j.input[j.guard + j.sourceOffset + i] =
                (rank + 1) * pattern(i) + generation / 32.f;
          check(cudaMalloc(&j.a, extent * sizeof(float)));
          if (!j.call.inPlace) check(cudaMalloc(&j.b, extent * sizeof(float)));
          j.call.ports = {
              {j.a + j.guard + j.sourceOffset,
               (j.call.inPlace ? j.a : j.b) + j.guard + j.targetOffset,
               streams[jobs.size() % 2], device}};
          jobs.push_back(std::move(job));
        }
#ifdef UCCL_LITE_TEST_EXTERNAL_ORDER
    cudaEvent_t externalDone;
    check(cudaEventCreateWithFlags(&externalDone, cudaEventDisableTiming));
#endif
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
      if (rank == 0) id = TestCommunicator::createUniqueId();
      MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      TestCommunicator comm(id, rank, device, device, device / 2, 1048576
#ifdef UCCL_LITE_TEST_DOUBLE_BUFFER
                            ,
                            true
#endif
      );
#ifdef UCCL_LITE_TEST_NCCL_ABI
      // Empty NCCL collectives add no communication to a saved graph. They
      // must not disable later eligible eager calls or advance route state.
      phase = "pre-existing empty graph";
      auto empty = jobs[0]->call;
      empty.count = 0;
      cudaGraph_t emptyGraph;
      cudaGraphExec_t emptyExecutable;
      check(
          cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal));
      for (auto op : {Collective::AllReduce, Collective::AllGather,
                      Collective::ReduceScatter}) {
        empty.collective = op;
        comm.enqueue(empty);
      }
      check(cudaStreamEndCapture(streams[0], &emptyGraph));
      check(cudaGraphInstantiate(&emptyExecutable, emptyGraph, nullptr, nullptr,
                                 0));
      check(cudaGraphLaunch(emptyExecutable, streams[0]));
#endif
#ifndef UCCL_LITE_TEST_NCCL_ABI
      if (comm.initialized())
        throw std::runtime_error("constructor prepared transport");
      try {
        comm.enqueue(jobs[0]->call);
        throw std::runtime_error("unprepared enqueue accepted");
      } catch (std::logic_error const&) {
      }
      phase = "prepare";
      PlacementSnapshot beforePrepare;
      comm.prepare(jobs[0]->call);
      if (!(beforePrepare == PlacementSnapshot()))
        throw std::runtime_error("prepare changed caller CPU/NUMA placement");
      if (!comm.initialized())
        throw std::runtime_error("prepare did not initialize transport");
#endif
      for (auto& job : jobs) {
        auto& j = *job;
        auto stream = j.call.ports[0].stream;
        check(cudaMemcpyAsync(j.a, j.input.data(),
                              j.input.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
        if (j.b)
          check(cudaMemcpyAsync(j.b, j.output.data(),
                                j.output.size() * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
        if (rank == 7 && j.generation == 1) delay<<<1, 1, 0, stream>>>();
        phase = "enqueue " + std::to_string(int(j.call.collective)) + "/" +
                std::to_string(j.call.count);
#ifdef UCCL_LITE_TEST_EXTERNAL_ORDER
        check(cudaStreamWaitEvent(stream, externalDone, 0));
        comm.enqueueOrdered(j.call, externalDone);
        check(cudaEventRecord(externalDone, stream));
#else
#ifdef UCCL_LITE_TEST_NCCL_ABI
        bool changeDevice = j.generation == 1 &&
                            j.call.collective == Collective::AllReduce &&
                            j.call.count == 4096;
        if (changeDevice) check(cudaSetDevice((device + 1) % 4));
#endif
        PlacementSnapshot beforeCall;
        comm.enqueue(j.call);
        if (!(beforeCall == PlacementSnapshot()))
          throw std::runtime_error(
              "collective changed caller CPU/NUMA placement");
#ifdef UCCL_LITE_TEST_NCCL_ABI
        if (changeDevice) {
          int current = -1;
          check(cudaGetDevice(&current));
          if (current != (device + 1) % 4)
            throw std::runtime_error("caller CUDA device was not restored");
          check(cudaSetDevice(device));
        }
#endif
#endif
      }
#ifdef UCCL_LITE_TEST_NCCL_ABI
      check(cudaGraphLaunch(emptyExecutable, streams[0]));
#endif
      phase = "drain";
      check(cudaDeviceSynchronize());
#ifdef UCCL_LITE_TEST_NCCL_ABI
      check(cudaGraphExecDestroy(emptyExecutable));
      check(cudaGraphDestroy(emptyGraph));
#endif
      phase = "verify";
      for (auto& job : jobs) {
        auto& j = *job;
        check(cudaMemcpy(j.output.data(), j.call.inPlace ? j.a : j.b,
                         j.output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < j.guard; ++i)
          if (j.output[i] != -12345 ||
              j.output[j.output.size() - 1 - i] != -12345)
            throw std::runtime_error("output guard modified");
        for (size_t i = 0; i < j.no; ++i) {
          auto op = j.call.collective;
          float expected =
              op == Collective::AllGather
                  ? (i / j.call.count + 1) * pattern(i % j.call.count) +
                        j.generation / 32.f
                  : 36 * pattern(i + (op == Collective::ReduceScatter
                                          ? rank * j.call.count
                                          : 0)) +
                        j.generation / 4.f;
          float got = j.output[j.guard + j.targetOffset + i];
          if (got != expected) {
            std::fprintf(
                stderr,
                "rank %d op %d count %zu gen %d index %zu got %g expected %g\n",
                rank, int(op), j.call.count, j.generation, i, got, expected);
            throw std::runtime_error("full-buffer oracle mismatch");
          }
        }
      }
#ifndef UCCL_LITE_TEST_NCCL_ABI
      auto invalid = jobs[0]->call;
      invalid.count = 33;
      try {
        comm.enqueue(invalid);
        throw std::runtime_error("invalid count accepted");
      } catch (std::invalid_argument const&) {
      }
      phase = "graph rejection";
      cudaGraph_t graph;
      check(
          cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal));
      invalid = jobs[0]->call;
      invalid.ports[0].stream = streams[0];
      try {
        comm.enqueue(invalid);
        throw std::runtime_error("capture accepted");
      } catch (std::invalid_argument const&) {
      }
      check(cudaStreamEndCapture(streams[0], &graph));
      check(cudaGraphDestroy(graph));
#else
      phase = "NCCL grouped/graph fallback";
      auto& first = *jobs[0];
      auto stream = first.call.ports[0].stream;
      if (lifetime == 1) {
        check(cudaMemcpyAsync(first.a, first.input.data(),
                              first.input.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
        if (ncclGroupStart() != ncclSuccess || ncclGroupStart() != ncclSuccess)
          throw std::runtime_error("group start failed");
        comm.enqueue(first.call);
        if (ncclGroupEnd() != ncclSuccess || ncclGroupEnd() != ncclSuccess)
          throw std::runtime_error("group end failed");
        check(cudaStreamSynchronize(stream));
      }
      cudaGraph_t graph;
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      comm.enqueue(first.call);
      check(cudaStreamEndCapture(stream, &graph));
      cudaGraphExec_t executable;
      check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
      for (int replay = 0; replay < 3; ++replay) {
        check(cudaMemcpyAsync(first.a, first.input.data(),
                              first.input.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        std::vector<float> result(first.no);
        check(cudaMemcpy(result.data(), first.call.ports[0].output,
                         first.no * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < first.no; ++i)
          if (result[i] != 36 * pattern(i))
            throw std::runtime_error("NCCL graph oracle mismatch");
      }
      check(cudaGraphExecDestroy(executable));
      check(cudaGraphDestroy(graph));
#endif
      MPI_Barrier(MPI_COMM_WORLD);
    }
#ifdef UCCL_LITE_TEST_NCCL_ABI
    // Capture a non-admitted count before any eligible call. Replaying this
    // existing graph later cannot be intercepted, so the communicator must
    // remain entirely on native NCCL, even when eligible eager calls follow.
    if (rank == 0) id = TestCommunicator::createUniqueId();
    MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    {
      phase = "pre-existing unsupported graph";
      TestCommunicator comm(id, rank, device, device, device / 2, 1048576);
      float *source = nullptr, *output = nullptr;
      check(cudaMalloc(&source, 33 * sizeof(float)));
      check(cudaMalloc(&output, 33 * sizeof(float)));
      std::vector<float> input(33, rank + 1), result(33);
      auto stream = streams[0];
      check(cudaMemcpyAsync(source, input.data(), 33 * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
      CollectiveCall call;
      call.dtype = ScalarType::Float32;
      call.count = 33;
      call.ports = {{source, output, stream, device}};
      comm.enqueue(call);
      check(cudaStreamSynchronize(stream));
      cudaGraph_t graph;
      cudaGraphExec_t executable;
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      comm.enqueue(call);
      check(cudaStreamEndCapture(stream, &graph));
      check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
      check(cudaGraphLaunch(executable, stream));
      check(cudaStreamSynchronize(stream));
      for (auto op : {Collective::AllReduce, Collective::AllGather,
                      Collective::ReduceScatter}) {
        auto it = std::find_if(jobs.begin(), jobs.end(), [&](auto const& j) {
          return j->generation == 0 && j->call.collective == op &&
                 j->call.count == 4096;
        });
        auto& j = **it;
        check(cudaMemcpyAsync(j.a, j.input.data(),
                              j.input.size() * sizeof(float),
                              cudaMemcpyHostToDevice, j.call.ports[0].stream));
        comm.enqueue(j.call);
        check(cudaStreamSynchronize(j.call.ports[0].stream));
      }
      check(cudaGraphLaunch(executable, stream));
      check(cudaStreamSynchronize(stream));
      check(cudaMemcpy(result.data(), output, 33 * sizeof(float),
                       cudaMemcpyDeviceToHost));
      if (!std::all_of(result.begin(), result.end(),
                       [](float v) { return v == 36; }))
        throw std::runtime_error("pre-existing graph replay failed");
      check(cudaGraphExecDestroy(executable));
      check(cudaGraphDestroy(graph));
      check(cudaFree(source));
      check(cudaFree(output));
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
#ifdef UCCL_LITE_TEST_EXTERNAL_ORDER
    check(cudaEventDestroy(externalDone));
#endif
    jobs.clear();
    for (auto stream : streams) check(cudaStreamDestroy(stream));
    if (rank == 0)
      std::puts(
          "PASS: network shared library, 90 mixed FP32 AR/AG/RS calls per "
          "rank, guards, changing sizes/streams, delayed producer, in-place "
          "layouts, two communicator lifetimes, backend prepare/capture checks "
          "or NCCL grouped/graph fallback");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "rank %d phase %s: %s\n", rank, phase.c_str(),
                 error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}
