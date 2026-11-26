// Explicit FP32 host reductions on two four-GPU NUMA hosts. Eager only.
#include "uccl_lite/multinode_host_common.cuh"
#include <chrono>
#include <memory>

namespace {
using namespace uccl_lite::host_collective;
struct Config {
  size_t begin = 1024, end = 1UL << 30, chunk = 262144;
  int warmups = 10, iterations = 10, repeats = 3, delay = -1;
  bool rs = false, nccl = true, pair = false;
};
Config parse(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::puts(
          "multinode_host_reduce --op allreduce|reducescatter "
          "-b BYTES -e BYTES -k CHUNK_BYTES -w WARMUPS -n CALLS "
          "-r ROUNDS --nccl 0|1 --pair-reduce 0|1 --delay-consumer-rank RANK");
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("missing value");
    std::string value = argv[i];
    if (key == "--op") {
      if (value != "allreduce" && value != "reducescatter")
        throw std::invalid_argument("unsupported operation");
      c.rs = value == "reducescatter";
    } else if (key == "-b")
      c.begin = std::stoull(value);
    else if (key == "-e")
      c.end = std::stoull(value);
    else if (key == "-k")
      c.chunk = std::stoull(value);
    else if (key == "-w")
      c.warmups = std::stoi(value);
    else if (key == "-n")
      c.iterations = std::stoi(value);
    else if (key == "-r")
      c.repeats = std::stoi(value);
    else if (key == "--delay-consumer-rank")
      c.delay = std::stoi(value);
    else if (key == "--pair-reduce" && (value == "0" || value == "1"))
      c.pair = value == "1";
    else if (key == "--nccl" && (value == "0" || value == "1"))
      c.nccl = value == "1";
    else
      throw std::invalid_argument("unknown option/value: " + key);
  }
  if (c.begin < 1024 || c.end < c.begin || c.end > (1UL << 30) ||
      c.begin % 128 || c.end % 128 || c.chunk < 128 || c.chunk % 128 ||
      c.chunk > (16UL << 20) || c.warmups < 1 || c.iterations < 1 ||
      c.repeats < 1 || c.delay < -1 || c.delay >= 8)
    throw std::invalid_argument("invalid sizes or repetitions");
  return c;
}
__global__ void reducePair(float const* a, float const* b, float* out,
                           size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    out[i] = a[i] + b[i];
}
__global__ void download(float const* source, float* output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = source[i];
}
__global__ void verifyReduction(float const* output, unsigned* wrong,
                                size_t count, size_t offset, int generation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    float expected = float(36 + 64 * generation) * float(1 + (offset + i) % 97);
    if (output[i] != expected) atomicAdd(wrong, 1);
  }
}
struct Piece {
  HostBlock own, peer, remote;
  size_t offset = 0, bytes = 0;
  int owner = 0, destination = 0, phase = 0;
  bool active = false;
  MPI_Request readySend = MPI_REQUEST_NULL, readyRecv = MPI_REQUEST_NULL;
  MPI_Request netSend = MPI_REQUEST_NULL, netRecv = MPI_REQUEST_NULL;
  MPI_Request done = MPI_REQUEST_NULL, ack = MPI_REQUEST_NULL;
};
bool completed(MPI_Request& request) {
  int flag = 0;
  check(MPI_Test(&request, &flag, MPI_STATUS_IGNORE));
  return flag;
}
int run(Config const& cfg, int rank, MPI_Comm local) {
  int group = rank % 2, node = rank / 2;
  std::array<int, 2> physical{2 * group, 2 * group + 1};
  int available;
  check(cudaGetDeviceCount(&available));
  if (available != 4) throw std::runtime_error("requires four GPUs per host");
  std::array<std::unique_ptr<Device>, 2> devices;
  for (int g = 0; g < 2; ++g)
    devices[g] = std::make_unique<Device>(physical[g], 2 * rank + g, cfg.end,
                                          cfg.rs ? cfg.end / 8 : cfg.end);
  size_t capacity = std::min(cfg.chunk, cfg.end / 8);
  std::array<float*, 2> partial{};
  if (cfg.pair)
    for (int g = 0; g < 2; ++g) {
      check(cudaSetDevice(physical[g]));
      int access = 0;
      check(cudaDeviceCanAccessPeer(&access, physical[g], physical[g ^ 1]));
      if (!access)
        throw std::runtime_error("pair reduction requires CUDA peer access");
      check(cudaDeviceEnablePeerAccess(physical[g ^ 1], 0));
      check(cudaMalloc(&partial[g], capacity));
    }
  SharedPackets shared;
  shared.initialize(3 * capacity, group, local);
  std::array<std::unique_ptr<Piece>, kSlots> slots;
  for (int s = 0; s < kSlots; ++s) {
    slots[s] = std::make_unique<Piece>();
    slots[s]->own.initialize(3 * capacity, group, physical, shared.own[s]);
    slots[s]->peer.initialize(3 * capacity, group, physical, shared.peer[s]);
    slots[s]->remote.initialize(capacity, group, physical);
  }
  std::array<ncclComm_t, 2> comms{};
  if (cfg.nccl) {
    ncclUniqueId id;
    if (rank == 0) check(ncclGetUniqueId(&id));
    check(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    check(ncclGroupStart());
    for (int g = 0; g < 2; ++g) {
      check(cudaSetDevice(physical[g]));
      check(ncclCommInitRank(&comms[g], 8, id, 2 * rank + g));
    }
    check(ncclGroupEnd());
  }
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(group, &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity))
    throw std::runtime_error("routing CPU affinity failed");
  auto gpuReady = [&](int s, bool consume) {
    bool ready = true;
    for (auto const& d : devices) {
      check(cudaSetDevice(d->physical));
      auto status =
          cudaEventQuery(consume ? d->consumeDone[s] : d->uploadDone[s]);
      if (status == cudaErrorNotReady)
        ready = false;
      else
        check(status);
    }
    return ready;
  };
  auto launch = [&](size_t bytes) {
    size_t shard = bytes / 8, perShard = (shard + capacity - 1) / capacity;
    size_t pieces = 8 * perShard, next = 0, retired = 0;
    while (retired != pieces) {
      for (int s = 0; s < kSlots; ++s) {
        Piece& p = *slots[s];
        if (!p.active) continue;
        int tag = 600 + 8 * s;
        bool owner = group == p.owner;
        bool receiver = !cfg.rs || node == p.destination / 4;
        if (p.phase == 0 && gpuReady(s, false)) {
          check(MPI_Isend(nullptr, 0, MPI_BYTE, rank ^ 1, tag, MPI_COMM_WORLD,
                          &p.readySend));
          p.phase = 1;
        }
        if (p.phase == 1 && completed(p.readyRecv)) {
          if (owner) {
            // Only the owner reduces this stripe. Shared packets avoid a
            // local payload exchange, and each input has one CPU reader.
            auto a = reinterpret_cast<float const*>(p.own.host);
            auto b = reinterpret_cast<float const*>(p.own.host + capacity);
            auto c = reinterpret_cast<float const*>(p.peer.host);
            auto d = reinterpret_cast<float const*>(p.peer.host + capacity);
            auto out = reinterpret_cast<float*>(p.own.host + 2 * capacity);
            if (cfg.pair) {
              for (size_t i = 0; i < p.bytes / 4; ++i) out[i] = a[i] + c[i];
            } else {
              for (size_t i = 0; i < p.bytes / 4; ++i)
                out[i] = (a[i] + b[i]) + (c[i] + d[i]);
            }
            if (!cfg.rs || !receiver)
              check(MPI_Isend(out, int(p.bytes), MPI_BYTE, rank ^ 2, tag + 1,
                              MPI_COMM_WORLD, &p.netSend));
          }
          p.phase = 2;
        }
        if (p.phase == 2 &&
            (owner ? completed(p.netRecv) : completed(p.done))) {
          if (owner) {
            // Send must finish before the node partial is modified in place.
            if (!completed(p.netSend)) continue;
            if (receiver) {
              auto out = reinterpret_cast<float*>(p.own.host + 2 * capacity);
              auto in = reinterpret_cast<float const*>(p.remote.host);
              for (size_t i = 0; i < p.bytes / 4; ++i) out[i] += in[i];
            }
            check(MPI_Isend(nullptr, 0, MPI_BYTE, rank ^ 1, tag + 2,
                            MPI_COMM_WORLD, &p.done));
          }
          for (int g = 0; g < 2; ++g) {
            auto& d = *devices[g];
            check(cudaSetDevice(d.physical));
            if (!cfg.rs || d.globalRank == p.destination) {
              if (cfg.delay == d.globalRank)
                delayConsumerKernel<<<1, 1, 0, d.consume>>>();
              auto src =
                  (owner ? p.own.mapped[g] : p.peer.mapped[g]) + 2 * capacity;
              size_t offset = cfg.rs ? p.offset % shard : p.offset;
              download<<<std::min<size_t>(128, (p.bytes / 4 + 255) / 256), 256,
                         0, d.consume>>>(reinterpret_cast<float const*>(src),
                                         d.output + offset / 4, p.bytes / 4);
              check(cudaGetLastError());
            }
            check(cudaEventRecord(d.consumeDone[s], d.consume));
          }
          p.phase = 3;
        }
        if (p.phase == 3 && gpuReady(s, true)) {
          if (!owner)
            check(MPI_Isend(nullptr, 0, MPI_BYTE, rank ^ 1, tag + 3,
                            MPI_COMM_WORLD, &p.ack));
          p.phase = 4;
        }
        if (p.phase == 4 && completed(p.ack) && completed(p.done) &&
            completed(p.readySend)) {
          p.active = false;
          ++retired;
        }
      }
      if (next < pieces && !slots[next % kSlots]->active) {
        int s = next % kSlots, tag = 600 + 8 * s;
        Piece& p = *slots[s];
        p.destination = next % 8;
        p.owner = (p.destination / 2) % 2;
        p.offset = p.destination * shard + (next / 8) * capacity;
        p.bytes = std::min(capacity, shard - (next / 8) * capacity);
        p.phase = 0;
        p.active = true;
        check(MPI_Irecv(nullptr, 0, MPI_BYTE, rank ^ 1, tag, MPI_COMM_WORLD,
                        &p.readyRecv));
        if (group == p.owner) {
          check(MPI_Irecv(nullptr, 0, MPI_BYTE, rank ^ 1, tag + 3,
                          MPI_COMM_WORLD, &p.ack));
          if (!cfg.rs || node == p.destination / 4)
            check(MPI_Irecv(p.remote.host, int(p.bytes), MPI_BYTE, rank ^ 2,
                            tag + 1, MPI_COMM_WORLD, &p.netRecv));
        } else {
          check(MPI_Irecv(nullptr, 0, MPI_BYTE, rank ^ 1, tag + 2,
                          MPI_COMM_WORLD, &p.done));
        }
        for (int g = 0; g < 2; ++g) {
          auto& d = *devices[g];
          check(cudaSetDevice(d.physical));
          if (cfg.pair) {
            size_t half = p.bytes / 2, offset = (p.offset + g * half) / 4;
            reducePair<<<std::min<size_t>(128, (half / 4 + 255) / 256), 256, 0,
                         d.upload>>>(d.input + offset,
                                     devices[g ^ 1]->input + offset, partial[g],
                                     half / 4);
            check(cudaGetLastError());
            check(cudaMemcpyAsync(p.own.host + g * half, partial[g], half,
                                  cudaMemcpyDeviceToHost, d.upload));
          } else {
            check(cudaMemcpyAsync(p.own.host + g * capacity,
                                  d.input + p.offset / 4, p.bytes,
                                  cudaMemcpyDeviceToHost, d.upload));
          }
          check(cudaEventRecord(d.uploadDone[s], d.upload));
        }
        ++next;
      }
      int flag;
      check(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag,
                       MPI_STATUS_IGNORE));
    }
  };
  auto call = [&](size_t bytes, int generation, bool nccl) {
    for (auto const& d : devices) {
      check(cudaSetDevice(d->physical));
      fillInput<<<256, 256, 0, d->upload>>>(d->input, bytes / 4, d->globalRank,
                                            generation);
      check(cudaGetLastError());
      check(cudaMemsetAsync(d->output, 0xff, cfg.rs ? bytes / 8 : bytes,
                            d->consume));
      check(cudaDeviceSynchronize());
    }
    check(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::steady_clock::now();
    if (!nccl)
      launch(bytes);
    else {
      check(ncclGroupStart());
      for (int g = 0; g < 2; ++g) {
        auto& d = *devices[g];
        check(cudaSetDevice(d.physical));
        if (cfg.rs)
          check(ncclReduceScatter(d.input, d.output, bytes / 32, ncclFloat,
                                  ncclSum, comms[g], d.consume));
        else
          check(ncclAllReduce(d.input, d.output, bytes / 4, ncclFloat, ncclSum,
                              comms[g], d.consume));
      }
      check(ncclGroupEnd());
      for (auto const& d : devices) {
        check(cudaSetDevice(d->physical));
        check(cudaStreamSynchronize(d->consume));
      }
    }
    double us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start)
                    .count();
    double maxUs;
    check(MPI_Allreduce(&us, &maxUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    for (auto const& d : devices) {
      check(cudaSetDevice(d->physical));
      check(cudaMemsetAsync(d->wrong, 0, sizeof(unsigned), d->consume));
      verifyReduction<<<256, 256, 0, d->consume>>>(
          d->output, d->wrong, cfg.rs ? bytes / 32 : bytes / 4,
          cfg.rs ? d->globalRank * (bytes / 32) : 0, generation);
      check(cudaGetLastError());
      unsigned wrong;
      check(cudaMemcpyAsync(&wrong, d->wrong, sizeof(wrong),
                            cudaMemcpyDeviceToHost, d->consume));
      check(cudaStreamSynchronize(d->consume));
      if (wrong)
        throw std::runtime_error("reduction mismatch on GPU " +
                                 std::to_string(d->globalRank));
    }
    return maxUs;
  };
  char const* op = cfg.rs ? "reducescatter" : "allreduce";
  if (rank == 0) {
    std::printf(
        "# %s FP32; 2x4 GPUs, 4 MPI ranks; input bytes/GPU; chunk=%zu; "
        "slots=3; eager; host enqueue-to-completion max rank; warmups=%d "
        "calls=%d rounds=%d; every call checked\n",
        op, capacity, cfg.warmups, cfg.iterations, cfg.repeats);
    std::printf("# pair_reduce=%d delay_rank=%d\n", int(cfg.pair), cfg.delay);
    std::puts("collective,bytes,backend,median_us,min_us,max_us");
  }
  if (cfg.nccl)
    for (int i = 0; i < cfg.warmups; ++i) call(cfg.begin, i, true);
  for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
    std::array<std::vector<double>, 2> runs;
    for (int r = 0; r < cfg.repeats; ++r) {
      for (int b = 0; b < (cfg.nccl ? 2 : 1); ++b) {
        int backend = cfg.nccl ? (b ^ (r % 2)) : 0;
        double sum = 0;
        for (int i = 0; i < cfg.warmups + cfg.iterations; ++i) {
          double us =
              call(bytes, 10 + (r * (cfg.warmups + cfg.iterations) + i) % 100,
                   backend);
          if (i >= cfg.warmups) {
            sum += us;
            if (!rank)
              std::printf("# sample,%s,%zu,%s,%d,%d,%.3f,0\n", op, bytes,
                          backend ? "nccl" : "host-shared-ready", r + 1,
                          i - cfg.warmups + 1, us);
          }
        }
        runs[backend].push_back(sum / cfg.iterations);
      }
    }
    if (!rank)
      for (int b = 0; b < (cfg.nccl ? 2 : 1); ++b) {
        auto bounds = std::minmax_element(runs[b].begin(), runs[b].end());
        std::printf("%s,%zu,%s,%.3f,%.3f,%.3f\n", op, bytes,
                    b ? "nccl" : "host-shared-ready", median(runs[b]),
                    *bounds.first, *bounds.second);
        std::fflush(stdout);
      }
    if (bytes > cfg.end / 2) break;
  }
  if (cfg.nccl)
    for (auto c : comms) check(ncclCommDestroy(c));
  slots = {};
  shared.close();
  if (cfg.pair)
    for (int g = 0; g < 2; ++g) {
      check(cudaSetDevice(physical[g]));
      check(cudaFree(partial[g]));
      check(cudaDeviceDisablePeerAccess(physical[g ^ 1]));
    }
  return 0;
}
}  // namespace
int main(int argc, char** argv) {
  bool initialized = false;
  try {
    Config cfg = parse(argc, argv);
    bindGroupBeforeMpi();
    check(MPI_Init(&argc, &argv));
    initialized = true;
    int rank, world, localRank, localSize;
    check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    check(MPI_Comm_size(MPI_COMM_WORLD, &world));
    MPI_Comm local;
    check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                              MPI_INFO_NULL, &local));
    check(MPI_Comm_rank(local, &localRank));
    check(MPI_Comm_size(local, &localSize));
    if (world != 4 || localSize != 2 || localRank != rank % 2)
      throw std::runtime_error("requires two nodes with two MPI ranks each");
    run(cfg, rank, local);
    check(MPI_Comm_free(&local));
    check(MPI_Finalize());
  } catch (std::exception const& e) {
    std::fprintf(stderr, "host reductions: %s\n", e.what());
    if (initialized) MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
