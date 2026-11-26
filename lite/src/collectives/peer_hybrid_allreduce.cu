// Experimental composition: pipelined GPU ReduceScatter followed by mapped-host
// packet AllGather. The caller-visible input/result is in place. Compare only
// with this harness's identically timed, one-process NCCL control.
#define main uccl_lite_peer_benchmark_main
#include "peer_ag_rs.cu"
#undef main

int main(int argc, char** argv) {
  try {
    auto cfg = parse(argc, argv);
    if (cfg.graph)
      throw std::invalid_argument(
          "hybrid AllReduce does not support graph capture");
    for (int r = 0; r < kRanks; ++r) {
      cudaCheck(cudaSetDevice(r));
      for (int p = 0; p < kRanks; ++p) {
        if (p == r) continue;
        auto status = cudaDeviceEnablePeerAccess(p, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled)
          cudaGetLastError();
        else
          cudaCheck(status);
      }
    }
    std::array<std::unique_ptr<Device>, kRanks> owned;
    std::array<Device*, kRanks> devices{};
    std::array<uccl_lite::cpu_router::PacketPort, kRanks> ports{};
    for (int r = 0; r < kRanks; ++r) {
      owned[r] =
          std::make_unique<Device>(r, cfg.end, 0, false, cfg.rsRingChunkBytes);
      devices[r] = owned[r].get();
      // RS writes one final shard into output; AG consumes that immutable
      // shard and replaces the complete input with the reduced result.
      ports[r] = {devices[r]->output, devices[r]->input, devices[r]->stream, r};
    }
    cfg.packetRouterGpuProgress = true;
    cfg.packetRouterPerSource = true;
    uccl_lite::cpu_router::PacketPipeline pipeline(PacketRouter::options(cfg));
    pipeline.prepare(ports);
    std::array<ncclComm_t, kRanks> comms{};
    int ranks[kRanks] = {0, 1, 2, 3};
    ncclCheck(ncclCommInitAll(comms.data(), kRanks, ranks));
    int version = 0;
    ncclCheck(ncclGetVersion(&version));
    std::printf(
        "# experimental hybrid AR; FP32 in-place; 4 L4; NCCL %d; "
        "slowest-GPU eager events; warmups=%d iterations=%d repeats=%d; "
        "RS piece=%zu AG packet=%zu blocks=%d\n",
        version, cfg.warmups, cfg.iterations, cfg.repeats, cfg.rsRingChunkBytes,
        cfg.packetBytes, cfg.packetBlocks);
    for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
      size_t count = bytes / sizeof(float), shard = count / kRanks;
      std::array<std::vector<double>, 2> means;
      for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
        for (int ordinal = 0; ordinal < 2; ++ordinal) {
          int backend = (repeat + ordinal) % 2;
          double total = 0;
          for (int call = 0; call < cfg.warmups + cfg.iterations; ++call) {
            int generation =
                10 + repeat * (cfg.warmups + cfg.iterations) + call;
            for (auto* d : devices) {
              cudaCheck(cudaSetDevice(d->rank));
              fillInput<<<256, 256, 0, d->stream>>>(
                  backend ? d->reference : d->input, count, d->rank,
                  generation);
            }
            synchronize(devices);
            for (auto* d : devices) {
              cudaCheck(cudaSetDevice(d->rank));
              cudaCheck(cudaEventRecord(d->start, d->stream));
            }
            if (backend == 0) {
              launchScatterRing(devices, shard);
              pipeline.enqueue(uccl_lite::cpu_router::PacketProgram::allGather(
                                   shard * 2, kRanks),
                               ports);
            } else {
              ncclCheck(ncclGroupStart());
              for (auto* d : devices) {
                cudaCheck(cudaSetDevice(d->rank));
                ncclCheck(ncclAllReduce(d->reference, d->reference, count,
                                        ncclFloat, ncclSum, comms[d->rank],
                                        d->stream));
              }
              ncclCheck(ncclGroupEnd());
            }
            for (auto* d : devices) {
              cudaCheck(cudaSetDevice(d->rank));
              cudaCheck(cudaEventRecord(d->end, d->stream));
            }
            synchronize(devices);
            double slowest = 0;
            for (auto* d : devices) {
              cudaCheck(cudaSetDevice(d->rank));
              float ms = 0;
              cudaCheck(cudaEventElapsedTime(&ms, d->start, d->end));
              slowest = std::max(slowest, double(ms) * 1000);
              cudaCheck(cudaMemsetAsync(d->wrong, 0, sizeof(unsigned int),
                                        d->stream));
              checkOutput<<<256, 256, 0, d->stream>>>(
                  backend ? d->reference : d->input, d->wrong, count, count, 0,
                  generation, false);
              unsigned int wrong = 0;
              cudaCheck(cudaMemcpyAsync(&wrong, d->wrong, sizeof(wrong),
                                        cudaMemcpyDeviceToHost, d->stream));
              cudaCheck(cudaStreamSynchronize(d->stream));
              if (wrong)
                throw std::runtime_error("hybrid AR full output mismatch");
            }
            if (call >= cfg.warmups) {
              total += slowest;
              std::printf("# sample,all_reduce,%zu,%s,%d,%d,%.3f\n", bytes,
                          backend ? "nccl" : "ring-packet", repeat + 1,
                          call - cfg.warmups + 1, slowest);
            }
          }
          means[backend].push_back(total / cfg.iterations);
        }
      }
      for (int backend = 0; backend < 2; ++backend)
        std::printf("all_reduce,%zu,%s,%.3f\n", bytes,
                    backend ? "nccl" : "ring-packet", median(means[backend]));
      std::fflush(stdout);
      if (bytes > cfg.end / 2) break;
    }
    for (auto comm : comms) ncclCheck(ncclCommDestroy(comm));
  } catch (std::exception const& error) {
    std::fprintf(stderr, "hybrid AR: %s\n", error.what());
    return 1;
  }
}
