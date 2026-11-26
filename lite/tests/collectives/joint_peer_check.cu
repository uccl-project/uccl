// Reuse the frozen benchmark's independent rank/index oracle and guarded
// buffers, not its timing loop. Exercise one prepared executor across disjoint
// caller stream sets while its preceding operation is still in flight.
#define main uccl_lite_joint_benchmark_main
#include "../../benchmarks/joint/joint_bench.cu"
#undef main
#include "../planner/peer_fixture.hpp"
#include <set>

namespace {
void queueChecked(JointCollectives& peer, Buffers& b, int op, size_t bytes,
                  unsigned generation) {
  auto call = b.call(op, bytes);
  int ranks = int(b.devices.size());
  size_t inputs = call.count * (op == 2 ? ranks : 1);
  size_t outputs = call.count * (op == 1 ? ranks : 1);
  for (int rank = 0; rank < ranks; ++rank) {
    ck(cudaSetDevice(b.devices[rank]));
    initialize<<<128, 256, 0, b.streams[rank]>>>(
        b.in[rank], b.out[rank], inputs, outputs, rank, generation,
        rank == ranks - 1 ? 100000 : 0);
    ck(cudaMemsetAsync(b.errors[rank], 0, sizeof(int), b.streams[rank]));
  }
  ck(cudaSetDevice(b.devices.back()));
  peer.enqueue(call);
  int current = -1;
  ck(cudaGetDevice(&current));
  if (current != b.devices.back())
    throw std::runtime_error("peer enqueue changed caller device");
  for (int rank = 0; rank < ranks; ++rank) {
    ck(cudaSetDevice(b.devices[rank]));
    verify<<<128, 256, 0, b.streams[rank]>>>(
        b.in[rank], b.out[rank], inputs, outputs, rank, ranks, op, call.count,
        generation, b.errors[rank]);
  }
  ck(cudaGetLastError());
}
void finishChecked(Buffers& b) {
  for (size_t rank = 0; rank < b.devices.size(); ++rank) {
    ck(cudaSetDevice(b.devices[rank]));
    ck(cudaStreamSynchronize(b.streams[rank]));
    int errors = 0;
    ck(cudaMemcpy(&errors, b.errors[rank], sizeof(errors),
                  cudaMemcpyDeviceToHost));
    if (errors)
      throw std::runtime_error("peer stream/family transition corrupted data");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int ranks = argc > 1 ? std::stoi(argv[1]) : 2;
    if (ranks < 1 || ranks > kPeerRankCapacity)
      throw std::invalid_argument(
          "usage: joint_peer_check RANKS "
          "[auto|direct|coordinator|bulk|model-test] [GPU,...]");
    std::vector<int> order;
    if (argc > 3) {
      std::stringstream input(argv[3]);
      std::string part;
      while (std::getline(input, part, ',')) order.push_back(std::stoi(part));
      int count = 0;
      ck(cudaGetDeviceCount(&count));
      if (order.size() != size_t(ranks) ||
          std::set<int>(order.begin(), order.end()).size() != order.size() ||
          std::any_of(order.begin(), order.end(), [count](int device) {
            return device < 0 || device >= count;
          }))
        throw std::invalid_argument("invalid device permutation");
    }
    constexpr size_t large = 64 * 1024 * 1024;
    Buffers a(ranks, large + 64, order), b(ranks, large + 64, order);
    Config cfg;
    cfg.source = "peer-common-transition-check";
    auto t = topology(cfg, a);
    PeerTuning tuning;
    if (argc > 2) {
      std::string family = argv[2];
      if (family == "auto") {
      } else if (family == "model-test")
        tuning = syntheticPeerModel(t);
      else if (family == "direct")
        tuning.validationFamily = PeerFamily::Direct;
      else if (family == "coordinator")
        tuning.validationFamily = PeerFamily::Coordinator;
      else if (family == "bulk")
        tuning.validationFamily = PeerFamily::Bulk;
      else
        throw std::invalid_argument("unknown validation family");
    }
    auto peer = std::make_unique<JointCollectives>(
        t, JointMode::Peer, BackendPolicy{}, cpu_router::CpuRouterOptions{},
        tuning);
    bool unprepared = false;
    try {
      peer->enqueue(a.call(0, 1024));
    } catch (std::logic_error const&) {
      unprepared = true;
    }
    if (!unprepared)
      throw std::runtime_error("unprepared workload was accepted");
    size_t sizes[] = {1024, 1024 * 1024, 8 * 1024 * 1024, large,
                      large + 2 * size_t(ranks)};
    for (size_t i = 0; i < 4; ++i)
      sizes[i] = sizes[i] / (16 * ranks) * (16 * ranks);
    sizes[4] = sizes[3] + 2 * ranks;
    if (!tuning.validationFamily && tuning.automaticCalibration)
      std::printf(
          "AUTO qualification: %d GPUs (%s), BF16, exact per-size selection; "
          "direct baseline; private buffers; host enqueue-to-completion mean; "
          "2 warmups, 3 calls/repetition, 3 repetitions\n",
          ranks, t.gpuNames.front().c_str());
    // Prepare every exact model key, before producer work or timing begins.
    for (int op = 0; op < 3; ++op)
      for (auto bytes : sizes) {
        peer->prepare(a.call(op, bytes));
        peer->prepare(b.call(op, bytes));
        if (!tuning.validationFamily && tuning.automaticCalibration) {
          auto const& plan = peer->peerPlan(a.call(op, bytes));
          if (!plan.qualified || plan.calibrationIdentity.empty())
            throw std::runtime_error(
                "automatic plan lacks local qualification");
          auto candidates =
              peerCandidates(t, a.call(op, bytes).workload(), tuning);
          auto admission =
              choosePlan(t, a.call(op, bytes).workload(), false, candidates);
          for (size_t family = 0; family < plan.localTimings.size(); ++family)
            if (plan.localTimings[family].size() !=
                (admission.candidates[family].eligible ? 3 : 0))
              throw std::runtime_error(
                  "automatic qualification did not retain all samples");
          std::printf("AUTO op=%d bytes=%zu family=%d modeled=%d", op, bytes,
                      int(plan.family), int(plan.modeled));
          for (size_t family = 0; family < plan.localTimings.size(); ++family) {
            std::printf(" f%zu_us=", family);
            for (double us : plan.localTimings[family])
              std::printf("%.3f,", us);
          }
          std::puts("");
          std::fflush(stdout);
        }
        if (tuning.validationFamily &&
            peer->peerPlan(a.call(op, bytes)).family !=
                *tuning.validationFamily)
          throw std::runtime_error("validation selected the wrong executor");
      }
    if (argc > 2 && std::string(argv[2]) == "model-test") {
      if (!peer->peerPlan(a.call(0, sizes[0])).modeled ||
          peer->peerPlan(a.call(0, sizes[0])).family !=
              (ranks == 1 ? PeerFamily::Direct : PeerFamily::Coordinator) ||
          (peer->peerPlan(a.call(0, sizes[3])).family !=
           (ranks > 2    ? PeerFamily::Bulk
            : ranks == 2 ? PeerFamily::Coordinator
                         : PeerFamily::Direct)))
        throw std::runtime_error(
            "executor did not consume resource-driven plans");
      std::puts(
          "Synthetic resource model: correctness only, no performance claim");
    }
    unsigned generation = 1;
    int cases = 0;
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        for (size_t from : sizes)
          for (size_t to : sizes) {
            queueChecked(*peer, a, first, from, ++generation);
            queueChecked(*peer, b, second, to, ++generation);
            finishChecked(a);
            finishChecked(b);
            ++cases;
          }
    // Destruction also waits for the two-rank caller-stream coordinator.
    queueChecked(*peer, a, 0, sizes[0], ++generation);
    peer.reset();
    finishChecked(a);
    hostOracle(cfg, a, JointMode::Peer);
    std::printf(
        "PASS: %d peer family/stream transitions on %d GPUs, aligned "
        "and scalar tails, delayed producers, guards, input "
        "preservation, device restoration and pending destruction\n",
        cases, ranks);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
