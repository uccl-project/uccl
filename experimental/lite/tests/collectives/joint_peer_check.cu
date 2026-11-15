// Reuse the frozen benchmark's independent rank/index oracle and guarded
// buffers, not its timing loop. Exercise one prepared executor across disjoint
// caller stream sets while its preceding operation is still in flight.
#define main uccl_lite_joint_benchmark_main
#include "../../benchmarks/joint/joint_bench.cu"
#undef main

namespace {
void queueChecked(JointCollectives& peer, Buffers& b, int op, size_t bytes,
                  unsigned generation) {
  auto call = b.call(op, bytes);
  int ranks = int(b.devices.size());
  size_t inputs = call.count * (op == 2 ? ranks : 1);
  size_t outputs = call.count * (op == 1 ? ranks : 1);
  for (int rank = 0; rank < ranks; ++rank) {
    ck(cudaSetDevice(rank));
    initialize<<<128, 256, 0, b.streams[rank]>>>(
        b.in[rank], b.out[rank], inputs, outputs, rank, generation,
        rank == ranks - 1 ? 100000 : 0);
    ck(cudaMemsetAsync(b.errors[rank], 0, sizeof(int), b.streams[rank]));
  }
  ck(cudaSetDevice(ranks - 1));
  peer.enqueue(call);
  int current = -1;
  ck(cudaGetDevice(&current));
  if (current != ranks - 1)
    throw std::runtime_error("peer enqueue changed caller device");
  for (int rank = 0; rank < ranks; ++rank) {
    ck(cudaSetDevice(rank));
    verify<<<128, 256, 0, b.streams[rank]>>>(
        b.in[rank], b.out[rank], inputs, outputs, rank, ranks, op, call.count,
        generation, b.errors[rank]);
  }
  ck(cudaGetLastError());
}
void finishChecked(Buffers& b) {
  for (size_t rank = 0; rank < b.devices.size(); ++rank) {
    ck(cudaSetDevice(int(rank)));
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
    if (ranks != 2 && ranks != 4)
      throw std::invalid_argument("usage: joint_peer_check [2|4]");
    constexpr size_t large = 64 * 1024 * 1024;
    Buffers a(ranks, large + 64), b(ranks, large + 64);
    Config cfg;
    cfg.source = "peer-common-transition-check";
    auto t = topology(cfg, a);
    auto peer = std::make_unique<JointCollectives>(t, JointMode::Peer);
    // Reserve for every later staged family at the explicit idle boundary.
    peer->prepare(a.call(2, large + 2 * ranks));
    size_t sizes[] = {1024, 1024 * 1024, 8 * 1024 * 1024, large,
                      large + 2 * size_t(ranks)};
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
    queueChecked(*peer, a, 0, 1024, ++generation);
    peer.reset();
    finishChecked(a);
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
