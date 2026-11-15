// Baseline and candidate in one process, with alternating invocation order.
// Keep the frozen benchmark's guarded inputs, correctness oracle and timer.
#define JointCollectives BaselineJointCollectives
#define JointMode BaselineJointMode
#include "peer_common_baseline.hpp"
#undef JointMode
#undef JointCollectives
#define main uccl_lite_frozen_benchmark_main
#include "../../joint/joint_bench.cu"
#undef main

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("usage: peer_common_paired GPUS NEW_SAMPLES");
    Config cfg;
    cfg.gpus = std::stoi(argv[1]);
    cfg.end = 1 << 20;
    cfg.source = "peer-common-in-process";
    cfg.warmup = 200;
    cfg.repeats = 9;
    cfg.iterations = 50;
    if (std::filesystem::exists(argv[2]))
      throw std::invalid_argument("samples file already exists");
    std::ofstream raw(argv[2]);
    if (!raw) throw std::runtime_error("cannot create samples file");
    raw << "collective,bytes,repeat,iteration,first,before_us,after_us\n";
    Buffers b(cfg.gpus, cfg.end);
    auto t = topology(cfg, b);
    uccl_lite::BaselineJointCollectives before(
        t, uccl_lite::BaselineJointMode::Peer);
    JointCollectives after(t, JointMode::Peer);
    unsigned generation = cfg.seed;
    std::cout << "collective,bytes,before_us,after_us,after_over_before\n";
    for (int op = 0; op < 3; ++op)
      for (size_t bytes = 1024; bytes <= cfg.end; bytes *= 2) {
        auto call = b.call(op, bytes);
        before.prepare(call);
        after.prepare(call);
        Enqueue oldEnqueue = [&] { before.enqueue(call); };
        Enqueue newEnqueue = [&] { after.enqueue(call); };
        for (int i = 0; i < cfg.warmup; ++i) {
          invocation(b, op, bytes, ++generation, oldEnqueue);
          invocation(b, op, bytes, ++generation, newEnqueue);
        }
        std::vector<double> oldRounds, newRounds;
        for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
          std::vector<double> oldSamples, newSamples;
          for (int i = 0; i < cfg.iterations; ++i) {
            double oldUs, newUs;
            bool oldFirst = (repeat + i) % 2 == 0;
            unsigned pairGeneration = ++generation;
            if (oldFirst) {
              oldUs = invocation(b, op, bytes, pairGeneration, oldEnqueue);
              newUs = invocation(b, op, bytes, pairGeneration, newEnqueue);
            } else {
              newUs = invocation(b, op, bytes, pairGeneration, newEnqueue);
              oldUs = invocation(b, op, bytes, pairGeneration, oldEnqueue);
            }
            oldSamples.push_back(oldUs);
            newSamples.push_back(newUs);
            raw << opName(op) << ',' << bytes << ',' << repeat << ',' << i
                << ',' << (oldFirst ? "before" : "after") << ','
                << std::setprecision(10) << oldUs << ',' << newUs << '\n';
          }
          oldRounds.push_back(median(oldSamples));
          newRounds.push_back(median(newSamples));
        }
        double oldUs = median(oldRounds), newUs = median(newRounds);
        std::cout << opName(op) << ',' << bytes << ',' << oldUs << ',' << newUs
                  << ',' << newUs / oldUs << '\n';
      }
    std::cerr << "PASS: paired in-process, every invocation checked\n";
  } catch (std::exception const& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
