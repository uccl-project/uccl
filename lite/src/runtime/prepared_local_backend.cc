#include "uccl_lite/runtime/prepared_local.hpp"
#include <cstdio>
#include <map>
#include <stdexcept>

namespace uccl_lite::runtime {
namespace {
LocalPreparationOptions options(ScalarType dtype) {
  LocalPreparationOptions result;
  result.dtype = dtype;
  return result;
}
char const* name(LocalBackend kind) {
  return kind == LocalBackend::PeerPacket ? "peer-packet"
         : kind == LocalBackend::Host     ? "host"
         : kind == LocalBackend::Fallback ? "fallback"
                                          : "peer";
}
class Adapter : public CollectiveBackend {
  BackendContext context_;
  PreparedLocalRuntime runtime_;
  std::map<std::pair<int, size_t>, size_t> plans_;
  bool prepared_ = false;

 public:
  explicit Adapter(BackendContext const& context, ScalarType dtype)
      : context_(context), runtime_(context, options(dtype)) {}
  void prepare() override {
    if (prepared_) return;
    if (context_.requests.empty())
      throw std::invalid_argument(
          "automatic runtime requires prepared workloads");
    for (auto const& request : context_.requests) {
      auto key = std::make_pair(int(request.operation), request.count);
      if (plans_.count(key))
        throw std::invalid_argument("duplicate prepared shape");
      auto plan = runtime_.prepare(request);
      plans_.emplace(key, plan);
      auto const& selected = runtime_.selection(plan);
      if (context_.rank == 0) {
        std::fprintf(stderr,
                     "UCCL Lite prepared plan: op=%d count=%zu selected=%s "
                     "resource_model_us=%.9g host_eligible=%d host_correct=%d "
                     "packet_eligible=%d packet_correct=%d topology=%s "
                     "artifact=%s layouts=",
                     key.first, key.second, name(selected.backend),
                     selected.peerResourceEstimateUs,
                     int(selected.hostEligible), int(selected.hostCorrect),
                     int(selected.peerPacketEligible),
                     int(selected.candidateCorrect[2]),
                     selected.topologyIdentity.c_str(),
                     selected.executorIdentity.c_str());
        for (auto const& layout : request.layouts)
          std::fprintf(stderr, "%d:%u,", int(layout.inPlace), layout.alignment);
        for (auto backend : kLocalAlternatives) {
          std::fprintf(stderr, " %s_us=", name(backend));
          for (double us : selected.samples[size_t(backend)])
            std::fprintf(stderr, "%.9g,", us);
        }
        std::fprintf(stderr, "\n");
      }
    }
    prepared_ = true;
  }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    runtime_.enqueue(plans_.at({int(op), count}), input, output, stream);
  }
  void synchronize() override { runtime_.synchronize(); }
  bool initialized() const override { return prepared_; }
  bool idle() const override { return runtime_.idle(); }
};
}  // namespace
BackendRegistration preparedLocalBackend() {
  return preparedLocalBackend(ScalarType::BFloat16);
}
BackendRegistration preparedLocalBackend(ScalarType dtype) {
  if (dtype != ScalarType::BFloat16 && dtype != ScalarType::Float32)
    throw std::invalid_argument("unsupported prepared provider dtype");
  return {
      {dtype == ScalarType::Float32 ? "prepared-local-fp32" : "prepared-local",
       "",
       {},
       dtype,
       true,
       false,
       2,
       32,
       true},
      [] { return hostPacketBackend().uniqueId(); },
      [dtype](BackendContext const& context) {
        return std::make_unique<Adapter>(context, dtype);
      }};
}
}  // namespace uccl_lite::runtime
