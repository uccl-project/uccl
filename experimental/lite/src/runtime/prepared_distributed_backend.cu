#include "qualification.cuh"
#include "uccl_lite/runtime/network_identity.hpp"
#include "uccl_lite/runtime/prepared_network.hpp"
#include <mscclpp/core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <sstream>

namespace uccl_lite::runtime {
namespace {
using qualification::check;
class DirectReference final : public CollectiveBackend {
  BackendContext context_;
  std::unique_ptr<CollectiveBackend> direct_;
  std::string snapshot_;
  std::string live() const {
    return inspectTopology(context_.localDevices).hardwareIdentity + ';' +
           networkTransportIdentity();
  }

 public:
  DirectReference(BackendContext const& context, ScalarType dtype)
      : context_(context),
        direct_(directExchangeBackend(dtype).create(context)) {}
  void prepare() override {
    if (!direct_->initialized()) {
      direct_->prepare();
      snapshot_ = live();
    } else if (snapshot_ != live())
      throw std::runtime_error("strict direct reference identity changed");
  }
  std::string configurationIdentity() const override {
    return direct_->configurationIdentity() + ';' + snapshot_;
  }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    direct_->enqueue(op, input, output, count, stream);
  }
  void synchronize() override { direct_->synchronize(); }
  bool initialized() const override { return direct_->initialized(); }
  bool idle() const override { return direct_->idle(); }
};
class Adapter final : public CollectiveBackend {
  BackendContext context_;
  ScalarType dtype_;
  std::shared_ptr<mscclpp::TcpBootstrap> control_;
  std::shared_ptr<DirectReference> reference_;
  std::unique_ptr<PreparedNetworkRuntime> runtime_;
  std::map<std::pair<int, size_t>, size_t> plans_;
  bool prepared_ = false;
  template <class T>
  std::vector<T> gather(T const& own) {
    std::vector<T> values(context_.ranks);
    values[context_.rank] = own;
    control_->allGather(values.data(), sizeof(T));
    return values;
  }
  NetworkId id() {
    struct Value {
      int valid = 1;
      NetworkId id{};
    } value;
    if (!context_.rank) try {
        value.id = PreparedNetworkRuntime::createUniqueId();
      } catch (...) {
        value.valid = 0;
      }
    value = gather(value).front();
    if (!value.valid)
      throw std::runtime_error("strict distributed bootstrap ID unavailable");
    return value.id;
  }
  std::vector<double> validateReference(CollectiveRequest const& request) {
    qualification::Probe probe;
    int ready = 1;
    try {
      probe.allocate(request, context_.rank, context_.ranks, dtype_);
    } catch (...) {
      ready = 0;
    }
    for (auto vote : gather(ready))
      if (!vote)
        throw std::runtime_error("strict reference probe allocation failed");
    std::vector<double> samples;
    for (int iteration = 0; iteration < 5; ++iteration) {
      probe.initialize(iteration);
      check(cudaStreamSynchronize(probe.stream));
      control_->barrier();
      auto begin = std::chrono::steady_clock::now();
      reference_->enqueue(request.operation, probe.input(), probe.output(),
                          request.count, probe.stream);
      reference_->synchronize();
      double us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
      auto times = gather(us);
      if (iteration >= 2)
        samples.push_back(*std::max_element(times.begin(), times.end()));
      check(cudaMemsetAsync(probe.wrong, 0, sizeof(unsigned), probe.stream));
      probe.verify(iteration);
      unsigned wrong = 0;
      check(cudaMemcpyAsync(&wrong, probe.wrong, sizeof(wrong),
                            cudaMemcpyDeviceToHost, probe.stream));
      check(cudaStreamSynchronize(probe.stream));
      for (auto errors : gather(wrong))
        if (errors)
          throw std::runtime_error(
              "strict direct reference failed full-buffer verification");
    }
    return samples;
  }

 public:
  Adapter(BackendContext const& context, ScalarType dtype)
      : context_(context), dtype_(dtype) {}
  void prepare() override {
    if (prepared_) return;
    if (context_.requests.empty())
      throw std::invalid_argument(
          "strict distributed prepare needs exact workloads");
    control_ =
        std::make_shared<mscclpp::TcpBootstrap>(context_.rank, context_.ranks);
    control_->initialize(context_.id, 120);
    auto directContext = context_;
    directContext.id = id();
    reference_ = std::make_shared<DirectReference>(directContext, dtype_);
    reference_->prepare();
    auto selectionContext = context_;
    selectionContext.id = id();
    NetworkPreparationOptions options;
    options.dtype = dtype_;
    struct View {
      char host[256]{};
      uint64_t topology = 0;
      size_t visible = 0;
    } view;
    if (gethostname(view.host, sizeof(view.host) - 1))
      throw std::runtime_error("strict host discovery failed");
    auto topology = inspectTopology(context_.localDevices).hardwareIdentity;
    view.topology = 14695981039346656037ULL;
    for (unsigned char c : topology)
      view.topology = (view.topology ^ c) * 1099511628211ULL;
    view.visible = context_.localDevices.size();
    auto views = gather(view);
    for (auto const& own : views) {
      size_t participants = 0;
      for (auto const& other : views)
        if (!std::strcmp(own.host, other.host)) {
          ++participants;
          options.allowHierarchy &= own.topology == other.topology;
        }
      options.allowHierarchy &= own.visible == participants;
    }
    options.allowDirect = false;  // The direct executor IS the reference.
    std::ostringstream identity;
    identity << "strict-direct-reference-v1;artifact="
             << artifactIdentity(
                    reinterpret_cast<uintptr_t>(&preparedDistributedBackend))
             << ";dtype=" << int(dtype_) << ";ranks=" << context_.ranks
             << ";capacity=" << context_.capacity;
    options.fallback = {reference_, dtype_, identity.str(),
                        reference_->configurationIdentity()};
    runtime_ =
        std::make_unique<PreparedNetworkRuntime>(selectionContext, options);
    for (auto const& request : context_.requests) {
      auto key = std::make_pair(int(request.operation), request.count);
      if (plans_.count(key))
        throw std::invalid_argument("duplicate strict distributed shape");
      auto samples = validateReference(request);
      auto plan = runtime_->prepare(request);
      plans_.emplace(key, plan);
      auto const& selected = runtime_->selection(plan);
      if (!context_.rank) {
        std::fprintf(stderr,
                     "UCCL Lite strict distributed plan: op=%d count=%zu "
                     "selected=%s hierarchy_eligible=%d hierarchy_correct=%d "
                     "reference=direct native=0 reference_us=",
                     key.first, key.second,
                     selected.backend == NetworkBackend::Hierarchy ? "hierarchy"
                                                                   : "direct",
                     int(selected.candidateEligible),
                     int(selected.candidateCorrect));
        for (auto us : samples) std::fprintf(stderr, "%.9g,", us);
        std::fprintf(stderr, " paired_reference_us=");
        for (auto us : selected.baselineSamples)
          std::fprintf(stderr, "%.9g,", us);
        std::fprintf(stderr, " hierarchy_us=");
        for (auto us : selected.candidateSamples)
          std::fprintf(stderr, "%.9g,", us);
        std::fprintf(stderr, "\n");
      }
    }
    prepared_ = true;
  }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    if (!prepared_)
      throw std::logic_error("strict distributed provider is not prepared");
    runtime_->enqueue(plans_.at({int(op), count}), input, output, stream);
  }
  void synchronize() override {
    if (runtime_) runtime_->synchronize();
  }
  bool initialized() const override { return prepared_; }
  bool idle() const override { return !runtime_ || runtime_->idle(); }
};
}  // namespace
BackendRegistration preparedDistributedBackend(ScalarType dtype) {
  if (dtype != ScalarType::Float32 && dtype != ScalarType::BFloat16)
    throw std::invalid_argument("unsupported strict distributed dtype");
  BackendSpec spec;
  spec.name = dtype == ScalarType::Float32 ? "prepared-distributed-fp32"
                                           : "prepared-distributed";
  spec.dtype = dtype;
  spec.automatic = true;
  spec.minLocalRanks = 1;
  spec.maxLocalRanks = 32;
  spec.minHosts = 2;
  spec.maxHosts = 32;
  spec.minTotalRanks = 2;
  spec.maxTotalRanks = 32;
  spec.extentMultiple = 1;
  spec.rankUuidProfiles = true;
  return {spec, [] { return PreparedNetworkRuntime::createUniqueId(); },
          [dtype](BackendContext const& context) {
            return std::make_unique<Adapter>(context, dtype);
          }};
}
}  // namespace uccl_lite::runtime
