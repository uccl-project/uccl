#include "uccl_lite/collectives/direct_exchange.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
namespace uccl_lite::runtime {
namespace {
class DirectAdapter final : public CollectiveBackend {
  DirectExchangeCollectives executor_;

 public:
  DirectAdapter(BackendContext const& c, ScalarType dtype)
      : executor_(c.id, c.rank, c.ranks, c.device, c.capacity, dtype) {}
  void prepare() override {
    if (!executor_.initialized()) executor_.prepare();
  }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    executor_.enqueue(op, input, output, count, stream);
  }
  void synchronize() override { executor_.synchronize(); }
  bool initialized() const override { return executor_.initialized(); }
  std::shared_ptr<DirectModelCalibration const> calibrate() {
    return executor_.calibrateModel();
  }
  std::string configurationIdentity() const override {
    return executor_.identity();
  }
  std::map<std::string, size_t> traffic(NetworkOperation op,
                                        size_t count) const override {
    auto t = executor_.traffic(op, count);
    return {{"chunks", t.chunks},
            {"gpu_memory_bytes", t.gpuMemoryBytes},
            {"peer_payload_bytes", t.peerPayloadBytes},
            {"network_payload_bytes", t.networkPayloadBytes},
            {"control_payload_bytes", t.controlPayloadBytes},
            {"input_copies", t.inputCopies},
            {"self_copies", t.selfCopies},
            {"output_copies", t.outputCopies},
            {"reduction_launches", t.reductionLaunches},
            {"caller_stream_synchronizations", t.callerStreamSynchronizations},
            {"transport_flushes", t.transportFlushes},
            {"agreements", t.agreements}};
  }
  bool idle() const override {
    return true;
  }  // Synchronous serialized host calls.
};
}  // namespace
BackendRegistration directExchangeBackend(ScalarType dtype) {
  if (dtype != ScalarType::Float32 && dtype != ScalarType::BFloat16)
    throw std::invalid_argument("unsupported direct exchange scalar type");
  BackendSpec spec;
  spec.name = dtype == ScalarType::Float32 ? "direct-exchange-fp32"
                                           : "direct-exchange-bf16";
  spec.dtype = dtype;
  spec.extentMultiple = 1;
  spec.minLocalRanks = 1;
  spec.maxLocalRanks = 32;
  spec.minHosts = 1;
  spec.maxHosts = 32;
  spec.minTotalRanks = 2;
  spec.maxTotalRanks = 32;
  return {spec, [] { return DirectExchangeCollectives::createUniqueId(); },
          [dtype](BackendContext const& c) {
            return std::make_unique<DirectAdapter>(c, dtype);
          }};
}
std::shared_ptr<DirectModelCalibration const> calibrateDirectExchange(
    CollectiveBackend& backend) {
  auto* direct = dynamic_cast<DirectAdapter*>(&backend);
  if (!direct)
    throw std::invalid_argument("not a direct-exchange model provider");
  return direct->calibrate();
}
}  // namespace uccl_lite::runtime
