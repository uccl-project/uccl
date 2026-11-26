#include "uccl_lite/collectives/peer_staged.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include <stdexcept>

namespace uccl_lite::runtime {
namespace {
class Adapter : public CollectiveBackend {
  PeerStagedCollectives executor_;

 public:
  explicit Adapter(BackendContext const& c, ScalarType dtype)
      : executor_(c.id, c.rank, c.device, c.ranks, c.capacity, dtype) {}
  Adapter(BackendContext const& c, ScalarType dtype, std::vector<int> parents)
      : executor_(c.id, c.rank, c.device, c.ranks, c.capacity, dtype,
                  std::move(parents)) {}
  void prepare() override { executor_.prepare(); }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    executor_.enqueue(op, input, output, count, stream);
  }
  void synchronize() override { executor_.synchronize(); }
  bool initialized() const override { return executor_.initialized(); }
  bool idle() const override { return executor_.idle(); }
  bool supportsGraphs() const override { return true; }
  size_t prepareGraph(NetworkOperation op, void const* input, void* output,
                      size_t count) override {
    return executor_.prepareGraph({{op, input, output, count}});
  }
  void launchGraph(size_t graph, cudaStream_t stream) override {
    executor_.launchGraph(graph, stream);
  }
  void releaseGraph(size_t graph) override { executor_.releaseGraph(graph); }
};
}  // namespace
BackendRegistration peerStagedBackend() {
  return peerStagedBackend(ScalarType::BFloat16);
}
BackendRegistration peerStagedBackend(ScalarType dtype) {
  if (dtype != ScalarType::BFloat16 && dtype != ScalarType::Float32)
    throw std::invalid_argument("unsupported staged peer dtype");
  return {{dtype == ScalarType::Float32 ? "peer-staged-fp32" : "peer-staged",
           "",
           {},
           dtype,
           true,
           false,
           2,
           32},
          [] { return PeerStagedCollectives::createUniqueId(); },
          [dtype](BackendContext const& c) {
            return std::make_unique<Adapter>(c, dtype);
          }};
}
BackendRegistration peerTreeBackend(ScalarType dtype,
                                    std::vector<int> parents) {
  if (dtype != ScalarType::BFloat16 && dtype != ScalarType::Float32)
    throw std::invalid_argument("unsupported peer tree dtype");
  return {{dtype == ScalarType::Float32 ? "peer-tree-fp32" : "peer-tree",
           "",
           {},
           dtype,
           false,
           false,
           2,
           32},
          [] { return PeerStagedCollectives::createUniqueId(); },
          [dtype, parents = std::move(parents)](BackendContext const& c) {
            return std::make_unique<Adapter>(c, dtype, parents);
          }};
}
}  // namespace uccl_lite::runtime
