// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "allgather.hpp"
#include "native_collectives.hpp"
#include <limits>
#include <utility>

namespace {

static mscclpp::CommResult toCommResult(ncclResult_t result) {
  return static_cast<mscclpp::CommResult>(result);
}

static uintptr_t extraValue(
    std::unordered_map<std::string, uintptr_t> const& extras,
    char const* name) {
  auto it = extras.find(name);
  return it == extras.end() ? 0 : it->second;
}

static std::shared_ptr<mscclpp::Communicator> bootstrapCommFromExtra(
    uintptr_t value) {
  if (value == 0) return nullptr;
  return *reinterpret_cast<std::shared_ptr<mscclpp::Communicator>*>(value);
}

class LiteAllgatherAlgorithm final : public mscclpp::Algorithm {
 public:
  LiteAllgatherAlgorithm(std::string name, LiteAllGatherPath path)
      : name_(std::move(name)),
        path_(path),
        tags_({{"lite", 1}}),
        range_({0, std::numeric_limits<size_t>::max()}) {}

  std::string const& name() const override { return name_; }
  std::string const& collective() const override { return collective_; }
  std::pair<size_t, size_t> const& messageRange() const override {
    return range_;
  }
  std::unordered_map<std::string, uint64_t> const& tags() const override {
    return tags_;
  }
  mscclpp::CollectiveBufferMode const& bufferMode() const override {
    return bufferMode_;
  }
  mscclpp::AlgorithmType type() const override {
    return mscclpp::AlgorithmType::Native;
  }
  Constraint constraint() const override { return {}; }
  void setMessageSizeRange(size_t minMessageSize,
                           size_t maxMessageSize) override {
    range_ = {minMessageSize, maxMessageSize};
  }

  mscclpp::CommResult execute(
      std::shared_ptr<mscclpp::Communicator>, void const* input, void* output,
      size_t inputSize, size_t, mscclpp::DataType, mscclpp::ReduceOp,
      cudaStream_t stream, std::shared_ptr<mscclpp::Executor>, int = 0,
      int = 0, bool = false,
      std::unordered_map<std::string, uintptr_t> const& extras = {}) override {
    auto commIt = extras.find("nccl_comm");
    if (commIt == extras.end()) {
      return mscclpp::CommResult::CommInvalidArgument;
    }

    ncclComm_t ncclComm = reinterpret_cast<ncclComm_t>(commIt->second);
    size_t sendcount = static_cast<size_t>(extraValue(extras, "sendcount"));
    ncclDataType_t datatype =
        static_cast<ncclDataType_t>(extraValue(extras, "datatype"));
    int rank = static_cast<int>(extraValue(extras, "rank"));
    int nRanks = static_cast<int>(extraValue(extras, "nranks"));
    int nRanksPerNode =
        static_cast<int>(extraValue(extras, "nranks_per_node"));
    int cudaDevice = static_cast<int>(extraValue(extras, "cuda_device"));
    auto bootstrapComm =
        bootstrapCommFromExtra(extraValue(extras, "bootstrap_comm"));

    switch (path_) {
      case LiteAllGatherPath::SingleNodeCudaIpc: {
        auto fn =
            reinterpret_cast<LiteAllgatherP2pFn>(extraValue(extras, "p2p_fn"));
        if (fn == nullptr) return mscclpp::CommResult::CommInvalidArgument;
        return toCommResult(
            fn(input, output, inputSize, ncclComm, stream, rank, nRanks));
      }
      case LiteAllGatherPath::SingleNodeShm: {
        auto fn = reinterpret_cast<LiteAllgatherHostFn>(
            extraValue(extras, "host_fn"));
        if (fn == nullptr) return mscclpp::CommResult::CommInvalidArgument;
        return toCommResult(fn(input, output, inputSize, ncclComm, stream, rank,
                               nRanks, nRanksPerNode, bootstrapComm,
                               cudaDevice));
      }
      case LiteAllGatherPath::MultiNode:
        return toCommResult(mscclpp::nccl::runLiteAllGather(
            input, output, sendcount, inputSize, datatype, ncclComm, stream,
            rank, nRanks, nRanksPerNode, bootstrapComm, cudaDevice));
    }
    return mscclpp::CommResult::CommInvalidUsage;
  }

  void reset() override {}

 private:
  std::string name_;
  std::string collective_ = "allgather";
  LiteAllGatherPath path_;
  std::unordered_map<std::string, uint64_t> tags_;
  std::pair<size_t, size_t> range_;
  mscclpp::CollectiveBufferMode bufferMode_ =
      mscclpp::CollectiveBufferMode::Any;
};

}  // namespace

LiteAllgatherAlgo::LiteAllgatherAlgo(std::string name, LiteAllGatherPath path)
    : name_(std::move(name)), path_(path) {}

std::shared_ptr<mscclpp::Algorithm> LiteAllgatherAlgo::build() {
  return std::make_shared<LiteAllgatherAlgorithm>(name_, path_);
}
