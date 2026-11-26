#include "native_fallback.hpp"
#include "uccl_lite/runtime/build_identity.hpp"
#include <cstring>
#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>
#include <dlfcn.h>

extern char** environ;
namespace uccl_lite::integration {
namespace {
void checked(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
std::string environment() {
  std::set<std::string> entries;
  for (char** item = environ; *item; ++item)
    if (!std::strncmp(*item, "NCCL_", 5)) entries.insert(*item);
  std::ostringstream out;
  for (auto const& entry : entries) out << entry.size() << ':' << entry << ';';
  return out.str();
}
struct Api {
  void* library = nullptr;
  decltype(&ncclGetVersion) version;
  decltype(&ncclGetErrorString) error;
  decltype(&ncclCommCount) count;
  decltype(&ncclCommUserRank) rank;
  decltype(&ncclCommCuDevice) device;
  decltype(&ncclAllReduce) allReduce;
  decltype(&ncclAllGather) allGather;
  decltype(&ncclReduceScatter) reduceScatter;
  std::string build;
  template <class T>
  T symbol(char const* name) {
    auto address = dlsym(library, name);
    if (!address)
      throw std::runtime_error(std::string("native NCCL symbol unavailable: ") +
                               name);
    auto identity =
        runtime::artifactIdentity(reinterpret_cast<uintptr_t>(address));
    if (build.empty())
      build = identity;
    else if (build != identity)
      throw std::invalid_argument(
          "native NCCL API resolves to different implementations");
    return reinterpret_cast<T>(address);
  }
  explicit Api(std::string const& path) {
    auto file = std::filesystem::canonical(path);
    library = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    try {
      if (dlsym(library, "uccl_lite_nccl_runtime_backends"))
        throw std::invalid_argument(
            "strict UCCL runtime is not a native NCCL fallback");
      version = symbol<decltype(version)>("ncclGetVersion");
      error = symbol<decltype(error)>("ncclGetErrorString");
      count = symbol<decltype(count)>("ncclCommCount");
      rank = symbol<decltype(rank)>("ncclCommUserRank");
      device = symbol<decltype(device)>("ncclCommCuDevice");
      allReduce = symbol<decltype(allReduce)>("ncclAllReduce");
      allGather = symbol<decltype(allGather)>("ncclAllGather");
      reduceScatter = symbol<decltype(reduceScatter)>("ncclReduceScatter");
    } catch (...) {
      dlclose(library);
      library = nullptr;
      throw;
    }
  }
  ~Api() {
    if (library) dlclose(library);
  }
  void check(ncclResult_t result) const {
    if (result != ncclSuccess)
      throw std::runtime_error(std::string("native NCCL: ") + error(result));
  }
};
class Backend : public runtime::CollectiveBackend {
  runtime::BackendContext context_;
  ncclComm_t comm_;
  std::shared_ptr<Api> api_;
  ncclDataType_t dtype_;
  std::string environment_;
  cudaEvent_t done_ = nullptr;
  cudaStream_t last_ = nullptr;
  bool recorded_ = false, prepared_ = false;
  bool graphSupported_ = false;
  std::vector<cudaGraphExec_t> graphs_;

 public:
  Backend(runtime::BackendContext context, ncclComm_t comm,
          std::shared_ptr<Api> api, ncclDataType_t dtype, std::string env,
          int version)
      : context_(std::move(context)),
        comm_(comm),
        api_(std::move(api)),
        dtype_(dtype),
        environment_(std::move(env)),
        graphSupported_(version >= 20900) {}
  ~Backend() {
    if (!done_) return;
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(context_.device);
    if (recorded_) cudaEventSynchronize(done_);
    for (auto graph : graphs_)
      if (graph) cudaGraphExecDestroy(graph);
    cudaEventDestroy(done_);
    if (previous >= 0 && previous != context_.device) cudaSetDevice(previous);
  }
  void prepare() override {
    if (environment() != environment_)
      throw std::invalid_argument(
          "native NCCL environment changed after fallback construction");
    int count = -1, rank = -1, device = -1;
    api_->check(api_->count(comm_, &count));
    api_->check(api_->rank(comm_, &rank));
    api_->check(api_->device(comm_, &device));
    if (count != context_.ranks || rank != context_.rank ||
        device != context_.device)
      throw std::invalid_argument(
          "caller native communicator does not match prepared participants");
    if (!prepared_) {
      checked(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
      prepared_ = true;
    }
  }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    chain(stream);
    submit(op, input, output, count, stream);
    record(stream);
  }
  void chain(cudaStream_t stream) {
    if (!prepared_) throw std::logic_error("native fallback not prepared");
    int device;
    checked(cudaGetDevice(&device));
    if (device != context_.device)
      throw std::invalid_argument("native fallback current device mismatch");
    cudaStreamCaptureStatus capture;
    checked(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument(
          "prepared fallback capture is not qualified; use the caller's native "
          "API outside this runtime");
    if (recorded_ && stream != last_)
      checked(cudaStreamWaitEvent(stream, done_, 0));
  }
  void submit(NetworkOperation op, void const* input, void* output,
              size_t count, cudaStream_t stream) {
    if (op == NetworkOperation::AllReduce)
      api_->check(api_->allReduce(input, output, count, dtype_, ncclSum, comm_,
                                  stream));
    else if (op == NetworkOperation::AllGather)
      api_->check(api_->allGather(input, output, count, dtype_, comm_, stream));
    else if (op == NetworkOperation::ReduceScatter)
      api_->check(api_->reduceScatter(input, output, count, dtype_, ncclSum,
                                      comm_, stream));
    else
      throw std::invalid_argument("unsupported native fallback operation");
  }
  void record(cudaStream_t stream) {
    checked(cudaEventRecord(done_, stream));
    recorded_ = true;
    last_ = stream;
  }
  bool supportsGraphs() const override { return graphSupported_; }
  size_t prepareGraph(NetworkOperation op, void const* input, void* output,
                      size_t count) override {
    if (!supportsGraphs() || !prepared_ || !idle())
      throw std::logic_error("native graph preparation requires idle backend");
    graphs_.reserve(graphs_.size() + 1);
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    try {
      checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      submit(op, input, output, count, stream);
      checked(cudaStreamEndCapture(stream, &graph));
      checked(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    } catch (...) {
      if (stream) {
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
            capture != cudaStreamCaptureStatusNone) {
          cudaGraph_t abandoned = nullptr;
          cudaStreamEndCapture(stream, &abandoned);
          if (abandoned) cudaGraphDestroy(abandoned);
        }
        cudaStreamDestroy(stream);
      }
      if (graph) cudaGraphDestroy(graph);
      if (executable) cudaGraphExecDestroy(executable);
      throw;
    }
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    graphs_.push_back(executable);
    return graphs_.size() - 1;
  }
  void launchGraph(size_t graph, cudaStream_t stream) override {
    if (graph >= graphs_.size() || !graphs_[graph])
      throw std::invalid_argument("native graph is absent or released");
    chain(stream);
    checked(cudaGraphLaunch(graphs_[graph], stream));
    record(stream);
  }
  void releaseGraph(size_t graph) override {
    if (graph >= graphs_.size() || !graphs_[graph] || !idle())
      throw std::invalid_argument("native graph release requires idle handle");
    checked(cudaGraphExecDestroy(graphs_[graph]));
    graphs_[graph] = nullptr;
  }
  void synchronize() override {
    if (recorded_) checked(cudaEventSynchronize(done_));
  }
  bool initialized() const override { return prepared_; }
  bool idle() const override {
    return !recorded_ || cudaEventQuery(done_) == cudaSuccess;
  }
};
}  // namespace
runtime::RuntimeContext nativeNcclContext(ncclComm_t communicator,
                                          std::string const& nativeLibrary,
                                          NetworkId const& id,
                                          size_t capacity) {
  if (!communicator || nativeLibrary.empty())
    throw std::invalid_argument(
        "native context needs a communicator and library");
  Api api(nativeLibrary);
  runtime::RuntimeContext context;
  context.id = id;
  context.capacity = capacity;
  api.check(api.count(communicator, &context.ranks));
  api.check(api.rank(communicator, &context.rank));
  api.check(api.device(communicator, &context.device));
  return context;
}
runtime::CallerFallback makeNativeNcclFallback(
    runtime::BackendContext const& context, ncclComm_t communicator,
    std::string const& nativeLibrary, std::string const& configurationIdentity,
    ScalarType dtype) {
  if (!communicator || nativeLibrary.empty() || configurationIdentity.empty())
    throw std::invalid_argument(
        "native fallback needs a communicator, library and configuration "
        "identity");
  ncclDataType_t type;
  switch (dtype) {
    case ScalarType::BFloat16:
      type = ncclBfloat16;
      break;
    case ScalarType::Float16:
      type = ncclFloat16;
      break;
    case ScalarType::Float32:
      type = ncclFloat32;
      break;
    default:
      throw std::invalid_argument("native fallback dtype unsupported");
  }
  auto api = std::make_shared<Api>(nativeLibrary);
  int version = 0;
  api->check(api->version(&version));
  std::string identity =
      "native-nccl:" + api->build + ";version=" + std::to_string(version) +
      ";dtype=" + std::to_string(int(dtype)) + ";adapter=" +
      runtime::artifactIdentity(reinterpret_cast<uintptr_t>(&environment)) +
      ";config=" + configurationIdentity;
  auto env = environment();
  return {std::make_shared<Backend>(context, communicator, std::move(api), type,
                                    env, version),
          dtype, std::move(identity), std::move(env)};
}
runtime::CallerFallback makeNativeNcclFallback(
    runtime::RuntimeContext const& context, ncclComm_t communicator,
    std::string const& nativeLibrary, std::string const& configurationIdentity,
    ScalarType dtype) {
  return makeNativeNcclFallback(context.backendContext(), communicator,
                                nativeLibrary, configurationIdentity, dtype);
}
}  // namespace uccl_lite::integration
