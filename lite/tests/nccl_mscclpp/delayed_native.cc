// Test-only native shim: keep numerical/native behavior, add a fixed host
// scheduling delay to force ABI transition coverage. Not performance evidence.
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <dlfcn.h>
#include <nccl.h>
namespace {
void* library() {
  static void* p = [] {
    auto path = std::getenv("UCCL_LITE_TEST_REAL_NCCL");
    if (!path) std::abort();
    auto h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) std::abort();
    return h;
  }();
  return p;
}
template <class T>
T real(char const* name) {
  auto p = dlsym(library(), name);
  if (!p) std::abort();
  return reinterpret_cast<T>(p);
}
void delay() { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
}  // namespace
extern "C" {
ncclResult_t ncclGetVersion(int* v) {
  return real<decltype(&ncclGetVersion)>("ncclGetVersion")(v);
}
char const* ncclGetErrorString(ncclResult_t r) {
  return real<decltype(&ncclGetErrorString)>("ncclGetErrorString")(r);
}
ncclResult_t ncclCommCount(ncclComm_t c, int* n) {
  return real<decltype(&ncclCommCount)>("ncclCommCount")(c, n);
}
ncclResult_t ncclCommUserRank(ncclComm_t c, int* n) {
  return real<decltype(&ncclCommUserRank)>("ncclCommUserRank")(c, n);
}
ncclResult_t ncclCommCuDevice(ncclComm_t c, int* n) {
  return real<decltype(&ncclCommCuDevice)>("ncclCommCuDevice")(c, n);
}
ncclResult_t ncclAllReduce(void const* in, void* out, size_t count,
                           ncclDataType_t dtype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  auto r = real<decltype(&ncclAllReduce)>("ncclAllReduce")(
      in, out, count, dtype, op, comm, stream);
  delay();
  return r;
}
ncclResult_t ncclAllGather(void const* in, void* out, size_t count,
                           ncclDataType_t dtype, ncclComm_t comm,
                           cudaStream_t stream) {
  auto r = real<decltype(&ncclAllGather)>("ncclAllGather")(in, out, count,
                                                           dtype, comm, stream);
  delay();
  return r;
}
ncclResult_t ncclReduceScatter(void const* in, void* out, size_t count,
                               ncclDataType_t dtype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
  auto r = real<decltype(&ncclReduceScatter)>("ncclReduceScatter")(
      in, out, count, dtype, op, comm, stream);
  delay();
  return r;
}
}
