#include "uccl_lite/collectives/peer_collectives.hpp"
#include "uccl_lite/joint_collectives.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <elf.h>
#include <link.h>

namespace uccl_lite {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct CurrentDevice {
  int device;
  CurrentDevice() { check(cudaGetDevice(&device)); }
  ~CurrentDevice() { cudaSetDevice(device); }
};
}  // namespace

class JointCollectives::Impl {
 public:
  Impl(Topology topology, JointMode mode, BackendPolicy policy,
       cpu_router::CpuRouterOptions options, PeerTuning tuning)
      : peerTuning(std::move(tuning)) {
    if (mode != JointMode::Peer && mode != JointMode::Router &&
        mode != JointMode::Auto)
      throw std::invalid_argument("invalid joint mode");
    if (topology.devices.empty() || topology.devices.size() > 32)
      throw std::invalid_argument(
          "joint collectives require 1..32 distinct devices");
    peerAvailable = JointCollectives::peerSupported(topology);
    if (mode == JointMode::Peer && !peerAvailable)
      throw std::invalid_argument("requested direct-peer topology unavailable");
    policy.mode = mode == JointMode::Router ? RouterMode::Force
                  : mode == JointMode::Peer ? RouterMode::Off
                                            : RouterMode::Auto;
    policy.fallbackIdentity = JointCollectives::peerIdentity();
    if (policy.mode == RouterMode::Auto &&
        policy.fallbackIdentity.find("build=unavailable") != std::string::npos)
      policy.measurements.clear();
    dispatcher = std::make_unique<CollectiveDispatcher>(
        std::move(topology), std::move(policy),
        [this](CollectiveCall const& call) {
          if (!peer)
            throw std::logic_error("prepare direct-peer before enqueue");
          peer->enqueue(call);
        },
        std::move(options));
  }
  void validate(CollectiveCall const& call) const {
    if (call.collective != Collective::AllReduce &&
        call.collective != Collective::AllGather &&
        call.collective != Collective::ReduceScatter)
      throw std::invalid_argument("invalid collective operation");
    if (call.dtype != ScalarType::BFloat16 || call.inPlace || !call.count)
      throw std::invalid_argument(
          "joint supports positive BF16 out-of-place calls");
    auto w = call.workload();
    if (w.alignment < 2)
      throw std::invalid_argument("BF16 pointers must have two-byte alignment");
  }
  void rejectCapture(CollectiveCall const& call) const {
    CurrentDevice restore;
    for (auto const& port : call.ports) {
      check(cudaSetDevice(port.device));
      cudaStreamCaptureStatus status;
      check(cudaStreamIsCapturing(port.stream, &status));
      if (status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument(
            "joint collectives currently support eager enqueue only");
    }
  }
  std::unique_ptr<PeerCollectives> peer;
  std::unique_ptr<CollectiveDispatcher> dispatcher;
  bool peerAvailable = false;
  PeerTuning peerTuning;
};

JointCollectives::JointCollectives(Topology topology, JointMode mode,
                                   BackendPolicy policy,
                                   cpu_router::CpuRouterOptions options,
                                   PeerTuning tuning)
    : impl_(std::make_unique<Impl>(std::move(topology), mode, std::move(policy),
                                   std::move(options), std::move(tuning))) {}
JointCollectives::~JointCollectives() = default;
BackendDecision JointCollectives::decision(CollectiveCall const& call) const {
  impl_->validate(call);
  auto result = impl_->dispatcher->decision(call);
  if (!result.cpuRouter && !impl_->peerAvailable)
    throw std::invalid_argument(
        "no admitted collective backend: peer unavailable and router lacks "
        "matching calibration; explicit Router mode is for validation only");
  return result;
}
BackendDecision JointCollectives::prepare(CollectiveCall const& call) {
  CurrentDevice restore;
  auto selected = decision(call);
  impl_->rejectCapture(call);
  if (!selected.cpuRouter) {
    if (!impl_->peer)
      impl_->peer =
          std::make_unique<PeerCollectives>(topology(), impl_->peerTuning);
    impl_->peer->prepare(call);
  }
  return impl_->dispatcher->prepare(call);
}
BackendDecision JointCollectives::enqueue(CollectiveCall const& call) {
  CurrentDevice restore;
  decision(call);
  impl_->rejectCapture(call);
  return impl_->dispatcher->enqueue(call);
}
Topology const& JointCollectives::topology() const {
  return impl_->dispatcher->topology();
}
PeerSelection const& JointCollectives::peerPlan(
    CollectiveCall const& call) const {
  if (!impl_->peer) throw std::logic_error("peer executor is not prepared");
  return impl_->peer->plan(call);
}
bool JointCollectives::routerInitialized() const {
  return impl_->dispatcher->routerInitialized();
}
std::string JointCollectives::peerIdentity() {
  static std::string const identity = [] {
    struct Image {
      uintptr_t address;
      std::string id;
    } image{reinterpret_cast<uintptr_t>(&JointCollectives::peerIdentity), {}};
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* context) {
          auto& image = *static_cast<Image*>(context);
          uintptr_t base = info->dlpi_addr;
          bool contains = false;
          for (int i = 0; i < info->dlpi_phnum; ++i) {
            auto const& p = info->dlpi_phdr[i];
            uintptr_t segment = base + p.p_vaddr;
            contains |= p.p_type == PT_LOAD && image.address >= segment &&
                        image.address - segment < p.p_memsz;
          }
          if (!contains) return 0;
          for (int i = 0; i < info->dlpi_phnum; ++i) {
            auto const& p = info->dlpi_phdr[i];
            if (p.p_type != PT_NOTE) continue;
            auto cursor =
                reinterpret_cast<unsigned char const*>(base + p.p_vaddr);
            auto end = cursor + p.p_filesz;
            while (size_t(end - cursor) >= sizeof(ElfW(Nhdr))) {
              auto const& note = *reinterpret_cast<ElfW(Nhdr) const*>(cursor);
              cursor += sizeof(ElfW(Nhdr));
              size_t names = (size_t(note.n_namesz) + 3) & ~size_t(3);
              size_t descs = (size_t(note.n_descsz) + 3) & ~size_t(3);
              if (names > size_t(end - cursor)) break;
              auto name = cursor;
              cursor += names;
              if (descs > size_t(end - cursor)) break;
              if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
                  std::memcmp(name, "GNU", 4) == 0 && note.n_descsz > 0 &&
                  note.n_descsz <= 64) {
                char const* digits = "0123456789abcdef";
                for (unsigned j = 0; j < note.n_descsz; ++j) {
                  image.id += digits[cursor[j] >> 4];
                  image.id += digits[cursor[j] & 15];
                }
                return 1;
              }
              cursor += descs;
            }
          }
          return 1;
        },
        &image);
    return std::string(
               "uccl-lite-joint-bf16-peer-coordinator-direct-self-v13;build=") +
           (image.id.empty() ? "unavailable" : image.id);
  }();
  return identity;
}
bool JointCollectives::peerSupported(Topology const& topology) {
  return PeerCollectives::supported(topology);
}
}  // namespace uccl_lite
