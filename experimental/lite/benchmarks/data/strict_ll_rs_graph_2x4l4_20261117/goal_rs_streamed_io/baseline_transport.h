#pragma once
// Private MSCCL++ transport ownership shared by the network library/harness.
#include "device_collectives.cuh"
namespace uccl_lite::multinode_kernels {
inline mscclpp::Transport ibTransport(int index) {
  if (index < 0 || index >= 8) throw std::invalid_argument("invalid IB rail");
  return static_cast<mscclpp::Transport>(
      static_cast<int>(mscclpp::Transport::IB0) + index);
}
inline int ringSuccessor(int localRank) {
  constexpr int successor[4] = {1, 3, 0, 2};
  return successor[localRank];
}
inline int ringPredecessor(int localRank) {
  constexpr int predecessor[4] = {2, 0, 3, 1};
  return predecessor[localRank];
}
class GpuTransport {
 public:
  std::shared_ptr<mscclpp::ProxyService> localProxy, ibProxy;
  LocalAgChans localAg{};
  LocalDirectChans localDirect{};
  PortChan localSend{}, localRecv{}, ib{}, duplicateIb{};

  GpuTransport(mscclpp::Communicator& comm, int rank, int localRank, int rail,
               char* buffer, size_t bufferBytes, bool direct, bool duplicate,
               bool localDma = true, bool transportScopedMemory = false) {
    try {
      auto incoming = cudaGetLastError();
      if (incoming != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(incoming));
      localMemory_ = comm.registerMemory(
          buffer, bufferBytes,
          transportScopedMemory
              ? mscclpp::TransportFlags(mscclpp::Transport::CudaIpc)
              : mscclpp::Transport::CudaIpc | ibTransport(rail));
      // Cross-host descriptors must not invite a CUDA IPC import of a remote
      // host's allocation. Keep the retained benchmark's registration default.
      ibMemory_ =
          transportScopedMemory
              ? comm.registerMemory(buffer, bufferBytes, ibTransport(rail))
              : localMemory_;

      int nodeBase = rank - localRank;
      int cross = rank ^ 4;
      std::array<int, 4> peers{nodeBase + ringSuccessor(localRank),
                               nodeBase + ringPredecessor(localRank),
                               nodeBase + (localRank ^ 3), cross};
      std::array<std::shared_future<mscclpp::Connection>, 4> connections;
      std::array<std::shared_future<mscclpp::RegisteredMemory>, 4> memories;
      for (int i = 0; i < 4; ++i) {
        connections[i] = comm.connect(
            i < 3 ? mscclpp::Transport::CudaIpc : ibTransport(rail), peers[i],
            i < 3 ? 100 : 102);
        comm.sendMemory(i < 3 ? localMemory_ : ibMemory_, peers[i],
                        i < 3 ? 200 : 202);
        memories[i] = comm.recvMemory(peers[i], i < 3 ? 200 : 202);
      }
      std::shared_future<mscclpp::Connection> extraConnection;
      std::shared_future<mscclpp::RegisteredMemory> extraMemory;
      if (duplicate) {
        extraConnection = comm.connect(ibTransport(rail), cross ^ 1, 104);
        comm.sendMemory(ibMemory_, cross ^ 1, 204);
        extraMemory = comm.recvMemory(cross ^ 1, 204);
      }
      for (int i = 0; i < 4; ++i) {
        connections_[i] = connections[i].get();
        memories_[i] = memories[i].get();
      }
      // All connection/memory futures returned successfully. The retained
      // MSCCL++ IPC import can leave a handled probe result in CUDART's
      // thread-local error slot. Consume that synchronous setup status here;
      // otherwise cudaGetLastError after our first successful kernel reports
      // the old IPC probe instead of that launch. Never consume device faults.
      auto ipcStatus = cudaGetLastError();
      if (ipcStatus != cudaSuccess &&
          ipcStatus != cudaErrorInvalidResourceHandle &&
          ipcStatus != cudaErrorPeerAccessAlreadyEnabled)
        throw std::runtime_error(cudaGetErrorString(ipcStatus));
      if (ipcStatus != cudaSuccess)
        std::fprintf(stderr,
                     "UCCL Lite network rank=%d: completed IPC import with "
                     "handled setup status %s\n",
                     rank, cudaGetErrorString(ipcStatus));
      auto synchronized = cudaDeviceSynchronize();
      if (synchronized != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(synchronized));
      localProxy = std::make_shared<mscclpp::ProxyService>();
      ibProxy = std::make_shared<mscclpp::ProxyService>();
      std::array<std::shared_future<mscclpp::Semaphore>, 4> semaphores;
      for (int i = 0; i < 4; ++i)
        semaphores[i] =
            comm.buildSemaphore(connections_[i], peers[i], i < 3 ? 300 : 302);
      for (int i = 0; i < 4; ++i) {
        auto proxy = i < 3 ? localProxy : ibProxy;
        auto semaphore = proxy->addSemaphore(semaphores[i].get());
        auto local = proxy->addMemory(i < 3 ? localMemory_ : ibMemory_);
        auto remote = proxy->addMemory(memories_[i]);
        channels_[i] = std::make_unique<mscclpp::PortChannel>(
            proxy->portChannel(semaphore, remote, local));
      }

      if (duplicate) {
        auto connection = extraConnection.get();
        auto memory = extraMemory.get();
        auto semaphore = comm.buildSemaphore(connection, cross ^ 1, 304);
        auto sem = ibProxy->addSemaphore(semaphore.get());
        auto remote = ibProxy->addMemory(memory);
        auto local = ibProxy->addMemory(ibMemory_);
        extra_ = std::make_unique<mscclpp::PortChannel>(
            ibProxy->portChannel(sem, remote, local));
        duplicateIb = extra_->deviceHandle();
      }
      localSend = channels_[0]->deviceHandle();
      localRecv = channels_[1]->deviceHandle();
      ib = channels_[3]->deviceHandle();
      for (int i = 0; i < 3; ++i)
        localAg.c[peers[i] - nodeBase] = channels_[i]->deviceHandle();
      if (direct) {
        std::array<std::shared_future<mscclpp::Semaphore>, 3> ready;
        for (int i = 0; i < 3; ++i)
          ready[i] = comm.buildSemaphore(connections_[i], peers[i], 320);
        for (int i = 0; i < 3; ++i) {
          direct_[i] = std::make_unique<mscclpp::MemoryChannel>(
              ready[i].get(), memories_[i], localMemory_, buffer);
          localDirect.c[peers[i] - nodeBase] = direct_[i]->deviceHandle();
        }
      }

      if (localDma) {
        localProxy->startProxy();
        localRunning_ = true;
      }
      ibProxy->startProxy();
      ibRunning_ = true;

    } catch (...) {
      stop();
      throw;
    }
  }
  ~GpuTransport() { stop(); }
  void stop() {
    if (ibRunning_) {
      ibProxy->stopProxy();
      ibRunning_ = false;
    }
    if (localRunning_) {
      localProxy->stopProxy();
      localRunning_ = false;
    }
  }

 private:
  bool localRunning_ = false, ibRunning_ = false;
  mscclpp::RegisteredMemory localMemory_;
  mscclpp::RegisteredMemory ibMemory_;
  std::array<mscclpp::Connection, 4> connections_;
  std::array<mscclpp::RegisteredMemory, 4> memories_;
  std::array<std::unique_ptr<mscclpp::PortChannel>, 4> channels_;
  std::unique_ptr<mscclpp::PortChannel> extra_;
  std::array<std::unique_ptr<mscclpp::MemoryChannel>, 3> direct_;
};
}  // namespace uccl_lite::multinode_kernels
