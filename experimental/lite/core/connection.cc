// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "connection.hpp"

#if defined(ENABLE_NPKIT)
#include "npkit.hpp"
#endif

#include "api.h"
#include "context.hpp"
#include "endpoint.hpp"
#include "gpu_utils_internal.hpp"
#include "logger.hpp"
#include "numa.hpp"
#include "utils.hpp"
#include <sstream>
#include <thread>

namespace mscclpp {

static void validateTransport(RegisteredMemory mem, Transport transport,
                              uint64_t offset = 0, uint64_t size = 0) {
  if (!mem.transports().has(transport)) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "RegisteredMemory does not support this transport");
  }
  if (offset + size > mem.size()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "RegisteredMemory out of bounds");
  }
}

static bool isSameProcess(Endpoint const& a, Endpoint const& b) {
  return a.hostHash() == b.hostHash() && a.pidHash() == b.pidHash();
}

// BaseConnection

Endpoint::Impl const& BaseConnection::getImpl(Endpoint const& endpoint) {
  return *(endpoint.pimpl_);
}

RegisteredMemory::Impl const& BaseConnection::getImpl(
    RegisteredMemory const& memory) {
  return *(memory.pimpl_);
}

Context::Impl& BaseConnection::getImpl(Context& context) {
  return *(context.pimpl_);
}

MSCCLPP_API_CPP BaseConnection::BaseConnection(std::shared_ptr<Context> context,
                                               Endpoint const& localEndpoint)
    : context_(context),
      localEndpoint_(localEndpoint),
      maxWriteQueueSize_(localEndpoint.maxWriteQueueSize()) {}

MSCCLPP_API_CPP std::shared_ptr<Context> BaseConnection::context() const {
  return context_;
}

MSCCLPP_API_CPP Device const& BaseConnection::localDevice() const {
  return localEndpoint_.device();
}

MSCCLPP_API_CPP int BaseConnection::getMaxWriteQueueSize() const {
  return maxWriteQueueSize_;
}

// Connection wrapper

Connection::Connection(std::shared_ptr<BaseConnection> impl) : impl_(impl) {}

MSCCLPP_API_CPP void Connection::write(RegisteredMemory dst, uint64_t dstOffset,
                                       RegisteredMemory src, uint64_t srcOffset,
                                       uint64_t size) {
  impl_->write(dst, dstOffset, src, srcOffset, size);
}

MSCCLPP_API_CPP void Connection::updateAndSync(RegisteredMemory dst,
                                               uint64_t dstOffset,
                                               uint64_t* src,
                                               uint64_t newValue) {
  impl_->updateAndSync(dst, dstOffset, src, newValue);
}

MSCCLPP_API_CPP void Connection::flush(int64_t timeoutUsec) {
  impl_->flush(timeoutUsec);
}

MSCCLPP_API_CPP Transport Connection::transport() const {
  return impl_->transport();
}

MSCCLPP_API_CPP Transport Connection::remoteTransport() const {
  return impl_->remoteTransport();
}

MSCCLPP_API_CPP std::shared_ptr<Context> Connection::context() const {
  return impl_->context();
}

MSCCLPP_API_CPP Device const& Connection::localDevice() const {
  return impl_->localDevice();
}

MSCCLPP_API_CPP int Connection::getMaxWriteQueueSize() const {
  return impl_->getMaxWriteQueueSize();
}

MSCCLPP_API_CPP std::shared_ptr<IbQp> Connection::getIbQp() const {
  auto* ibConn = dynamic_cast<IBConnection*>(impl_.get());
  return ibConn ? ibConn->getIbQp() : nullptr;
}

// CudaIpcConnection

CudaIpcConnection::CudaIpcConnection(std::shared_ptr<Context> context,
                                     Endpoint const& localEndpoint,
                                     Endpoint const& remoteEndpoint)
    : BaseConnection(context, localEndpoint) {
  if (localEndpoint.transport() != Transport::CudaIpc ||
      remoteEndpoint.transport() != Transport::CudaIpc) {
    THROW(CONN, Error, ErrorCode::InternalError,
          "CudaIpc transport is required for CudaIpcConnection");
  }
  if (localEndpoint.device().type == DeviceType::GPU &&
      localEndpoint.device().id < 0) {
    THROW(CONN, Error, ErrorCode::InternalError,
          "No GPU device ID provided for local endpoint");
  }
  if (remoteEndpoint.device().type == DeviceType::GPU &&
      remoteEndpoint.device().id < 0) {
    THROW(CONN, Error, ErrorCode::InternalError,
          "No GPU device ID provided for remote endpoint");
  }
  int localDeviceId = localEndpoint.device().id;
  int remoteDeviceId = remoteEndpoint.device().id;
  if (localEndpoint.device().type != DeviceType::GPU &&
      remoteEndpoint.device().type != DeviceType::GPU) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "CudaIpcConnection requires at least one GPU endpoint");
  } else if (localEndpoint.device().type == DeviceType::GPU &&
             remoteEndpoint.device().type == DeviceType::GPU) {
    if (isSameProcess(localEndpoint, remoteEndpoint) &&
        localDeviceId != remoteDeviceId) {
      // Connecting two GPUs in the same process - need to enable peer access
      // explicitly
      CudaDeviceGuard deviceGuard(localDeviceId);
      auto ret = cudaDeviceEnablePeerAccess(remoteDeviceId, 0);
      if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
        MSCCLPP_CUDATHROW(ret);
      }
    }
  }
  int streamDeviceId = (localEndpoint.device().type == DeviceType::GPU)
                           ? localDeviceId
                           : remoteDeviceId;
  auto& ctxImpl = getImpl(*context);
#if defined(MSCCLPP_DEVICE_HIP)
  ctxImpl.ipcStreams_.emplace_back(
      std::make_shared<CudaIpcStream>(streamDeviceId));
#else   // !defined(MSCCLPP_DEVICE_HIP)
  if (ctxImpl.ipcStreams_.empty()) {
    ctxImpl.ipcStreams_.emplace_back(
        std::make_shared<CudaIpcStream>(streamDeviceId));
  }
#endif  // !defined(MSCCLPP_DEVICE_HIP)
  stream_ = ctxImpl.ipcStreams_.back();
}

Transport CudaIpcConnection::transport() const { return Transport::CudaIpc; }

Transport CudaIpcConnection::remoteTransport() const {
  return Transport::CudaIpc;
}

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset,
                              RegisteredMemory src, uint64_t srcOffset,
                              uint64_t size) {
#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_WRITE_ENTRY, uint32_t(size),
                         0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  char* dstPtr = (char*)dst.data();
  char* srcPtr = (char*)src.data();

  stream_->memcpyD2D(dstPtr + dstOffset, srcPtr + srcOffset, size);

  INFO(CONN, "CudaIpcConnection write: from ", srcPtr + srcOffset, " to ",
       dstPtr + dstOffset, ", size ", size);

#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_WRITE_EXIT, uint32_t(size),
                         0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void CudaIpcConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                                      uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_ENTRY, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport());
  uint64_t oldValue = *src;
  *src = newValue;
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(
      reinterpret_cast<char*>(dst.data()) + dstOffset);

  stream_->memcpyH2D(dstPtr + dstOffset, src, sizeof(uint64_t));

  INFO(CONN, "CudaIpcConnection atomic write: from ", src, " to ",
       dstPtr + dstOffset, ", ", oldValue, " -> ", newValue);

#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_EXIT, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif
}

void CudaIpcConnection::flush(int64_t timeoutUsec) {
#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_ENTRY, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif

  if (timeoutUsec >= 0) {
    INFO(CONN, "CudaIpcConnection flush: timeout is not supported, ignored");
  }

  stream_->sync();

  INFO(CONN, "CudaIpcConnection flushing connection");

#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_EXIT, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif
}

// IBConnection

void IBConnection::recvThreadFunc() {
  // Set the CUDA device context for this thread
  if (localGpuDeviceId_ >= 0) {
    cudaError_t err = cudaSetDevice(localGpuDeviceId_);
    if (err != cudaSuccess) {
      WARN(NET, "IBConnection recvThreadFunc: cudaSetDevice(",
           localGpuDeviceId_, ") failed: ", cudaGetErrorString(err));
      return;
    }
    // Bind this thread to the NUMA node of the local GPU for optimal memory
    // access
    int deviceNumaNode = getDeviceNumaNode(localGpuDeviceId_);
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
    }
  }

  // Host-side buffer to receive newValue from imm_data (need 64-bit for
  // cudaMemcpy)
  uint64_t newValueHost = 0;

  while (!stopRecvThread_.load(std::memory_order_relaxed)) {
    auto qp = qp_.lock();
    if (!qp) break;

    int wcNum = qp->pollRecvCq();
    if (wcNum < 0) {
      WARN(NET, "IBConnection recvThreadFunc: pollRecvCq failed");
      break;
    }

    for (int i = 0; i < wcNum; ++i) {
      int status = qp->getRecvWcStatus(i);
      if (status != static_cast<int>(WsStatus::Success)) {
        WARN(NET, "IBConnection recvThreadFunc: recv work completion failed: ",
             qp->getRecvWcStatusString(i));
        // Post another recv to replace the failed one
        qp->stageRecv(/*wrId=*/0);
        qp->postRecv();
        continue;
      }

      // The imm_data contains newValue (32-bit, extended to 64-bit)
      // Note: getRecvWcImmData already converts from network byte order via
      // ntohl
      unsigned int immData = qp->getRecvWcImmData(i);
      newValueHost = static_cast<uint64_t>(immData);

      // Read dstGpuAddr from the local stored address (set by
      // setRemoteUpdateDstAddr)
      uint64_t dstGpuAddr = remoteUpdateDstAddr_;
      if (dstGpuAddr != 0) {
        uint64_t* dstPtr = reinterpret_cast<uint64_t*>(dstGpuAddr);

        // Use cudaMemcpyAsync with our dedicated stream to avoid blocking on
        // the default stream
        MSCCLPP_CUDATHROW(
            cudaMemcpyAsync(dstPtr, &newValueHost, sizeof(uint64_t),
                            cudaMemcpyHostToDevice, signalStream_));

        INFO(CONN, "IBConnection recvThreadFunc: updated GPU ptr ", dstPtr,
             " to ", newValueHost, " (immData=", immData, ")");
      }

      // Post another recv for future messages
      qp->stageRecv(/*wrId=*/0);
      qp->postRecv();
    }
  }
}

IBConnection::IBConnection(std::shared_ptr<Context> context,
                           Endpoint const& localEndpoint,
                           Endpoint const& remoteEndpoint)
    : BaseConnection(context, localEndpoint),
      transport_(localEndpoint.transport()),
      remoteTransport_(remoteEndpoint.transport()),
      dummyAtomicSource_(std::make_unique<uint64_t>(0)),
      ibNoAtomic_(getImpl(localEndpoint).ibNoAtomic_),
      stopRecvThread_(false),
      localGpuDeviceId_(localEndpoint.device().id),
      signalStream_(nullptr),
      remoteUpdateDstAddr_(0) {
  qp_ = getImpl(localEndpoint).ibQp_;
  qp_.lock()->rtr(getImpl(remoteEndpoint).ibQpInfo_);
  qp_.lock()->rts();
  dummyAtomicSourceMem_ = context->registerMemory(dummyAtomicSource_.get(),
                                                  sizeof(uint64_t), transport_);
  validateTransport(dummyAtomicSourceMem_, transport_);
  dstTransportInfo_ =
      getImpl(dummyAtomicSourceMem_).getTransportInfo(transport_);

  if (ibNoAtomic_) {
    // Create a CUDA stream for async memory copies
    MSCCLPP_CUDATHROW(
        cudaStreamCreateWithFlags(&signalStream_, cudaStreamNonBlocking));

    // Pre-post receive requests for incoming write-with-imm
    auto qp = qp_.lock();
    int maxRecvWr = localEndpoint.config().ib.maxRecvWr;
    for (int i = 0; i < maxRecvWr; ++i) {
      qp->stageRecv(/*wrId=*/0);
    }
    qp->postRecv();
    // Start the background thread to poll recv CQ
    recvThread_ = std::thread([this]() { this->recvThreadFunc(); });
    INFO(CONN, "IBConnection via ", getIBDeviceName(transport_),
         " created with no-atomic mode");
  } else {
    INFO(CONN, "IBConnection via ", getIBDeviceName(transport_),
         " created with atomic mode");
  }
}

IBConnection::~IBConnection() {
  if (ibNoAtomic_) {
    stopRecvThread_.store(true, std::memory_order_relaxed);
    if (recvThread_.joinable()) {
      recvThread_.join();
    }
    if (signalStream_ != nullptr) {
      // Synchronize stream to ensure all async copies are complete before
      // destruction. Do not throw: destructor is noexcept (-Wterminate).
      MSCCLPP_CUDA_WARN_IF_NOT_TEARDOWN(cudaStreamSynchronize(signalStream_));
      MSCCLPP_CUDA_WARN_IF_NOT_TEARDOWN(cudaStreamDestroy(signalStream_));
    }
  }
}

Transport IBConnection::transport() const { return transport_; }

Transport IBConnection::remoteTransport() const { return remoteTransport_; }

void IBConnection::setRemoteUpdateDstAddr(uint64_t addr) {
  remoteUpdateDstAddr_ = addr;
  INFO(CONN, "IBConnection setRemoteUpdateDstAddr: ", (void*)addr);
}

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset,
                         RegisteredMemory src, uint64_t srcOffset,
                         uint64_t size) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_WRITE_ENTRY, uint32_t(size), 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "dst is local, which is not supported");
  }
  auto srcTransportInfo = getImpl(src).getTransportInfo(transport());
  if (!srcTransportInfo.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "src is remote, which is not supported");
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  auto srcMr = srcTransportInfo.ibMr;

  qp_.lock()->stageSendWrite(srcMr, dstMrInfo, (uint32_t)size, /*wrId=*/0,
                             /*srcOffset=*/srcOffset,
                             /*dstOffset=*/dstOffset, /*signaled=*/true);

  qp_.lock()->postSend();
  INFO(CONN, "IBConnection write: from ",
       (uint8_t*)srcMr->getBuff() + srcOffset, " to ",
       (uint8_t*)dstMrInfo.addr + dstOffset, ", size ", size);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_WRITE_EXIT, uint32_t(size), 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif
}

void IBConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                                 uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_ENTRY, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport());
  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "dst is local, which is not supported");
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  // assert that src is on host
  uint64_t oldValue = *src;
  *src = newValue;

  if (ibNoAtomic_) {
    // Use RDMA write-with-imm instead of atomic operation
    // Send only newValue in imm_data (0-byte write)
    // The remote's recvThreadFunc will use its stored remoteUpdateDstAddr_ to
    // write

    // Put newValue in imm_data (truncated to 32-bit; semaphore counters should
    // fit)
    unsigned int immData = static_cast<unsigned int>(newValue);

    // Send 0-byte write-with-imm; use dstMrInfo as target (we don't actually
    // write anything)
    qp_.lock()->stageSendWriteWithImm(nullptr, dstMrInfo,
                                      /*size=*/0, /*wrId=*/0,
                                      /*srcOffset=*/0, /*dstOffset=*/0,
                                      /*signaled=*/true, /*immData=*/immData);
    qp_.lock()->postSend();
    INFO(CONN, "IBConnection write-with-imm: value ", oldValue, " -> ",
         newValue);
  } else {
    qp_.lock()->stageSendAtomicAdd(dstTransportInfo_.ibMr, dstMrInfo,
                                   /*wrId=*/0, dstOffset, newValue - oldValue,
                                   /*signaled=*/true);
    qp_.lock()->postSend();
    INFO(CONN, "IBConnection atomic Write: from ", src, " to ",
         (uint8_t*)dstMrInfo.addr + dstOffset, ", ", oldValue, " -> ",
         newValue);
  }

#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_EXIT, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif
}

void IBConnection::flush(int64_t timeoutUsec) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_FLUSH_ENTRY, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif

  Timer timer;
  while (qp_.lock()->getNumSendCqItems()) {
    int wcNum = qp_.lock()->pollSendCq();
    if (wcNum < 0) {
      THROW(NET, IbError, errno, "pollSendCq failed");
    } else if (timeoutUsec >= 0) {
      auto elapsed = timer.elapsed();
      if (elapsed > timeoutUsec) {
        THROW(CONN, Error, ErrorCode::Timeout,
              "pollSendCq timed out: waited for ", elapsed / 1e6,
              " seconds. Expected ", qp_.lock()->getNumSendCqItems(),
              " signals");
      }
    }
    for (int i = 0; i < wcNum; ++i) {
      int status = qp_.lock()->getSendWcStatus(i);
      if (status != static_cast<int>(WsStatus::Success)) {
        THROW(NET, Error, ErrorCode::SystemError,
              "an IB work item failed: ", qp_.lock()->getSendWcStatusString(i));
      }
    }
  }
  INFO(CONN, "IBConnection flushing connection");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_FLUSH_EXIT, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif
}

// EthernetConnection

EthernetConnection::EthernetConnection(std::shared_ptr<Context> context,
                                       Endpoint const& localEndpoint,
                                       Endpoint const& remoteEndpoint,
                                       uint64_t sendBufferSize,
                                       uint64_t recvBufferSize)
    : BaseConnection(context, localEndpoint),
      abortFlag_(0),
      sendBufferSize_(sendBufferSize),
      recvBufferSize_(recvBufferSize) {
  // Validating Transport Protocol
  if (localEndpoint.transport() != Transport::Ethernet ||
      remoteEndpoint.transport() != Transport::Ethernet) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "Ethernet connection can only be made from Ethernet endpoints");
  }

  // Instanciating Buffers
  sendBuffer_.resize(sendBufferSize_);
  recvBuffer_.resize(recvBufferSize_);

  // Creating Thread to Accept the Connection
  auto parameter = getImpl(localEndpoint).socket_.get();
  std::thread t([this, parameter]() {
    recvSocket_ = std::make_unique<Socket>(nullptr, MSCCLPP_SOCKET_MAGIC,
                                           SocketTypeUnknown, abortFlag_);
    recvSocket_->accept(parameter);
  });

  // Starting Connection
  sendSocket_ = std::make_unique<Socket>(
      &(getImpl(remoteEndpoint).socketAddress_), MSCCLPP_SOCKET_MAGIC,
      SocketTypeBootstrap, abortFlag_);
  sendSocket_->connect();

  // Ensure the Connection was Established
  t.join();

  // Starting Thread to Receive Messages
  int deviceId = -1;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  threadRecvMessages_ = std::thread([deviceId, this]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(deviceId));
    this->recvMessages();
  });

  INFO(CONN, "Ethernet connection created");
}

EthernetConnection::~EthernetConnection() {
  sendSocket_->close();
  recvSocket_->close();
  threadRecvMessages_.join();
}

Transport EthernetConnection::transport() const { return Transport::Ethernet; }

Transport EthernetConnection::remoteTransport() const {
  return Transport::Ethernet;
}

void EthernetConnection::write(RegisteredMemory dst, uint64_t dstOffset,
                               RegisteredMemory src, uint64_t srcOffset,
                               uint64_t size) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_WRITE_ENTRY, uint32_t(size), 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif

  // Validating Transport Protocol
  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  // Initializing Variables
  char* srcPtr = reinterpret_cast<char*>(src.data()) + srcOffset / sizeof(char);
  char* dstPtr =
      reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset / sizeof(char);
  uint64_t sentDataSize = 0;
  uint64_t headerSize = 0;

  // Copying Meta Data to Send Buffer
  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr),
            sendBuffer_.data() + headerSize / sizeof(char));
  headerSize += sizeof(dstPtr);
  char* sizeBytes = reinterpret_cast<char*>(&size);
  std::copy(sizeBytes, sizeBytes + sizeof(size),
            sendBuffer_.data() + headerSize / sizeof(char));
  headerSize += sizeof(size);

  // Getting Data From GPU and Sending Message
  while (sentDataSize < size) {
    uint64_t dataSize = std::min(sendBufferSize_ - headerSize / sizeof(char),
                                 (size - sentDataSize) / sizeof(char)) *
                        sizeof(char);
    uint64_t messageSize = dataSize + headerSize;
    mscclpp::gpuMemcpy(sendBuffer_.data() + headerSize / sizeof(char),
                       srcPtr + (sentDataSize / sizeof(char)), dataSize,
                       cudaMemcpyDeviceToHost);
    sendSocket_->send(sendBuffer_.data(), messageSize);
    sentDataSize += messageSize;
    headerSize = 0;
  }

  INFO(CONN, "EthernetConnection write: from ", srcPtr, " to ", dstPtr,
       ", size ", size);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_WRITE_EXIT, uint32_t(size), 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                                       uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_ENTRY, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif

  // Validating Transport Protocol
  validateTransport(dst, remoteTransport());

  // Initializing Variables
  uint64_t oldValue = *src;
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(
      reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset);
  uint64_t dataSize = sizeof(uint64_t);
  uint64_t messageSize = 0;
  *src = newValue;

  // Copying Data to Send Buffer
  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr),
            sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += sizeof(dstPtr);
  char* sizeBytes = reinterpret_cast<char*>(&dataSize);
  std::copy(sizeBytes, sizeBytes + sizeof(dataSize),
            sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += sizeof(dataSize);
  char* dataBytes = reinterpret_cast<char*>(src);
  std::copy(dataBytes, dataBytes + dataSize,
            sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += dataSize;

  // Sending Message
  sendSocket_->send(sendBuffer_.data(), messageSize);

  INFO(CONN, "EthernetConnection atomic write: from ", src, " to ",
       dstPtr + dstOffset, ", ", oldValue, " -> ", newValue);

#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_EXIT, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::flush(int64_t) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_FLUSH_ENTRY, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif

  INFO(CONN, "EthernetConnection flushing connection");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_FLUSH_EXIT, 0, 0,
                         *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::recvMessages() {
  // Declarating Variables
  char* ptr;
  uint64_t size;
  uint64_t recvSize;
  int closed = 0;
  bool received = true;

  // Receiving Messages Until Connection is Closed
  while (recvSocket_->getState() != SocketStateClosed) {
#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_META_ENTRY)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_META_ENTRY, 0, 0,
                           *NpKit::GetCpuTimestamp(), 1);
#endif

    // Receiving Data Address
    if (closed == 0) recvSocket_->recvUntilEnd(&ptr, sizeof(char*), &closed);
    received &= !closed;

    // Receiving data size
    if (closed == 0)
      recvSocket_->recvUntilEnd(&size, sizeof(uint64_t), &closed);
    received &= !closed;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_META_EXIT)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_META_EXIT, uint32_t(size),
                           0, *NpKit::GetCpuTimestamp(), 1);
#endif

#if defined(ENABLE_NPKIT) && \
    defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_DATA_ENTRY)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_DATA_ENTRY, uint32_t(size),
                           0, *NpKit::GetCpuTimestamp(), 1);
#endif

    // Receiving Data and Copying Data yo GPU
    recvSize = 0;
    while (recvSize < size && closed == 0) {
      uint64_t messageSize =
          std::min(recvBufferSize_, (size - recvSize) / sizeof(char)) *
          sizeof(char);
      recvSocket_->recvUntilEnd(recvBuffer_.data(), messageSize, &closed);
      received &= !closed;

      if (received)
        mscclpp::gpuMemcpy(ptr + (recvSize / sizeof(char)), recvBuffer_.data(),
                           messageSize, cudaMemcpyHostToDevice);
      recvSize += messageSize;
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_DATA_EXIT)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_DATA_EXIT, uint32_t(size),
                           0, *NpKit::GetCpuTimestamp(), 1);
#endif
  }
}

}  // namespace mscclpp
