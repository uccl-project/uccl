// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_IB_HPP_
#define MSCCLPP_IB_HPP_

#include "core.hpp"
#include <list>
#include <memory>
#include <string>

// Forward declarations of IB structures
struct ibv_context;
struct ibv_pd;
struct ibv_mr;
struct ibv_qp;
struct ibv_cq;
struct ibv_wc;
struct ibv_send_wr;
struct ibv_recv_wr;
struct ibv_sge;

namespace mscclpp {

struct IbMrInfo {
  uint64_t addr;
  uint32_t rkey;
};

class IbMr {
 public:
  ~IbMr();

  IbMrInfo getInfo() const;
  void const* getBuff() const;
  uint32_t getLkey() const;

 private:
  IbMr(ibv_pd* pd, void* buff, std::size_t size);

  ibv_mr* mr_;
  void* buff_;
  std::size_t size_;

  friend class IbCtx;
};

// QP info to be shared with the remote peer
struct IbQpInfo {
  uint16_t lid;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  int mtu;
  uint64_t iid;
  bool isGrh;
};

enum class WsStatus {
  Success,
};

class IbQp {
 public:
  ~IbQp();

  void rtr(IbQpInfo const& info);
  void rts();
  void stageSendWrite(IbMr const* mr, IbMrInfo const& info, uint32_t size,
                      uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                      bool signaled);
  void stageSendAtomicAdd(IbMr const* mr, IbMrInfo const& info, uint64_t wrId,
                          uint64_t dstOffset, uint64_t addVal, bool signaled);
  void stageSendWriteWithImm(IbMr const* mr, IbMrInfo const& info,
                             uint32_t size, uint64_t wrId, uint64_t srcOffset,
                             uint64_t dstOffset, bool signaled,
                             unsigned int immData);
  void postSend();

  void stageRecv(uint64_t wrId);
  void stageRecv(IbMr const* mr, uint64_t wrId, uint32_t size,
                 uint64_t offset = 0);
  void postRecv();

  int pollSendCq();
  int pollRecvCq();

  IbQpInfo& getInfo() { return info_; }
  int getSendWcStatus(int idx) const;
  std::string getSendWcStatusString(int idx) const;
  int getNumSendCqItems() const;
  int getRecvWcStatus(int idx) const;
  std::string getRecvWcStatusString(int idx) const;
  unsigned int getRecvWcImmData(int idx) const;

 private:
  struct SendWrInfo {
    ibv_send_wr* wr;
    ibv_sge* sge;
  };

  struct RecvWrInfo {
    ibv_recv_wr* wr;
    ibv_sge* sge;
  };

  IbQp(ibv_context* ctx, ibv_pd* pd, int portNum, int gidIndex,
       int maxSendCqSize, int maxSendCqPollNum, int maxSendWr, int maxRecvWr,
       int maxWrPerSend);
  SendWrInfo getNewSendWrInfo();
  RecvWrInfo getNewRecvWrInfo();

  int portNum_;
  int gidIndex_;

  IbQpInfo info_;

  ibv_qp* qp_;
  ibv_cq* sendCq_;
  ibv_cq* recvCq_;
  std::shared_ptr<std::vector<ibv_wc>> sendWcs_;
  std::shared_ptr<std::vector<ibv_wc>> recvWcs_;
  std::shared_ptr<std::vector<ibv_send_wr>> sendWrs_;
  std::shared_ptr<std::vector<ibv_sge>> sendSges_;
  std::shared_ptr<std::vector<ibv_recv_wr>> recvWrs_;
  std::shared_ptr<std::vector<ibv_sge>> recvSges_;
  int numStagedSend_;
  int numStagedRecv_;
  int numPostedSignaledSend_;
  int numStagedSignaledSend_;

  int const maxSendCqPollNum_;
  int const maxSendWr_;
  int const maxWrPerSend_;
  int const maxRecvWr_;

  friend class IbCtx;
};

class IbCtx {
 public:
#if defined(USE_IBVERBS)
  IbCtx(std::string const& devName);
  ~IbCtx();

  std::shared_ptr<IbQp> createQp(int port, int gidIndex, int maxSendCqSize,
                                 int maxSendCqPollNum, int maxSendWr,
                                 int maxRecvWr, int maxWrPerSend);
  std::unique_ptr<const IbMr> registerMr(void* buff, std::size_t size);
  bool supportsRdmaAtomics() const;
#else
  IbCtx([[maybe_unused]] std::string const& devName) {}
  ~IbCtx() {}

  std::shared_ptr<IbQp> createQp(int, int, int, int, int, int, int) {
    return nullptr;
  }
  std::unique_ptr<const IbMr> registerMr([[maybe_unused]] void* buff,
                                         [[maybe_unused]] std::size_t size) {
    return nullptr;
  }
  bool supportsRdmaAtomics() const { return false; }
#endif

  std::string const& getDevName() const { return devName_; };

 private:
  bool isPortUsable(int port, int gidIndex) const;
  int getAnyUsablePort(int gidIndex) const;

  const std::string devName_;
  ibv_context* ctx_;
  ibv_pd* pd_;
  bool supportsRdmaAtomics_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_IB_HPP_
