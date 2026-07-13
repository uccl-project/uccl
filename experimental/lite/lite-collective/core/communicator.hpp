// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_COMMUNICATOR_HPP_
#define MSCCLPP_COMMUNICATOR_HPP_

#include "core.hpp"
#include "utils_internal.hpp"
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mscclpp {

class BaseRecvItem {
 public:
  virtual ~BaseRecvItem() = default;
  virtual void wait() = 0;
  virtual bool isReady() const = 0;
};

template <typename T>
class RecvItem : public BaseRecvItem {
 public:
  RecvItem(std::shared_future<T> future) : future_(future) {}

  void wait() { future_.wait(); }

  bool isReady() const {
    return future_.wait_for(std::chrono::seconds(0)) ==
           std::future_status::ready;
  }

 private:
  std::shared_future<T> future_;
};

class LocalRecvMemory {
 public:
  LocalRecvMemory() : future_(promise_.get_future()) {}

  void set(RegisteredMemory memory) { promise_.set_value(std::move(memory)); }

  std::shared_future<RegisteredMemory> reference() const { return future_; }

  bool isReady() const {
    return future_.valid() && (future_.wait_for(std::chrono::seconds(0)) ==
                               std::future_status::ready);
  }

 private:
  std::promise<RegisteredMemory> promise_;
  std::shared_future<RegisteredMemory> future_;
};

struct ConnectionInfo {
  int remoteRank;
  int tag;
};

struct Communicator::Impl {
  std::shared_ptr<Bootstrap> bootstrap_;
  std::shared_ptr<Context> context_;
  std::unordered_map<BaseConnection const*, ConnectionInfo> connectionInfos_;

  // Temporary storage for the latest RecvItem of each {remoteRank, tag} pair.
  // If the RecvItem gets ready, it will be removed at the next call to
  // getLastRecvItem.
  std::unordered_map<std::pair<int, int>, std::shared_ptr<BaseRecvItem>,
                     PairHash>
      lastRecvItems_;

  // RegisteredMemory items sent to the local rank of each tag. Sending memory
  // to the local rank is unnecessary, but we enable this for consistency of the
  // Communicator interface.
  std::unordered_map<int, std::list<LocalRecvMemory>> localRecvMemories_;

  Impl(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context);

  // Set the last RecvItem for a {remoteRank, tag} pair.
  // This is used to store the corresponding RecvItem of a future returned by
  // recvMemory() or connect().
  void setLastRecvItem(int remoteRank, int tag,
                       std::shared_ptr<BaseRecvItem> item);

  // Return the last RecvItem that is not ready.
  // If the item is ready, it will be removed from the map and nullptr will be
  // returned.
  std::shared_ptr<BaseRecvItem> getLastRecvItem(int remoteRank, int tag);

  struct Connector;
};

}  // namespace mscclpp

#endif  // MSCCLPP_COMMUNICATOR_HPP_
