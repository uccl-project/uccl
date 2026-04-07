// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "proxy.hpp"
#include "api.h"
#include "core.hpp"
#include "debug.h"
#include "gpu_utils.hpp"
#include "numa.hpp"
#include "utils.hpp"
#include <atomic>
#include <thread>

namespace mscclpp {

constexpr int ProxyStopCheckPeriod = 1000;
constexpr int ProxyStartWarnPeriod = 1000;

struct Proxy::Impl {
  ProxyHandler handler;
  std::function<void()> threadInit;
  std::shared_ptr<Fifo> fifo;
  std::atomic_bool threadStarted;
  std::atomic_bool threadFailed;
  std::atomic_bool threadExited;
  std::thread service;
  std::atomic_bool running;

  Impl(ProxyHandler handler, std::function<void()> threadInit, int fifoSize)
      : handler(handler),
        threadInit(threadInit),
        fifo(std::make_shared<Fifo>(fifoSize)),
        threadStarted(false),
        threadFailed(false),
        threadExited(false),
        running(false) {}
};

MSCCLPP_API_CPP Proxy::Proxy(ProxyHandler handler,
                             std::function<void()> threadInit, int fifoSize) {
  pimpl_ = std::make_unique<Impl>(handler, threadInit, fifoSize);
}

MSCCLPP_API_CPP Proxy::Proxy(ProxyHandler handler, int fifoSize) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  int deviceNumaNode = getDeviceNumaNode(cudaDevice);
  auto initFunc = [cudaDevice, deviceNumaNode]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
    }
  };
  pimpl_ = std::make_unique<Impl>(handler, initFunc, fifoSize);
}

MSCCLPP_API_CPP Proxy::~Proxy() {
  if (pimpl_) {
    stop();
  }
}

MSCCLPP_API_CPP void Proxy::start(bool blocking) {
  pimpl_->running.store(true, std::memory_order_release);
  pimpl_->threadStarted.store(false, std::memory_order_release);
  pimpl_->threadFailed.store(false, std::memory_order_release);
  pimpl_->threadExited.store(false, std::memory_order_release);
  pimpl_->service = std::thread([this] {
    try {
      // threadInit() is responsible for setting up the runtime context for the
      // thread. The default implementation sets the CUDA device and NUMA
      // affinity to match the main thread (see Proxy ctor). It should be
      // called before any CUDA API calls to avoid resource allocation on
      // unwanted GPUs.
      pimpl_->threadInit();

      // never capture in a proxy thread
      auto mode = cudaStreamCaptureModeRelaxed;
      MSCCLPP_CUDATHROW(cudaThreadExchangeStreamCaptureMode(&mode));

      pimpl_->threadStarted.store(true, std::memory_order_release);

      ProxyHandler handler = this->pimpl_->handler;
      auto fifo = this->pimpl_->fifo;
      ProxyTrigger trigger;

      int runCnt = ProxyStopCheckPeriod;
      for (;;) {
        if (runCnt-- == 0) {
          runCnt = ProxyStopCheckPeriod;
          if (!this->pimpl_->running.load(std::memory_order_acquire)) {
            break;
          }
        }
        // Poll to see if we are ready to send anything
        trigger = fifo->poll();
        if (trigger.fst == 0 ||
            trigger.snd == 0) {  // TODO: this check is a potential pitfall for
                                 // custom triggers
          continue;              // there is one in progress
        }
        trigger.snd ^=
            (uint64_t{1} << uint64_t{
                 63});  // this is where the last bit of snd is reverted.

        ProxyHandlerResult result = handler(trigger);

        // Send completion: reset only the high 64 bits
        fifo->pop();

        if (result == ProxyHandlerResult::Stop) {
          break;
        }
      }
    } catch (std::exception const& ex) {
      WARN("Proxy thread failed with exception: %s", ex.what());
      pimpl_->threadFailed.store(true, std::memory_order_release);
      pimpl_->running.store(false, std::memory_order_release);
      pimpl_->threadStarted.store(true, std::memory_order_release);
    } catch (...) {
      WARN("Proxy thread failed with unknown exception");
      pimpl_->threadFailed.store(true, std::memory_order_release);
      pimpl_->running.store(false, std::memory_order_release);
      pimpl_->threadStarted.store(true, std::memory_order_release);
    }
    pimpl_->threadExited.store(true, std::memory_order_release);
  });

  if (blocking) {
    int count = ProxyStartWarnPeriod;
    while (!pimpl_->threadStarted.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      count--;
      if (count == 0) {
        count = ProxyStartWarnPeriod;
        WARN("Proxy thread startup taking longer than expected.");
      }
    }
    if (pimpl_->threadFailed.load(std::memory_order_acquire)) {
      if (pimpl_->service.joinable()) {
        pimpl_->service.join();
      }
      throw Error("Proxy thread failed to start", ErrorCode::InternalError);
    }
  }
}

MSCCLPP_API_CPP void Proxy::stop() {
  pimpl_->running.store(false, std::memory_order_release);
  if (pimpl_->service.joinable()) {
    pimpl_->service.join();
  }
  pimpl_->threadStarted.store(false, std::memory_order_release);
}

MSCCLPP_API_CPP std::shared_ptr<Fifo> Proxy::fifo() { return pimpl_->fifo; }

}  // namespace mscclpp
