// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ibverbs_wrapper.hpp"
#include "env.hpp"
#include "logger.hpp"
#include <memory>
#include <dlfcn.h>

namespace mscclpp {

static std::unique_ptr<void, int (*)(void*)> globalIBVerbsHandle(nullptr,
                                                                 &::dlclose);

void* IBVerbs::dlsym(std::string const& symbol, bool allowReturnNull) {
  if (!globalIBVerbsHandle) {
    if (mscclpp::env()->ibvSo != "") {
      void* handle = ::dlopen(mscclpp::env()->ibvSo.c_str(), RTLD_NOW);
      if (handle) {
        globalIBVerbsHandle.reset(handle);
      } else {
        THROW(
            NET, SysError, errno,
            "Failed to load libibverbs library specified by MSCCLPP_IBV_SO ('",
            mscclpp::env()->ibvSo, "'): ", std::string(::dlerror()));
      }
    } else {
      char const* possibleLibNames[] = {"libibverbs.so", "libibverbs.so.1",
                                        nullptr};
      for (int i = 0; possibleLibNames[i] != nullptr; i++) {
        void* handle = ::dlopen(possibleLibNames[i], RTLD_NOW);
        if (handle) {
          globalIBVerbsHandle.reset(handle);
          break;
        }
      }
    }
    if (!globalIBVerbsHandle) {
      THROW(NET, SysError, errno,
            "Failed to open libibverbs: ", std::string(::dlerror()));
    }
  }
  void* ptr = ::dlsym(globalIBVerbsHandle.get(), symbol.c_str());
  if (!ptr && !allowReturnNull) {
    THROW(NET, SysError, errno, "Failed to load libibverbs symbol: ", symbol);
  }
  return ptr;
}

bool IBVerbs::isDmabufSupported() {
  static int isSupported = -1;
  if (isSupported == -1) {
    void* ptr = IBVerbs::dlsym("ibv_reg_dmabuf_mr", true);
    isSupported = (ptr != nullptr);
    if (!isSupported) {
      DEBUG(NET, "This platform does not support DMABUF");
    }
  }
  return isSupported;
}

struct ibv_mr* IBVerbs::ibv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset,
                                          size_t length, uint64_t iova, int fd,
                                          int access) {
  using FuncType =
      struct ibv_mr* (*)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int);
  static FuncType impl = nullptr;
  if (!isDmabufSupported())
    THROW(NET, Error, ErrorCode::InvalidUsage,
          "libibverbs does not support ibv_reg_dmabuf_mr in this platform.");
  if (!impl)
    impl = reinterpret_cast<FuncType>(IBVerbs::dlsym("ibv_reg_dmabuf_mr"));
  return impl(pd, offset, length, iova, fd, access);
}

}  // namespace mscclpp
