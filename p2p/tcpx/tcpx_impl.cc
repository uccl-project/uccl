/**
 * @file tcpx_impl.cc
 * @brief TCPX plugin wrapper implementation
 *
 * Provides C API wrapper around NCCL GPUDirect TCPX plugin.
 * - Dynamic plugin loading via dlopen
 * - Thin wrappers with error checking and debug logging
 * - Debug logging controlled by UCCL_TCPX_DEBUG environment variable
 */

#include "tcpx_interface.h"
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

//=============================================================================
// Debug Logging
//=============================================================================

static bool tcpx_debug_enabled() {
  char const* v = std::getenv("UCCL_TCPX_DEBUG");
  return v && *v && *v != '0';
}

static void tcpx_dbg(char const* fmt, ...) {
  if (!tcpx_debug_enabled()) return;
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[TCPX] ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

// Logger callback for NCCL plugin init
static int nccl_debug_logger(int level, char const* abbrev, char const* file,
                             int line, char const* fmt, ...) {
  if (!tcpx_debug_enabled()) return 0;
  std::fprintf(stderr, "[ncclNet:%d] %s:%d ", level, file ? file : "?", line);
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fprintf(stderr, "\n");
  return 0;
}

//=============================================================================
// NCCL Plugin Function Table (v7 API)
//=============================================================================

typedef int (*ncclLogFn)(int, char const*, char const*, int, char const*, ...);

struct ncclNet_v7 {
  char const* name;
  int (*init)(ncclLogFn);
  int (*devices)(int*);
  int (*getProperties)(int, void*);
  int (*listen)(int, void*, void**);
  int (*connect)(int, void*, void**, void**);
  int (*accept)(void*, void**, void**);
  int (*regMr)(void*, void*, int, int, void**);
  int (*regMrDmaBuf)(void*, void*, size_t, int, uint64_t, int, void**);
  int (*deregMr)(void*, void*);
  int (*isend)(void*, void*, int, int, void*, void**);
  int (*irecv)(void*, int, void**, int*, int*, void**, void**);
  int (*iflush)(void*, int, void**, int*, void**, void**);
  int (*test)(void*, int*, int*);
  int (*closeSend)(void*);
  int (*closeRecv)(void*);
  int (*closeListen)(void*);
  int (*getDeviceMr)(void*, void*, void**);
  int (*irecvConsumed)(void*, int, void*);
};

//=============================================================================
// Global State
//=============================================================================

static void* g_plugin_handle = nullptr;  // Plugin shared library handle
static ncclNet_v7* g_net = nullptr;      // Plugin function table
static int g_inited = 0;                 // Initialization flag

//=============================================================================
// Helper Functions
//=============================================================================

static void* resolve_symbol(void* handle, char const* sym) {
  dlerror();
  void* p = dlsym(handle, sym);
  char const* e = dlerror();
  if (e) {
    tcpx_dbg("dlsym(%s) failed: %s", sym, e);
    return nullptr;
  }
  return p;
}

//=============================================================================
// Plugin Initialization
//=============================================================================

int tcpx_load_plugin(char const* plugin_path) {
  tcpx_dbg("Loading plugin: %s", plugin_path);
  g_plugin_handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
  if (!g_plugin_handle) {
    std::fprintf(stderr, "[TCPX] dlopen failed: %s\n", dlerror());
    return -1;
  }

  // Try v8 first, then v7
  void* sym = resolve_symbol(g_plugin_handle, "ncclNetPlugin_v8");
  if (!sym) sym = resolve_symbol(g_plugin_handle, "ncclNetPlugin_v7");
  if (!sym) {
    std::fprintf(stderr, "[TCPX] ncclNetPlugin_v7/v8 not found\n");
    dlclose(g_plugin_handle);
    g_plugin_handle = nullptr;
    return -1;
  }

  g_net = reinterpret_cast<ncclNet_v7*>(sym);
  if (!g_net || !g_net->init || !g_net->devices) {
    std::fprintf(stderr, "[TCPX] invalid plugin function table\n");
    dlclose(g_plugin_handle);
    g_plugin_handle = nullptr;
    g_net = nullptr;
    return -1;
  }

  tcpx_dbg("Calling net->init with safe logger");
  int rc = g_net->init(&nccl_debug_logger);
  tcpx_dbg("net->init rc=%d", rc);
  g_inited = 1;
  return 0;
}

int tcpx_get_device_count() {
  if (!g_inited) {
    char const* path = std::getenv("UCCL_TCPX_PLUGIN_PATH");
    if (!path) path = "/usr/local/tcpx/lib64/libnccl-net-tcpx.so";
    if (tcpx_load_plugin(path) != 0) {
      tcpx_dbg("Failed to load plugin");
      return -1;
    }
  }
  int ndev = 0;
  int rc = g_net->devices(&ndev);
  tcpx_dbg("net->devices rc=%d ndev=%d", rc, ndev);
  return rc == 0 ? ndev : -1;
}

// Connection management implementation - use v5 API for consistency
int tcpx_listen(int dev, void* handle, void** listen_comm) {
  tcpx_dbg("tcpx_listen: dev=%d (using v5 API for consistency)", dev);

  if (!g_net || !g_net->listen) {
    tcpx_dbg("tcpx_listen: plugin not initialized or listen not available");
    return -1;
  }

  int rc = g_net->listen(dev, handle, listen_comm);
  tcpx_dbg("tcpx_listen (v5): rc=%d listen_comm=%p", rc, *listen_comm);
  return rc;
}

// Note: The NCCL plugin connect API signature differs from our original
// assumption The actual TCPX plugin uses different function names and
// parameters
int tcpx_connect_v5(int dev, void* handle, void** send_comm,
                    void** send_dev_handle) {
  tcpx_dbg("tcpx_connect_v5: dev=%d handle=%p (using v5 API)", dev, handle);

  if (!g_net || !g_net->connect) {
    tcpx_dbg(
        "tcpx_connect_v5: plugin not initialized or connect not available");
    return -1;
  }

  int rc = g_net->connect(dev, handle, send_comm, send_dev_handle);
  tcpx_dbg("tcpx_connect_v5: rc=%d send_comm=%p send_dev_handle=%p", rc,
           *send_comm, *send_dev_handle);
  return rc;
}

int tcpx_accept_v5(void* listen_comm, void** recv_comm,
                   void** recv_dev_handle) {
  tcpx_dbg("tcpx_accept_v5: listen_comm=%p (using v5 API)", listen_comm);

  if (!g_net || !g_net->accept) {
    tcpx_dbg("tcpx_accept_v5: plugin not initialized or accept not available");
    return -1;
  }

  int rc = g_net->accept(listen_comm, recv_comm, recv_dev_handle);
  tcpx_dbg("tcpx_accept_v5: rc=%d recv_comm=%p recv_dev_handle=%p", rc,
           *recv_comm, *recv_dev_handle);
  return rc;
}

// Memory registration implementation
int tcpx_reg_mr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (!g_net || !g_net->regMr) {
    tcpx_dbg("tcpx_reg_mr: plugin not initialized or regMr not available");
    return -1;
  }
  tcpx_dbg("tcpx_reg_mr: comm=%p data=%p size=%zu type=%d", comm, data, size,
           type);
  int rc = g_net->regMr(comm, data, size, type, mhandle);
  tcpx_dbg("tcpx_reg_mr: rc=%d mhandle=%p", rc, *mhandle);
  return rc;
}

int tcpx_dereg_mr(void* comm, void* mhandle) {
  if (!g_net || !g_net->deregMr) {
    tcpx_dbg("tcpx_dereg_mr: plugin not initialized or deregMr not available");
    return -1;
  }
  tcpx_dbg("tcpx_dereg_mr: comm=%p mhandle=%p", comm, mhandle);
  int rc = g_net->deregMr(comm, mhandle);
  tcpx_dbg("tcpx_dereg_mr: rc=%d", rc);
  return rc;
}

// Data transfer implementation
int tcpx_isend(void* send_comm, void* data, int size, int tag, void* mhandle,
               void** request) {
  if (!g_net || !g_net->isend) {
    tcpx_dbg("tcpx_isend: plugin not initialized or isend not available");
    return -1;
  }

  // Check for null communication handle to avoid segfault
  if (!send_comm) {
    tcpx_dbg("tcpx_isend: send_comm is null, returning error");
    return -1;
  }

  tcpx_dbg("tcpx_isend: send_comm=%p data=%p size=%d tag=%d mhandle=%p",
           send_comm, data, size, tag, mhandle);
  int rc = g_net->isend(send_comm, data, size, tag, mhandle, request);
  tcpx_dbg("tcpx_isend: rc=%d request=%p", rc, request ? *request : nullptr);
  return rc;
}

int tcpx_irecv(void* recv_comm, int n, void** data, int* sizes, int* tags,
               void** mhandles, void** request) {
  if (!g_net || !g_net->irecv) {
    tcpx_dbg("tcpx_irecv: plugin not initialized or irecv not available");
    return -1;
  }

  // Check for null communication handle to avoid segfault
  if (!recv_comm) {
    tcpx_dbg("tcpx_irecv: recv_comm is null, returning error");
    return -1;
  }

  tcpx_dbg("tcpx_irecv: recv_comm=%p n=%d", recv_comm, n);
  int rc = g_net->irecv(recv_comm, n, data, sizes, tags, mhandles, request);
  tcpx_dbg("tcpx_irecv: rc=%d request=%p", rc, request ? *request : nullptr);
  return rc;
}

int tcpx_test(void* request, int* done, int* size) {
  if (!g_net || !g_net->test) {
    tcpx_dbg("tcpx_test: plugin not initialized or test not available");
    return -1;
  }

  // Check for null request handle to avoid segfault
  if (!request) {
    tcpx_dbg("tcpx_test: request is null, returning error");
    return -1;
  }

  int rc = g_net->test(request, done, size);
  if (done && *done) {
    tcpx_dbg("tcpx_test: rc=%d done=%d size=%d", rc, *done, size ? *size : -1);
  }
  return rc;
}

// Completion helpers implementation

int tcpx_irecv_consumed(void* comm, int n, void* request) {
  if (!g_net || !g_net->irecvConsumed) {
    tcpx_dbg(
        "tcpx_irecv_consumed: plugin not initialized or irecvConsumed not "
        "available");

    return -1;
  }

  if (!request) {
    tcpx_dbg("tcpx_irecv_consumed: request is null, returning error");

    return -1;
  }

  tcpx_dbg("tcpx_irecv_consumed: comm=%p n=%d request=%p", comm, n, request);

  int rc = g_net->irecvConsumed(comm, n, request);

  tcpx_dbg("tcpx_irecv_consumed: rc=%d", rc);

  return rc;
}

// Connection cleanup implementation
int tcpx_close_send(void* send_comm) {
  if (!g_net || !g_net->closeSend) {
    tcpx_dbg(
        "tcpx_close_send: plugin not initialized or closeSend not available");
    return -1;
  }
  tcpx_dbg("tcpx_close_send: send_comm=%p", send_comm);
  int rc = g_net->closeSend(send_comm);
  tcpx_dbg("tcpx_close_send: rc=%d", rc);
  return rc;
}

int tcpx_close_recv(void* recv_comm) {
  if (!g_net || !g_net->closeRecv) {
    tcpx_dbg(
        "tcpx_close_recv: plugin not initialized or closeRecv not available");
    return -1;
  }
  tcpx_dbg("tcpx_close_recv: recv_comm=%p", recv_comm);
  int rc = g_net->closeRecv(recv_comm);
  tcpx_dbg("tcpx_close_recv: rc=%d", rc);
  return rc;
}

int tcpx_close_listen(void* listen_comm) {
  if (!g_net || !g_net->closeListen) {
    tcpx_dbg(
        "tcpx_close_listen: plugin not initialized or closeListen not "
        "available");
    return -1;
  }
  tcpx_dbg("tcpx_close_listen: listen_comm=%p", listen_comm);
  int rc = g_net->closeListen(listen_comm);
  tcpx_dbg("tcpx_close_listen: rc=%d", rc);
  return rc;
}
