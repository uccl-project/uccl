#include "tcpx_interface.h"
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

// Debug helper controlled by UCCL_TCPX_DEBUG
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

// Safe logger for NCCL net plugin init
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

// Minimal NCCL net v7-like table (only members we use)
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

// globals
static void* g_plugin_handle = nullptr;
static ncclNet_v7* g_net = nullptr;
static int g_inited = 0;

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

// Connection management implementation
int tcpx_listen(int dev, void* handle, void** listen_comm) {
  if (!g_net || !g_net->listen) {
    tcpx_dbg("tcpx_listen: plugin not initialized or listen not available");
    return -1;
  }
  tcpx_dbg("tcpx_listen: dev=%d", dev);
  int rc = g_net->listen(dev, handle, listen_comm);
  tcpx_dbg("tcpx_listen: rc=%d listen_comm=%p", rc, *listen_comm);
  return rc;
}

// Note: The NCCL plugin connect API signature differs from our original
// assumption The actual TCPX plugin uses different function names and
// parameters
int tcpx_connect_v5(int dev, void* handle, void** send_comm,
                    void** send_dev_handle) {
  tcpx_dbg("tcpx_connect_v5: dev=%d handle=%p", dev, handle);

  // We need to call the real TCPX plugin function here
  // However, our g_net structure may not expose the v5-specific functions
  // Let's first try to get the function pointer via dlsym

  if (!g_plugin_handle) {
    tcpx_dbg("tcpx_connect_v5: plugin not loaded");
    return -1;
  }

  // Attempt to get the tcpxConnect_v5 function using the C++ mangled name
  typedef int (*tcpxConnect_v5_fn)(int, void*, void**, void**);
  tcpxConnect_v5_fn connect_fn = (tcpxConnect_v5_fn)dlsym(
      g_plugin_handle, "_Z14tcpxConnect_v5iPvPS_PP24ncclNetDeviceHandle_v7_t");

  if (!connect_fn) {
    tcpx_dbg("tcpxConnect_v5 function not found: %s", dlerror());
    return -1;
  }

  int rc = connect_fn(dev, handle, send_comm, send_dev_handle);
  tcpx_dbg("tcpx_connect_v5: rc=%d send_comm=%p send_dev_handle=%p", rc,
           *send_comm, *send_dev_handle);
  return rc;
}

int tcpx_accept_v5(void* listen_comm, void** recv_comm,
                   void** recv_dev_handle) {
  tcpx_dbg("tcpx_accept_v5: listen_comm=%p", listen_comm);

  if (!g_plugin_handle) {
    tcpx_dbg("tcpx_accept_v5: plugin not loaded");
    return -1;
  }

  // Attempt to get the tcpxAccept_v5 function using the C++ mangled name
  typedef int (*tcpxAccept_v5_fn)(void*, void**, void**);
  tcpxAccept_v5_fn accept_fn = (tcpxAccept_v5_fn)dlsym(
      g_plugin_handle, "_Z13tcpxAccept_v5PvPS_PP24ncclNetDeviceHandle_v7_t");

  if (!accept_fn) {
    tcpx_dbg("tcpxAccept_v5 function not found: %s", dlerror());
    return -1;
  }

  int rc = accept_fn(listen_comm, recv_comm, recv_dev_handle);
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
  tcpx_dbg("tcpx_isend: send_comm=%p data=%p size=%d tag=%d mhandle=%p",
           send_comm, data, size, tag, mhandle);
  int rc = g_net->isend(send_comm, data, size, tag, mhandle, request);
  tcpx_dbg("tcpx_isend: rc=%d request=%p", rc, *request);
  return rc;
}

int tcpx_irecv(void* recv_comm, int n, void** data, int* sizes, int* tags,
               void** mhandles, void** request) {
  if (!g_net || !g_net->irecv) {
    tcpx_dbg("tcpx_irecv: plugin not initialized or irecv not available");
    return -1;
  }
  tcpx_dbg("tcpx_irecv: recv_comm=%p n=%d", recv_comm, n);
  int rc = g_net->irecv(recv_comm, n, data, sizes, tags, mhandles, request);
  tcpx_dbg("tcpx_irecv: rc=%d request=%p", rc, *request);
  return rc;
}

int tcpx_test(void* request, int* done, int* size) {
  if (!g_net || !g_net->test) {
    tcpx_dbg("tcpx_test: plugin not initialized or test not available");
    return -1;
  }
  int rc = g_net->test(request, done, size);
  if (*done) {
    tcpx_dbg("tcpx_test: rc=%d done=%d size=%d", rc, *done, size ? *size : -1);
  }
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
