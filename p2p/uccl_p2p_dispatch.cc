// Thin dispatcher for runtime transport selection.
// Reads UCCL_P2P_TRANSPORT env var and dlopen()s the corresponding backend .so.
// All uccl_engine_* calls are forwarded through the backend's ops table.

#include "uccl_p2p_ops.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <dlfcn.h>

static uccl_p2p_ops* g_ops = nullptr;
static void* g_handle = nullptr;
static std::once_flag g_init_flag;

static void init_backend() {
  char const* transport = std::getenv("UCCL_P2P_TRANSPORT");
  if (!transport || transport[0] == '\0') transport = "rdma";

  // Normalize: tcp and tcpx are aliases for the unified nccl backend.
  std::string backend = transport;
  if (backend == "tcp" || backend == "tcpx") backend = "nccl";

  char lib_name[256];
  std::snprintf(lib_name, sizeof(lib_name), "libuccl_p2p_%s.so", backend.c_str());

  g_handle = dlopen(lib_name, RTLD_NOW | RTLD_LOCAL);
  if (!g_handle) {
    std::fprintf(stderr,
                 "UCCL P2P: failed to load backend '%s' (%s): %s\n"
                 "  Set UCCL_P2P_TRANSPORT to one of: rdma, efa, tcp/tcpx\n",
                 transport, lib_name, dlerror());
    std::abort();
  }

  auto get_ops = reinterpret_cast<uccl_p2p_ops* (*)()>(
      dlsym(g_handle, "uccl_p2p_get_ops"));
  if (!get_ops) {
    std::fprintf(stderr,
                 "UCCL P2P: %s is missing uccl_p2p_get_ops symbol: %s\n",
                 lib_name, dlerror());
    std::abort();
  }

  g_ops = get_ops();
  std::fprintf(stderr, "UCCL P2P: loaded transport backend '%s'\n", transport);
}

static inline uccl_p2p_ops* ops() {
  std::call_once(g_init_flag, init_backend);
  return g_ops;
}

// ---------------------------------------------------------------------------
// Forwarding implementations for every uccl_engine_* function.
// ---------------------------------------------------------------------------

uccl_engine_t* uccl_engine_create(int num_cpus, bool in_python) {
  return ops()->create(num_cpus, in_python);
}

void uccl_engine_destroy(uccl_engine_t* engine) { ops()->destroy(engine); }

uccl_conn_t* uccl_engine_connect(uccl_engine_t* engine, char const* ip_addr,
                                 char const* remote_gpu, int remote_port,
                                 bool same_process) {
  return ops()->connect(engine, ip_addr, remote_gpu, remote_port,
                        same_process);
}

uccl_conn_t* uccl_engine_accept(uccl_engine_t* engine, char* ip_addr_buf,
                                size_t ip_addr_buf_len, int* remote_gpu_idx) {
  return ops()->accept(engine, ip_addr_buf, ip_addr_buf_len, remote_gpu_idx);
}

int uccl_engine_reg(uccl_engine_t* engine, uintptr_t data, size_t size,
                    uccl_mr_t& mr_id) {
  return ops()->reg(engine, data, size, mr_id);
}

int uccl_engine_read(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                     size_t size, FifoItem fifo_item, uint64_t* transfer_id) {
  return ops()->read(conn, mr, data, size, fifo_item, transfer_id);
}

int uccl_engine_read_vector(uccl_conn_t* conn, std::vector<uccl_mr_t> mr_ids,
                            std::vector<void*> dst_v,
                            std::vector<size_t> size_v,
                            std::vector<FifoItem> fifo_items, int num_iovs,
                            uint64_t* transfer_id,
                            std::vector<char*> ipc_bufs) {
  return ops()->read_vector(conn, mr_ids, dst_v, size_v, fifo_items, num_iovs,
                            transfer_id, ipc_bufs);
}

int uccl_engine_send(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                     size_t size, uint64_t* transfer_id) {
  return ops()->send(conn, mr, data, size, transfer_id);
}

int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t mr, void const* data,
                      size_t size, FifoItem fifo_item, uint64_t* transfer_id) {
  return ops()->write(conn, mr, data, size, fifo_item, transfer_id);
}

int uccl_engine_write_vector(uccl_conn_t* conn, std::vector<uccl_mr_t> mr_ids,
                             std::vector<void*> dst_v,
                             std::vector<size_t> size_v,
                             std::vector<FifoItem> fifo_items, int num_iovs,
                             uint64_t* transfer_id,
                             std::vector<char*> ipc_bufs) {
  return ops()->write_vector(conn, mr_ids, dst_v, size_v, fifo_items, num_iovs,
                             transfer_id, ipc_bufs);
}

int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t mr, void* data,
                     size_t max_size) {
  return ops()->recv(conn, mr, data, max_size);
}

bool uccl_engine_xfer_status(uccl_conn_t* conn, uint64_t transfer_id) {
  return ops()->xfer_status(conn, transfer_id);
}

int uccl_engine_start_listener(uccl_conn_t* conn) {
  return ops()->start_listener(conn);
}

void uccl_engine_stop_accept(uccl_engine_t* engine) {
  ops()->stop_accept(engine);
}

void uccl_engine_conn_destroy(uccl_conn_t* conn) { ops()->conn_destroy(conn); }

void uccl_engine_mr_destroy(uccl_engine_t* engine, uccl_mr_t mr) {
  ops()->mr_destroy(engine, mr);
}

int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata_str) {
  return ops()->get_metadata(engine, metadata_str);
}

std::vector<notify_msg_t> uccl_engine_get_notifs() {
  return ops()->get_notifs();
}

int uccl_engine_send_notif(uccl_conn_t* conn, notify_msg_t* notify_msg) {
  return ops()->send_notif(conn, notify_msg);
}

int uccl_engine_prepare_fifo(uccl_engine_t* engine, uccl_mr_t mr,
                             void const* data, size_t size, char* fifo_buf) {
  return ops()->prepare_fifo(engine, mr, data, size, fifo_buf);
}

int uccl_engine_update_fifo(FifoItem& fifo_item, uint64_t remote_addr,
                            uint32_t size) {
  return ops()->update_fifo(fifo_item, remote_addr, size);
}

bool uccl_engine_conn_is_local(uccl_conn_t* conn) {
  return ops()->conn_is_local(conn);
}

int uccl_engine_get_ipc_info(uccl_engine_t* engine, uintptr_t addr,
                             char* ipc_buf, bool* has_ipc) {
  return ops()->get_ipc_info(engine, addr, ipc_buf, has_ipc);
}

int uccl_engine_update_ipc_info(char* ipc_buf, uintptr_t addr,
                                uintptr_t base_addr, size_t size) {
  return ops()->update_ipc_info(ipc_buf, addr, base_addr, size);
}
