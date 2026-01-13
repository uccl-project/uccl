#include "tcp/tcp_endpoint.h"

#include <cstring>

namespace tcp {
namespace {
uccl::ConnID make_invalid_conn() {
  uccl::ConnID invalid{};
  invalid.flow_id = UINT64_MAX;
  return invalid;
}
}  // namespace

TCPEndpoint::TCPEndpoint(int gpu_index, uint16_t port)
    : gpu_index_(gpu_index), listen_port_(port), listen_fd_(-1) {}

TCPEndpoint::~TCPEndpoint() = default;

uccl::ConnID TCPEndpoint::uccl_connect(int dev, int local_gpuidx, int remote_dev,
                                       int remote_gpuidx,
                                       std::string remote_ip,
                                       uint16_t remote_port) {
  (void)dev;
  (void)local_gpuidx;
  (void)remote_dev;
  (void)remote_gpuidx;
  (void)remote_ip;
  (void)remote_port;
  return make_invalid_conn();
}

uccl::ConnID TCPEndpoint::uccl_accept(int dev, int listen_fd, int local_gpuidx,
                                      std::string& remote_ip, int* remote_dev,
                                      int* remote_gpuidx) {
  (void)dev;
  (void)listen_fd;
  (void)local_gpuidx;
  remote_ip.clear();
  if (remote_dev) *remote_dev = 0;
  if (remote_gpuidx) *remote_gpuidx = 0;
  return make_invalid_conn();
}

int TCPEndpoint::uccl_regmr(uccl::UcclFlow* flow, void* data, size_t len,
                            int type, struct uccl::Mhandle** mhandle) {
  (void)flow;
  (void)data;
  (void)len;
  (void)type;
  if (mhandle) *mhandle = nullptr;
  return 0;
}

int TCPEndpoint::uccl_regmr(void* data, size_t len, MRArray& mr_array) {
  (void)data;
  (void)len;
  (void)mr_array;
  return 0;
}

int TCPEndpoint::uccl_regmr(int dev, void* data, size_t len, int type,
                            struct uccl::Mhandle** mhandle) {
  (void)dev;
  (void)data;
  (void)len;
  (void)type;
  if (mhandle) *mhandle = nullptr;
  return 0;
}

void TCPEndpoint::uccl_deregmr(struct uccl::Mhandle* mhandle) {
  (void)mhandle;
}

void TCPEndpoint::uccl_deregmr(MRArray const& mr_array) { (void)mr_array; }

int TCPEndpoint::uccl_send_async(uccl::UcclFlow* flow,
                                 struct uccl::Mhandle* mh, void const* data,
                                 size_t size,
                                 struct uccl::ucclRequest* ureq) {
  (void)flow;
  (void)mh;
  (void)data;
  (void)size;
  if (ureq) {
    ureq->engine_idx = 0;
    ureq->n = 0;
  }
  return -1;
}

int TCPEndpoint::uccl_recv_async(uccl::UcclFlow* flow,
                                 struct uccl::Mhandle** mhandles, void** data,
                                 int* sizes, int n,
                                 struct uccl::ucclRequest* ureq) {
  (void)flow;
  (void)mhandles;
  (void)data;
  (void)sizes;
  (void)n;
  if (ureq) {
    ureq->engine_idx = 0;
    ureq->n = 0;
  }
  return -1;
}

int TCPEndpoint::uccl_read_async(uccl::UcclFlow* flow,
                                 struct uccl::Mhandle* mh, void* dst,
                                 size_t size,
                                 uccl::FifoItem const& slot_item,
                                 uccl::ucclRequest* ureq) {
  (void)flow;
  (void)mh;
  (void)dst;
  (void)size;
  (void)slot_item;
  if (ureq) {
    ureq->engine_idx = 0;
    ureq->n = 0;
  }
  return -1;
}

int TCPEndpoint::uccl_write_async(uccl::UcclFlow* flow,
                                  struct uccl::Mhandle* mh, void* src,
                                  size_t size,
                                  uccl::FifoItem const& slot_item,
                                  uccl::ucclRequest* ureq) {
  (void)flow;
  (void)mh;
  (void)src;
  (void)size;
  (void)slot_item;
  if (ureq) {
    ureq->engine_idx = 0;
    ureq->n = 0;
  }
  return -1;
}

bool TCPEndpoint::uccl_poll_ureq_once(struct uccl::ucclRequest* ureq) {
  (void)ureq;
  return false;
}

int TCPEndpoint::prepare_fifo_metadata(uccl::UcclFlow* flow,
                                       struct uccl::Mhandle** mhandle,
                                       void const* data, size_t size,
                                       char* out_buf) {
  (void)flow;
  (void)mhandle;
  uccl::FifoItem item{};
  item.addr = reinterpret_cast<uint64_t>(data);
  item.size = static_cast<uint32_t>(size);
  std::memset(item.padding, 0, sizeof(item.padding));
  uccl::serialize_fifo_item(item, out_buf);
  return 0;
}

}  // namespace tcp
