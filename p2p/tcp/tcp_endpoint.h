#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <transport.h>  // For uccl::ConnID, uccl::FifoItem, uccl::ucclRequest.

namespace tcp {

struct MRArray {
  void* dummy = nullptr;
};

class TCPEndpoint {
 public:
  explicit TCPEndpoint(int gpu_index, uint16_t port = 0);
  ~TCPEndpoint();

  int gpuIndex() const { return gpu_index_; }

  uccl::ConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                            int remote_gpuidx, std::string remote_ip,
                            uint16_t remote_port);
  uint16_t get_p2p_listen_port(int dev) { return listen_port_; }
  int get_p2p_listen_fd(int dev) { return listen_fd_; }
  uccl::ConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                           std::string& remote_ip, int* remote_dev,
                           int* remote_gpuidx);

  int uccl_regmr(uccl::UcclFlow* flow, void* data, size_t len, int type,
                 struct uccl::Mhandle** mhandle);
  int uccl_regmr(void* data, size_t len, MRArray& mr_array);
  int uccl_regmr(int dev, void* data, size_t len, int type,
                 struct uccl::Mhandle** mhandle);
  void uccl_deregmr(struct uccl::Mhandle* mhandle);
  void uccl_deregmr(MRArray const& mr_array);

  int uccl_send_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                      void const* data, size_t size,
                      struct uccl::ucclRequest* ureq);
  int uccl_recv_async(uccl::UcclFlow* flow, struct uccl::Mhandle** mhandles,
                      void** data, int* sizes, int n,
                      struct uccl::ucclRequest* ureq);
  int uccl_read_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh, void* dst,
                      size_t size, uccl::FifoItem const& slot_item,
                      uccl::ucclRequest* ureq);
  int uccl_write_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                       void* src, size_t size, uccl::FifoItem const& slot_item,
                       uccl::ucclRequest* ureq);
  bool uccl_poll_ureq_once(struct uccl::ucclRequest* ureq);
  int prepare_fifo_metadata(uccl::UcclFlow* flow,
                            struct uccl::Mhandle** mhandle, void const* data,
                            size_t size, char* out_buf);

  int get_best_dev_idx(int gpu_idx) { return 0; }

  bool initialize_engine_by_dev(int dev, bool enable_p2p_listen) {
    return true;
  }

  void create_unified_p2p_socket() {}

 private:
  int gpu_index_;
  uint16_t listen_port_;
  int listen_fd_;
};

}  // namespace tcp
