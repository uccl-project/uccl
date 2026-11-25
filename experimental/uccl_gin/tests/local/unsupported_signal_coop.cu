#include "thirdparty/nccl-ep/adapter/uccl_gin_net.cuh"

__global__ void unsupported_signal(uccl_gin::UCCLGinResources resources) {
  nccl_ep_adapter::UcclGinNet net(resources, 0);
  net.signal({}, 1, ncclGin_SignalAdd{0, 1}, ncclCoopCta{});
}
