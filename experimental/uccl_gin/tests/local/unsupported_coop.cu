#include "thirdparty/nccl-ep/adapter/uccl_gin_net.cuh"

__global__ void unsupported(uccl_gin::UCCLGinResources resources) {
  nccl_ep_adapter::UcclGinNet(resources, 0).flush(ncclCoopCta());
}
