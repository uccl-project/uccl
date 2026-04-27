add_deep_ep_library(runtime_nccl_cuda nccl.cu)
add_deep_ep_library(runtime_cuda_driver cuda_driver.cu)

# Link these libraries later
set(RUNTIME_CUDA_LIBRARIES runtime_nccl_cuda runtime_cuda_driver CACHE INTERNAL "Runtime kernels")
