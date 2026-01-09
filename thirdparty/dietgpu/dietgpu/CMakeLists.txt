find_package(Torch REQUIRED)

add_library(dietgpu SHARED
  DietGpu.cpp
)
add_dependencies(dietgpu
  gpu_float_compress
)
target_link_libraries(dietgpu PRIVATE
  gpu_float_compress
  glog::glog
  ${TORCH_LIBRARIES}
)
target_include_directories(dietgpu PRIVATE
  $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)
