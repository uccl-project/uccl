add_library(dietgpu_utils SHARED
  DeviceUtils.cpp
  StackDeviceMemory.cpp
)

target_include_directories(dietgpu_utils PUBLIC
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
 "${CUDA_INCLUDE_DIRS}"
)
target_link_libraries(dietgpu_utils PUBLIC
  ${CUDA_LIBRARIES}
  glog::glog
)
target_compile_options(dietgpu_utils PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --generate-line-info
  #--device-debug
>)

get_property(GLOBAL_CUDA_ARCHITECTURES GLOBAL PROPERTY CUDA_ARCHITECTURES)
set_target_properties(dietgpu_utils PROPERTIES
  CUDA_ARCHITECTURES "${GLOBAL_CUDA_ARCHITECTURES}"
)
