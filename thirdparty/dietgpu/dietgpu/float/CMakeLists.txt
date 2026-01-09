add_library(gpu_float_compress SHARED
  GpuFloatCompress.cu
  GpuFloatDecompress.cu
  GpuFloatInfo.cu
)
add_dependencies(gpu_float_compress
  gpu_ans
  dietgpu_utils
)

target_include_directories(gpu_float_compress PUBLIC
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
)
target_link_libraries(gpu_float_compress PUBLIC
  gpu_ans
  dietgpu_utils
)
target_link_libraries(gpu_float_compress PRIVATE
  glog::glog
)
target_compile_options(gpu_float_compress PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --generate-line-info
  #--device-debug
>)

enable_testing()
include(GoogleTest)

add_executable(float_test FloatTest.cu)
target_link_libraries(float_test
  gpu_float_compress
  gtest_main
)
gtest_discover_tests(float_test)


get_property(GLOBAL_CUDA_ARCHITECTURES GLOBAL PROPERTY CUDA_ARCHITECTURES)
set_target_properties(gpu_float_compress float_test PROPERTIES
  CUDA_ARCHITECTURES "${GLOBAL_CUDA_ARCHITECTURES}"
)
