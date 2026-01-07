add_library(gpu_ans SHARED
  GpuANSDecode.cu
  GpuANSEncode.cu
  GpuANSInfo.cu
)
add_dependencies(gpu_ans
  dietgpu_utils
)
target_include_directories(gpu_ans PUBLIC
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
)
target_link_libraries(gpu_ans PUBLIC
  dietgpu_utils
)
target_link_libraries(gpu_ans PRIVATE
  glog::glog
)
target_compile_options(gpu_ans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --generate-line-info
  #--device-debug
>)


enable_testing()
include(GoogleTest)

add_executable(ans_test ANSTest.cu)
target_link_libraries(ans_test
  gpu_ans
  gtest_main
)
gtest_discover_tests(ans_test)

add_executable(ans_statistics_test ANSStatisticsTest.cu)
target_link_libraries(ans_statistics_test
  gpu_ans
  gtest_main
  dietgpu_utils
)
gtest_discover_tests(ans_statistics_test)

add_executable(batch_prefix_sum_test BatchPrefixSumTest.cu)
target_link_libraries(batch_prefix_sum_test
  gpu_ans
  gtest_main
)
gtest_discover_tests(batch_prefix_sum_test)

get_property(GLOBAL_CUDA_ARCHITECTURES GLOBAL PROPERTY CUDA_ARCHITECTURES)
set_target_properties(gpu_ans ans_test ans_statistics_test batch_prefix_sum_test
  PROPERTIES CUDA_ARCHITECTURES "${GLOBAL_CUDA_ARCHITECTURES}"
)
