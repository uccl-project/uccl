# NCCL EP Example/Library CMake Configuration

if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch LESS 90)
      message(FATAL_ERROR "NCCL_EP requires sm_90 or higher. CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    endif()
  endforeach()
endif()

# Source files
set(NCCL_EP_HOST_SRC
    nccl_ep.cc
)

set(NCCL_EP_DEVICE_SRC
    device/low_latency.cu
    device/hybridep_adapter.cu
)

set(NCCL_EP_SRC
    ${NCCL_EP_HOST_SRC}
    ${NCCL_EP_DEVICE_SRC}
)

add_library(nccl_ep_shared SHARED
    ${NCCL_EP_SRC}
)
add_library(nccl_ep_static STATIC
    ${NCCL_EP_SRC}
)
add_custom_target(nccl_ep_lib DEPENDS nccl_ep_shared nccl_ep_static)

set(NCCL_EP_TARGETS nccl_ep_shared nccl_ep_static)

set_target_properties(nccl_ep_shared nccl_ep_static PROPERTIES
    OUTPUT_NAME "nccl_ep"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "90"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

set_target_properties(nccl_ep_shared PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "$ORIGIN"
)

# Configurable LSA team size range for compile-time instantiation.
# Restricting the range reduces build time during development.  Example:
#   cmake ... -D_NCCL_EP_LSA_TEAM_SIZE_MIN=8 -D_NCCL_EP_LSA_TEAM_SIZE_MAX=8
set(_NCCL_EP_LSA_TEAM_SIZE_MIN "4" CACHE STRING
    "Minimum LSA team size to compile (must be multiple of 4, min 4)")
set(_NCCL_EP_LSA_TEAM_SIZE_MAX "32" CACHE STRING
    "Maximum LSA team size to compile (must be multiple of 4, max 32)")
set(_NCCL_EP_NUM_LSA_TEAMS_LIST "1;2;3;4;8" CACHE STRING
    "List of NUM_LSA_TEAMS values to compile (subset of 1 2 3 4 8)")

set(NCCL_EP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/nccl_device
    ${CMAKE_SOURCE_DIR}/src/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

# Allow device code to call host constexpr with --expt-relaxed-constexpr
# Hides all symbols by default in the host compiled object files with -fvisibility=hidden
foreach(t IN LISTS NCCL_EP_TARGETS)
    target_compile_options(${t} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fvisibility=hidden>
    )
    target_compile_definitions(${t} PRIVATE
        _NCCL_EP_LSA_TEAM_SIZE_MIN=${_NCCL_EP_LSA_TEAM_SIZE_MIN}
        _NCCL_EP_LSA_TEAM_SIZE_MAX=${_NCCL_EP_LSA_TEAM_SIZE_MAX})
    target_include_directories(${t} PUBLIC ${NCCL_EP_INCLUDE_DIRS})
    target_link_libraries(${t} PUBLIC nccl ${CUDAToolkit_LIBRARIES})
endforeach()

foreach(N IN LISTS _NCCL_EP_NUM_LSA_TEAMS_LIST)
    foreach(t IN LISTS NCCL_EP_TARGETS)
        target_compile_definitions(${t} PRIVATE _NCCL_EP_NUM_LSA_TEAMS_${N}=1)
    endforeach()
endforeach()

# Copy headers to build/include (runs at configure)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
configure_file(include/nccl_ep.h "${CMAKE_BINARY_DIR}/include/nccl_ep.h" COPYONLY)

# Install to custom prefix
install(FILES
    include/nccl_ep.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS ${NCCL_EP_TARGETS}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
