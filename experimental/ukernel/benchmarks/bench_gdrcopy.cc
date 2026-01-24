#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <getopt.h>
#include <time.h>

#ifdef ENABLE_CUDA
#include "gdrapi.h"
#include <cuda.h>
#endif
#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

static inline double time_diff_us(timespec const& beg, timespec const& end) {
  double const s = (double)(end.tv_sec - beg.tv_sec);
  double const ns = (double)(end.tv_nsec - beg.tv_nsec);
  return s * 1e6 + ns / 1e3;
}

static inline size_t round_up(size_t x, size_t a) {
  return (x + a - 1) / a * a;
}

#if defined(__x86_64__) || defined(__i386__)
static inline void cpu_sfence() { asm volatile("sfence" ::: "memory"); }
static inline void cpu_lfence() { asm volatile("lfence" ::: "memory"); }
#else
static inline void cpu_sfence() { __sync_synchronize(); }
static inline void cpu_lfence() { __sync_synchronize(); }
#endif

static void init_hbuf_walking_bit(uint32_t* p, size_t bytes) {
  size_t n = bytes / sizeof(uint32_t);
  for (size_t i = 0; i < n; ++i) {
    p[i] = (uint32_t)(1u << (i % 32));
  }
}

static void print_usage(char const* path) {
  std::cout << "Usage: " << path << " [options]\n\n"
            << "Options:\n"
            << "  -h            Print help\n"
            << "  -d <gpu>      GPU ID (default: 0)\n"
            << "  -s <size>     Buffer size in bytes (default: 1<<24)\n"
            << "  -w <iters>    Write iters for each size (default: 10000)\n"
            << "  -r <iters>    Read iters for each size (default: 100)\n"
            << "  -c            Also run cuMemcpy H2D/D2H (default: no)\n"
            << "  -C            Use cold cache mode (only meaningful for "
               "caching mapping)\n";
}

/*
# ref: https://github.com/NVIDIA/gdrcopy/blob/master/README.md
# install gdrcopy on ubuntu
1. sudo apt update
   sudo apt install build-essential devscripts debhelper fakeroot pkg-config
dkms
2. git submodule update --init --recursive
   cd thirdparty/gdrcopy
   make CUDA=/usr/local/cuda all
   sudo make install
3. sudo ./insmod.sh
4. check:
   lsmod | grep -E '^gdrdrv\b' || true
*/
void bench_nvidia_gpu_gdrcopy(int dev_id, size_t req_size, int num_write_iters,
                              int num_read_iters, bool do_cumemcpy,
                              bool use_cold_cache) {
#ifdef ENABLE_CUDA
  auto CK = [](CUresult r, char const* msg) {
    if (r != CUDA_SUCCESS) {
      const char* name = nullptr;
      const char* str = nullptr;
      cuGetErrorName(r, &name);
      cuGetErrorString(r, &str);
      std::fprintf(stderr, "CUDA ERROR: %s: %s (%s)\n", msg, name ? name : "?",
                   str ? str : "?");
      std::exit(1);
    }
  };

  CK(cuInit(0), "cuInit");

  int ndev = 0;
  CK(cuDeviceGetCount(&ndev), "cuDeviceGetCount");
  if (ndev <= 0) {
    std::fprintf(stderr, "No CUDA devices found.\n");
    return;
  }
  if (dev_id < 0 || dev_id >= ndev) {
    std::fprintf(stderr, "Invalid GPU id %d (found %d devices)\n", dev_id,
                 ndev);
    return;
  }

  std::cout << "Detected " << ndev << " CUDA device(s):\n";
  for (int n = 0; n < ndev; ++n) {
    CUdevice dev;
    CK(cuDeviceGet(&dev, n), "cuDeviceGet");
    char name[256];
    CK(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    int dom = 0, bus = 0, devfn = 0;
    CK(cuDeviceGetAttribute(&dom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev),
       "pci domain");
    CK(cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev),
       "pci bus");
    CK(cuDeviceGetAttribute(&devfn, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev),
       "pci device");
    std::cout << "  GPU id:" << n << "; name:" << name << "; Bus:" << std::hex
              << std::setfill('0') << std::setw(4) << dom << ":" << std::setw(2)
              << bus << ":" << std::setw(2) << devfn << std::dec << "\n";
  }

  CUdevice dev;
  CK(cuDeviceGet(&dev, dev_id), "cuDeviceGet(selected)");
  CUcontext ctx;
  CK(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  CK(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

  // mapping 64KB boundary
  size_t page_size = 64 * 1024;
  size_t size = round_up(req_size, page_size);

  CUdeviceptr d_A = 0;
  CK(cuMemAlloc(&d_A, size), "cuMemAlloc");
  std::cout << "Selected GPU " << dev_id << ", allocated " << size
            << " bytes, dptr=0x" << std::hex << (uint64_t)d_A << std::dec
            << "\n";

  // host pinned buffers
  uint32_t* init_buf = nullptr;
  uint32_t* h_buf = nullptr;
  CK(cuMemAllocHost((void**)&init_buf, size), "cuMemAllocHost(init_buf)");
  CK(cuMemAllocHost((void**)&h_buf, size), "cuMemAllocHost(h_buf)");
  init_hbuf_walking_bit(init_buf, size);

  // optional cuMemcpy bench (Driver API uses cuMemcpy)
  if (do_cumemcpy) {
    std::cout << "\ncuMemcpy_H2D iters=" << num_write_iters << "\n";
    std::printf("Test\t\t\tSize(B)\t\tAvg.Time(us)\n");
    for (size_t copy_size = 1; copy_size <= size; copy_size <<= 1) {
      timespec beg{}, end{};
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_write_iters; ++i) {
        CK(cuMemcpy(d_A, (CUdeviceptr)init_buf, copy_size), "cuMemcpy H2D");
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      double lat = time_diff_us(beg, end) / (double)num_write_iters;
      std::printf("cuMemcpy_H2D\t\t%8zu\t\t%11.4f\n", copy_size, lat);
    }

    std::cout << "\ncuMemcpy_D2H iters=" << num_read_iters << "\n";
    std::printf("Test\t\t\tSize(B)\t\tAvg.Time(us)\n");
    for (size_t copy_size = 1; copy_size <= size; copy_size <<= 1) {
      timespec beg{}, end{};
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_read_iters; ++i) {
        CK(cuMemcpy((CUdeviceptr)h_buf, d_A, copy_size), "cuMemcpy D2H");
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      double lat = time_diff_us(beg, end) / (double)num_read_iters;
      std::printf("cuMemcpy_D2H\t\t%8zu\t\t%11.4f\n", copy_size, lat);
    }
    std::cout << "\n";
  }

  // gdrcopy path
  gdr_t g = gdr_open();
  if (!g) {
    std::fprintf(
        stderr,
        "ERROR: gdr_open failed. Is gdrdrv loaded? (lsmod | grep gdrdrv)\n");
    std::exit(1);
  }

  gdr_mh_t mh{};
  int rc = gdr_pin_buffer(g, d_A, size, 0, 0, &mh);
  if (rc != 0) {
    std::fprintf(
        stderr,
        "ERROR: gdr_pin_buffer failed rc=%d (need root? or driver mismatch?)\n",
        rc);
    std::exit(1);
  }

  void* map_d_ptr = nullptr;
  rc = gdr_map(g, mh, &map_d_ptr, size);
  if (rc != 0 || !map_d_ptr) {
    std::fprintf(stderr, "ERROR: gdr_map failed rc=%d map=%p\n", rc, map_d_ptr);
    std::exit(1);
  }

  gdr_info_t info{};
  rc = gdr_get_info(g, mh, &info);
  if (rc != 0) {
    std::fprintf(stderr, "ERROR: gdr_get_info failed rc=%d\n", rc);
    std::exit(1);
  }

  std::cout << "gdr_map ptr: " << map_d_ptr << "\n";
  std::cout << "info.va=0x" << std::hex << (uint64_t)info.va << std::dec
            << " mapped_size=" << info.mapped_size
            << " page_size=" << info.page_size
            << " wc_mapping=" << info.wc_mapping
            << " mapping_type=" << info.mapping_type << "\n";

  // mapping 64KB aligned offset
  int const off = (int)((uint64_t)info.va - (uint64_t)d_A);
  std::cout << "page offset: " << off << " bytes\n";

  uint8_t* buf_ptr_u8 = (uint8_t*)map_d_ptr + off;

  std::cout << "use cold cache: " << (use_cold_cache ? "yes" : "no") << "\n";
  if (use_cold_cache && info.mapping_type != GDR_MAPPING_TYPE_CACHING) {
    std::fprintf(stderr,
                 "ERROR: -C only meaningful for caching mapping (e.g., "
                 "Grace-Hopper cache mapping)\n");
    std::exit(1);
  }

  std::cout << "\n[gdr_copy_to_mapping] write iters=" << num_write_iters
            << "\n";
  std::cout << "WARNING: measures CPU-side API invocation latency; may not "
               "imply full GPU visibility ordering.\n";
  std::printf("Test\t\t\t\tSize(B)\t\tAvg.Time(us)\n");

  for (size_t copy_size = 1; copy_size <= size; copy_size <<= 1) {
    timespec beg{}, end{};
    double total_us = 0.0;

    if (use_cold_cache) {
      // total (memset + copy)
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_write_iters; ++i) {
        // sync memops: memset returns after done (driver API)
        CK(cuMemsetD8(d_A, 0, copy_size), "cuMemsetD8");
        gdr_copy_to_mapping(mh, buf_ptr_u8, init_buf, copy_size);
        cpu_sfence();
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us = time_diff_us(beg, end);

      // subtract memset cost
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_write_iters; ++i) {
        CK(cuMemsetD8(d_A, 0, copy_size), "cuMemsetD8(sub)");
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us -= time_diff_us(beg, end);
    } else {
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_write_iters; ++i) {
        gdr_copy_to_mapping(mh, buf_ptr_u8, init_buf, copy_size);
        cpu_sfence();
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us = time_diff_us(beg, end);
    }

    double lat = total_us / (double)num_write_iters;
    std::printf("gdr_copy_to_mapping\t\t%8zu\t\t%11.4f\n", copy_size, lat);
  }

  std::cout << "\n[gdr_copy_from_mapping] read iters=" << num_read_iters
            << "\n";
  std::printf("Test\t\t\t\tSize(B)\t\tAvg.Time(us)\n");

  for (size_t copy_size = 1; copy_size <= size; copy_size <<= 1) {
    timespec beg{}, end{};
    double total_us = 0.0;

    if (use_cold_cache) {
      // total (memset + copy)
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_read_iters; ++i) {
        CK(cuMemsetD8(d_A, 0, copy_size), "cuMemsetD8");
        gdr_copy_from_mapping(mh, h_buf, buf_ptr_u8, copy_size);
        cpu_lfence();
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us = time_diff_us(beg, end);

      // subtract memset cost
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_read_iters; ++i) {
        CK(cuMemsetD8(d_A, 0, copy_size), "cuMemsetD8(sub)");
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us -= time_diff_us(beg, end);
    } else {
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_read_iters; ++i) {
        gdr_copy_from_mapping(mh, h_buf, buf_ptr_u8, copy_size);
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us = time_diff_us(beg, end);
    }

    double lat = total_us / (double)num_read_iters;
    std::printf("gdr_copy_from_mapping\t\t%8zu\t\t%11.4f\n", copy_size, lat);
  }

  // cleanup
  std::cout << "\nCleaning up...\n";
  rc = gdr_unmap(g, mh, map_d_ptr, size);
  if (rc != 0) std::fprintf(stderr, "WARN: gdr_unmap rc=%d\n", rc);

  rc = gdr_unpin_buffer(g, mh);
  if (rc != 0) std::fprintf(stderr, "WARN: gdr_unpin_buffer rc=%d\n", rc);

  rc = gdr_close(g);
  if (rc != 0) std::fprintf(stderr, "WARN: gdr_close rc=%d\n", rc);

  CK(cuMemFreeHost(init_buf), "cuMemFreeHost(init_buf)");
  CK(cuMemFreeHost(h_buf), "cuMemFreeHost(h_buf)");
  CK(cuMemFree(d_A), "cuMemFree");

  std::cout << "Done.\n";
#endif
}

void bench_amd_gpu_gdrcopy(int dev_id, size_t req_size, int num_write_iters,
                           int num_read_iters, bool do_hipmemcpy,
                           bool use_cold_cache) {
#ifdef ENABLE_HIP
  auto HK = [](hipError_t e, char const* msg) {
    if (e != hipSuccess) {
      std::fprintf(stderr, "HIP ERROR: %s: %s\n", msg, hipGetErrorString(e));
      std::exit(1);
    }
  };

  int ndev = 0;
  HK(hipGetDeviceCount(&ndev), "hipGetDeviceCount");
  if (ndev <= 0) {
    std::fprintf(stderr, "No HIP devices found.\n");
    return;
  }
  if (dev_id < 0 || dev_id >= ndev) {
    std::fprintf(stderr, "Invalid GPU id %d (found %d devices)\n", dev_id,
                 ndev);
    return;
  }

  std::cout << "Detected " << ndev << " HIP device(s):\n";
  for (int n = 0; n < ndev; ++n) {
    hipDeviceProp_t prop{};
    HK(hipGetDeviceProperties(&prop, n), "hipGetDeviceProperties");
    std::cout << "  GPU id:" << n << "; name:" << prop.name
              << "; pciDomainID:" << prop.pciDomainID
              << "; pciBusID:" << prop.pciBusID
              << "; pciDeviceID:" << prop.pciDeviceID << "\n";
  }

  HK(hipSetDevice(dev_id), "hipSetDevice");

  // 64KB
  size_t page_size = 64 * 1024;
  size_t size = round_up(req_size, page_size);

  // hipMalloc device memory
  uint8_t* d_A = nullptr;
  HK(hipMalloc((void**)&d_A, size), "hipMalloc(d_A)");
  std::cout << "Selected GPU " << dev_id << ", allocated DEVICE memory " << size
            << " bytes, dptr=" << (void*)d_A << "\n";

  // host pinned buffers：init / readback
  uint32_t* init_buf = nullptr;
  uint32_t* h_buf = nullptr;
  HK(hipHostMalloc((void**)&init_buf, size, hipHostMallocDefault),
     "hipHostMalloc(init_buf)");
  HK(hipHostMalloc((void**)&h_buf, size, hipHostMallocDefault),
     "hipHostMalloc(h_buf)");
  init_hbuf_walking_bit(init_buf, size);

  // host touch device pointer
  std::cout << "Probing host access to device pointer...\n";
  uint8_t volatile* probe = (uint8_t volatile*)d_A;
  probe[0] = 0;
  (void)probe[0];
  std::cout << "Probe ok: host can access device memory pointer.\n";

  std::cout << "use cold cache: " << (use_cold_cache ? "yes" : "no") << "\n";

  // hipMemcpy（device <-> host pinned）
  if (do_hipmemcpy) {
    std::cout << "\n[hipMemcpy H2D] iters=" << num_write_iters << "\n";
    std::printf("Test\t\t\tSize(B)\t\tAvg.Time(us)\n");
    for (size_t copy_size = 1; copy_size <= size; copy_size <<= 1) {
      timespec beg{}, end{};
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_write_iters; ++i) {
        HK(hipMemcpy(d_A, init_buf, copy_size, hipMemcpyHostToDevice),
           "hipMemcpy H2D");
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      double lat = time_diff_us(beg, end) / (double)num_write_iters;
      std::printf("hipMemcpy_H2D\t\t%8zu\t\t%11.4f\n", copy_size, lat);
    }

    std::cout << "\n[hipMemcpy D2H] iters=" << num_read_iters << "\n";
    std::printf("Test\t\t\tSize(B)\t\tAvg.Time(us)\n");
    for (size_t copy_size = 1; copy_size <= size; copy_size <<= 1) {
      timespec beg{}, end{};
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_read_iters; ++i) {
        HK(hipMemcpy(h_buf, d_A, copy_size, hipMemcpyDeviceToHost),
           "hipMemcpy D2H");
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      double lat = time_diff_us(beg, end) / (double)num_read_iters;
      std::printf("hipMemcpy_D2H\t\t%8zu\t\t%11.4f\n", copy_size, lat);
    }
    std::cout << "\n";
  }

  // AMD gdrcopy-like path：CPU memcpy device memory pointer
  std::cout << "\n[CPU memcpy -> hipMalloc(device)] write iters="
            << num_write_iters << "\n";
  std::cout << "WARNING: measures CPU-side memcpy into device memory (host "
               "zero-copy access).\n";
  std::printf("Test\t\t\t\tSize(B)\t\tAvg.Time(us)\n");

  for (size_t copy_size = 1; copy_size <= size; copy_size <<= 1) {
    timespec beg{}, end{};
    double total_us = 0.0;

    if (use_cold_cache) {
      // total (memset + copy)
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_write_iters; ++i) {
        std::memset((void*)d_A, 0, copy_size);
        std::memcpy((void*)d_A, (void*)init_buf, copy_size);
        cpu_sfence();
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us = time_diff_us(beg, end);

      // subtract memset cost
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_write_iters; ++i) {
        std::memset((void*)d_A, 0, copy_size);
        cpu_sfence();
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us -= time_diff_us(beg, end);
    } else {
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_write_iters; ++i) {
        std::memcpy((void*)d_A, (void*)init_buf, copy_size);
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us = time_diff_us(beg, end);
    }

    double lat = total_us / (double)num_write_iters;
    std::printf("hipMalloc_host_write\t\t%8zu\t\t%11.4f\n", copy_size, lat);
  }

  std::cout << "\n[CPU memcpy <- hipMalloc(device)] read iters="
            << num_read_iters << "\n";
  std::printf("Test\t\t\t\tSize(B)\t\tAvg.Time(us)\n");

  for (size_t copy_size = 1; copy_size <= size; copy_size <<= 1) {
    timespec beg{}, end{};
    double total_us = 0.0;

    if (use_cold_cache) {
      // total (memset + copy)
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_read_iters; ++i) {
        std::memset((void*)h_buf, 0, copy_size);
        std::memcpy((void*)h_buf, (void*)d_A, copy_size);
        cpu_lfence();
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us = time_diff_us(beg, end);

      // subtract memset cost
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_read_iters; ++i) {
        std::memset((void*)h_buf, 0, copy_size);
        cpu_lfence();
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us -= time_diff_us(beg, end);
    } else {
      clock_gettime(CLOCK_MONOTONIC, &beg);
      for (int i = 0; i < num_read_iters; ++i) {
        std::memcpy((void*)h_buf, (void*)d_A, copy_size);
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_us = time_diff_us(beg, end);
    }

    double lat = total_us / (double)num_read_iters;
    std::printf("hipMalloc_host_read\t\t%8zu\t\t%11.4f\n", copy_size, lat);
  }

  // cleanup
  std::cout << "\nCleaning up...\n";
  HK(hipHostFree(init_buf), "hipHostFree(init_buf)");
  HK(hipHostFree(h_buf), "hipHostFree(h_buf)");
  HK(hipFree(d_A), "hipFree(d_A)");

  std::cout << "Done.\n";
#endif
}

int main(int argc, char** argv) {
  int dev_id = 0;
  size_t size = (size_t)1 << 24;
  int w = 10000;
  int r = 100;
  bool do_cumemcpy = false;
  bool cold = false;

  int opt;
  while ((opt = getopt(argc, argv, "hd:s:w:r:cC")) != -1) {
    switch (opt) {
      case 'h':
        print_usage(argv[0]);
        return 0;
      case 'd':
        dev_id = std::strtol(optarg, nullptr, 0);
        break;
      case 's':
        size = (size_t)std::strtoull(optarg, nullptr, 0);
        break;
      case 'w':
        w = std::strtol(optarg, nullptr, 0);
        break;
      case 'r':
        r = std::strtol(optarg, nullptr, 0);
        break;
      case 'c':
        do_cumemcpy = true;
        break;
      case 'C':
        cold = true;
        break;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }
  /*
  Usage:
  ./bench_gdrcopy -d 5 -s $((1<<18)) -w 10000 -r 10000 -c
  */
#ifdef ENABLE_CUDA
  bench_nvidia_gpu_gdrcopy(dev_id, size, w, r, do_cumemcpy, cold);
#endif
#ifdef ENABLE_HIP
  bench_amd_gpu_gdrcopy(dev_id, size, w, r, do_cumemcpy, cold);
#endif

  return 0;
}