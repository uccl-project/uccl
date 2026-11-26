// Host DRAM STREAM-style triad: two reads and one write per element.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <omp.h>

int main(int argc, char** argv) {
  try {
    std::size_t const n = 1ULL << 26;  // 512 MiB per array, 1.5 GiB total.
    std::string gate;
    if (argc == 3 && std::strcmp(argv[1], "--gate") == 0) gate = argv[2];
    else if (argc != 1) throw std::invalid_argument("usage: stream_triad [--gate FILE]");
    auto allocate = [&] {
      void* p = std::aligned_alloc(64, n * sizeof(double));
      if (!p) throw std::bad_alloc();
      return static_cast<double*>(p);
    };
    double* a = allocate();
    double* b = allocate();
    double* c = allocate();
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = 1.0;
      b[i] = 2.0;
      c[i] = 0.0;
    }
    std::cout << "# elements=" << n << " bytes_per_array=" << n * sizeof(double)
              << " threads=" << omp_get_max_threads() << "\n";
    if (!gate.empty()) {
      std::cout << "READY" << std::endl;
      while (access(gate.c_str(), F_OK) != 0) usleep(1000);
    }
    std::vector<double> rates;
    for (int pass = -3; pass < 10; ++pass) {
      auto begin = std::chrono::steady_clock::now();
#pragma omp parallel for simd schedule(static)
      for (std::size_t i = 0; i < n; ++i) c[i] = a[i] + 3.0 * b[i];
      auto end = std::chrono::steady_clock::now();
      double seconds = std::chrono::duration<double>(end - begin).count();
      double gbps = 3.0 * n * sizeof(double) / seconds / 1e9;
      if (pass >= 0) {
        rates.push_back(gbps);
        std::cout << "sample," << pass << ',' << seconds << ',' << gbps << '\n';
      }
    }
    double checksum = 0.0;
    for (std::size_t i = 0; i < n; i += 4096) checksum += c[i];
    std::sort(rates.begin(), rates.end());
    std::cout << "summary," << rates.front() << ','
              << (rates[4] + rates[5]) / 2.0 << ',' << rates.back()
              << ",checksum," << checksum << '\n';
    std::free(a);
    std::free(b);
    std::free(c);
  } catch (std::exception const& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
