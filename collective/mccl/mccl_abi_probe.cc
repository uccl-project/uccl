#include "mccl_plugin_probe.h"
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: mccl_abi_probe PLUGIN.so\n";
    return 2;
  }

  auto const result = uccl::probe_mccl_net_plugin(argv[1]);
  if (!result.loaded) {
    std::cerr << "MCCL net plugin probe failed: " << result.error << "\n";
    return 1;
  }

  std::cout << "MCCL net plugin ABI v11 is loadable\n";
  return 0;
}
