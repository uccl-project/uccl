#pragma once
#include "build_identity.hpp"
#include <linux/mempolicy.h>
#include <mscclpp/env.hpp>
#include <mscclpp/utils.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>

namespace uccl_lite::runtime {
inline std::string networkTransportIdentity() {
  int memoryMode = 0;
  std::array<unsigned long, 16> memoryNodes{};
  if (syscall(SYS_get_mempolicy, &memoryMode, memoryNodes.data(),
              sizeof(memoryNodes) * 8, nullptr, 0))
    throw std::runtime_error("network caller memory policy unavailable");
  // Force verbs/provider discovery before collecting loaded implementation IDs.
  (void)mscclpp::getIBDeviceCount();
  auto configuration = mscclpp::env();
  std::string verbs =
      configuration->ibvSo.empty() ? "libibverbs.so.1" : configuration->ibvSo;
  void* library = dlopen(verbs.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library)
    throw std::runtime_error("cannot identify loaded verbs implementation");
  std::string verbsId;
  try {
    auto symbol = dlsym(library, "ibv_get_device_list");
    if (!symbol) throw std::runtime_error("verbs identity symbol unavailable");
    verbsId = artifactIdentity(reinterpret_cast<uintptr_t>(symbol));
  } catch (...) {
    dlclose(library);
    throw;
  }
  dlclose(library);
  utsname kernel{};
  if (uname(&kernel))
    throw std::runtime_error("network kernel identity unavailable");
  std::ostringstream result;
  result
      << "network-transport-v2;mscclpp="
      << artifactIdentity(
             reinterpret_cast<uintptr_t>(&mscclpp::getIBDeviceCount))
      << ";verbs=" << verbsId << ";mode="
      << configuration->ibvMode
      // This MSCCL++ flag disables GDRCopy, not GPU-memory RDMA registration.
      << ";disable-gdr=" << configuration->forceDisableGdr
      << ";gid-index=" << configuration->ibGidIndex
      << ";log-level=" << configuration->logLevel
      << ";log-subsys=" << configuration->logSubsys
      << ";debug=" << configuration->debug
      << ";debug-subsys=" << configuration->debugSubsys
      << ";kernel=" << kernel.release;
  result << ";caller-memory-mode=" << memoryMode << ";caller-memory-nodes=";
  int baseMemoryMode = memoryMode & ~MPOL_MODE_FLAGS;
  if (baseMemoryMode != MPOL_DEFAULT && baseMemoryMode != MPOL_LOCAL)
    for (auto mask : memoryNodes) result << mask << ',';
  for (auto module : {"mlx5_core", "mlx5_ib", "nvidia_peermem", "gdrdrv"}) {
    std::ifstream in(std::string("/sys/module/") + module + "/srcversion");
    std::string version;
    if (in >> version) result << ';' << module << '=' << version;
  }
  std::map<std::string, uintptr_t> providers;
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void* pointer) {
        std::string file = info->dlpi_name;
        if (file.find("/libibverbs/") == std::string::npos &&
            file.find("libmlx5") == std::string::npos &&
            file.find("libgdrapi") == std::string::npos)
          return 0;
        auto& found = *static_cast<std::map<std::string, uintptr_t>*>(pointer);
        for (int i = 0; i < info->dlpi_phnum; ++i)
          if (info->dlpi_phdr[i].p_type == PT_LOAD &&
              info->dlpi_phdr[i].p_memsz) {
            found[std::filesystem::path(file).filename().string()] =
                info->dlpi_addr + info->dlpi_phdr[i].p_vaddr;
            break;
          }
        return 0;
      },
      &providers);
  for (auto const& [name, address] : providers)
    result << ";provider:" << name << '=' << artifactIdentity(address);
  std::map<std::string, std::filesystem::path> devices;
  for (auto const& entry :
       std::filesystem::directory_iterator("/sys/class/infiniband"))
    devices[entry.path().filename().string()] = entry.path();
  for (auto const& [name, path] : devices) {
    for (auto field :
         {"node_guid", "sys_image_guid", "fw_ver", "hca_type", "board_id"}) {
      std::ifstream in(path / field);
      std::string value;
      if (!std::getline(in, value)) value = "unknown";
      result << ";hca:" << name << ':' << field << '=' << value;
    }
    std::map<std::string, std::filesystem::path> ports;
    for (auto const& entry :
         std::filesystem::directory_iterator(path / "ports"))
      ports[entry.path().filename().string()] = entry.path();
    for (auto const& [port, location] : ports) {
      for (auto field : {"link_layer", "lid", "sm_lid"}) {
        std::ifstream in(location / field);
        std::string value;
        if (!std::getline(in, value)) value = "unknown";
        result << ";port:" << name << ':' << port << ':' << field << '='
               << value;
      }
      std::map<std::string, std::filesystem::path> gids;
      for (auto const& entry :
           std::filesystem::directory_iterator(location / "gids"))
        gids[entry.path().filename().string()] = entry.path();
      result << ";gid-count:" << name << ':' << port << '=' << gids.size();
      for (auto const& [index, file] : gids) {
        std::ifstream in(file);
        std::string gid;
        if (!std::getline(in, gid)) gid = "unknown";
        if (gid == "0000:0000:0000:0000:0000:0000:0000:0000") continue;
        std::ifstream type(location / "gid_attrs" / "types" / index);
        std::string kind;
        if (!std::getline(type, kind)) kind = "unknown";
        result << ";gid:" << name << ':' << port << ':' << index << '=' << gid
               << '/' << kind;
      }
    }
  }
  // HCA enumeration is deliberately absent: path keys use physical NIC names.
  return result.str();
}
}  // namespace uccl_lite::runtime
