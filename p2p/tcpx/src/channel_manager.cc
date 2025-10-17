/**
 * @file channel_manager.cc
 * @brief Implementation of multi-channel TCPX connection manager
 */

#include "channel_manager.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace {

std::string trim(std::string const& s) {
  size_t start = s.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\n\r");
  return s.substr(start, end - start + 1);
}

std::string to_lower_copy(std::string v) {
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v;
}

bool read_gpu_pci_bdf(int gpu_id, std::string& bdf_out) {
  // Try CUDA runtime first. Force a lightweight runtime init so GPU 0 works
  // even before the caller sets a CUDA context.
  cudaError_t rt_err = cudaSetDevice(gpu_id);
  if (rt_err != cudaSuccess && rt_err != cudaErrorSetOnActiveProcess) {
    // clear the sticky error to avoid impacting later CUDA calls
    (void)cudaGetLastError();
  }

  char bus_id[64] = {0};
  if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), gpu_id) == cudaSuccess) {
    bdf_out = trim(bus_id);
    if (bdf_out.size() >= 12) bdf_out = bdf_out.substr(bdf_out.size() - 12);
    if (!bdf_out.empty()) return true;
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, gpu_id) == cudaSuccess) {
    char prop_bus_id[32] = {0};
    std::snprintf(prop_bus_id, sizeof(prop_bus_id), "%04x:%02x:%02x.%d",
                  prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0);
    bdf_out = trim(prop_bus_id);
    if (bdf_out.size() >= 12) bdf_out = bdf_out.substr(bdf_out.size() - 12);
    if (!bdf_out.empty()) return true;
  }

  if (cuInit(0) != CUDA_SUCCESS) return false;
  CUdevice dev;
  if (cuDeviceGet(&dev, gpu_id) != CUDA_SUCCESS) return false;

  char driver_bus_id[64] = {0};
  if (cuDeviceGetPCIBusId(driver_bus_id, sizeof(driver_bus_id), dev) ==
      CUDA_SUCCESS) {
    bdf_out = trim(driver_bus_id);
    if (bdf_out.size() >= 12) bdf_out = bdf_out.substr(bdf_out.size() - 12);
    if (!bdf_out.empty()) return true;
  }

  int domain = 0, bus = 0;
  bool have_attrs = true;
  have_attrs &= cuDeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                     dev) == CUDA_SUCCESS;
  have_attrs &= cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                     dev) == CUDA_SUCCESS;
  if (!have_attrs) return false;

  // In absence of the exact device/function numbers, fall back to bus-level
  // matching.
  char attr_bus_id[32] = {0};
  std::snprintf(attr_bus_id, sizeof(attr_bus_id), "%04x:%02x:00.0", domain,
                bus);
  bdf_out = trim(attr_bus_id);
  if (bdf_out.size() >= 12) bdf_out = bdf_out.substr(bdf_out.size() - 12);
  return !bdf_out.empty();
}

std::string canonical_path(std::string const& path) {
  if (path.empty()) return path;
  std::error_code ec;
  auto p = std::filesystem::path(path);
  auto canonical = std::filesystem::weakly_canonical(p, ec);
  if (ec) return path;
  return canonical.string();
}

int read_numa_node_from_path(std::string const& sysfs_path) {
  if (sysfs_path.empty()) return -1;
  std::error_code ec;
  std::filesystem::path p(sysfs_path);
  if (!std::filesystem::exists(p, ec)) return -1;
  std::filesystem::path numa_file = p / "numa_node";
  if (!std::filesystem::exists(numa_file, ec)) return -1;
  std::ifstream f(numa_file);
  if (!f) return -1;
  int node = -1;
  f >> node;
  return node;
}

int read_numa_node_from_bdf(std::string const& bdf) {
  if (bdf.empty()) return -1;
  std::string lower_bdf = to_lower_copy(bdf);
  return read_numa_node_from_path("/sys/bus/pci/devices/" + lower_bdf);
}

std::vector<std::string> extract_pci_segments(std::string const& path) {
  std::vector<std::string> segments;
  if (path.empty()) return segments;
  std::stringstream ss(path);
  std::string item;
  while (std::getline(ss, item, '/')) {
    if (item.find(':') != std::string::npos && item.size() >= 7) {
      segments.push_back(item);
    }
  }
  return segments;
}

int compute_pci_score(std::vector<std::string> const& gpu_path,
                      std::vector<std::string> const& nic_path) {
  if (gpu_path.empty() || nic_path.empty()) return -1000;
  size_t min_len = std::min(gpu_path.size(), nic_path.size());
  size_t common = 0;
  for (size_t i = 0; i < min_len; ++i) {
    if (gpu_path[i] == nic_path[i]) {
      ++common;
    } else {
      break;
    }
  }
  if (common == 0) return -1000;
  int distance =
      static_cast<int>(gpu_path.size() + nic_path.size() - 2 * common);
  return static_cast<int>(common * 100 - distance);
}

}  // namespace

ChannelManager::ChannelManager(int num_channels, int gpu_id)
    : num_channels_(num_channels), gpu_id_(gpu_id) {
  // Validate against actual TCPX device count
  int tcpx_dev_count = tcpx_get_device_count();
  if (tcpx_dev_count < 0) {
    std::cerr << "[ChannelManager] Failed to get TCPX device count"
              << std::endl;
    num_channels_ = 0;
    return;
  }

  if (num_channels_ > tcpx_dev_count) {
    std::cerr
        << "[ChannelManager] Info: Requested " << num_channels_
        << " channels but only " << tcpx_dev_count
        << " TCPX devices are present. Reusing NICs to satisfy the request."
        << std::endl;
  }

  std::string gpu_bdf;
  std::vector<std::string> gpu_pci_segments;
  int gpu_numa = -1;
  if (read_gpu_pci_bdf(gpu_id_, gpu_bdf)) {
    std::string gpu_sysfs = canonical_path("/sys/bus/pci/devices/" + gpu_bdf);
    gpu_pci_segments = extract_pci_segments(gpu_sysfs);
    std::cout << "[ChannelManager] GPU " << gpu_id_ << " PCI BDF " << gpu_bdf
              << " (" << gpu_sysfs << ")" << std::endl;
    gpu_numa = read_numa_node_from_path(gpu_sysfs);
    if (gpu_numa < 0) {
      gpu_numa = read_numa_node_from_bdf(gpu_bdf);
    }
    if (gpu_numa >= 0) {
      std::cout << "[ChannelManager] GPU " << gpu_id_ << " NUMA node "
                << gpu_numa << std::endl;
    }
  } else {
    std::cerr
        << "[ChannelManager] Warning: Unable to determine PCI path for GPU "
        << gpu_id_ << ". Falling back to naive NIC ordering." << std::endl;
  }

  struct Candidate {
    int dev = -1;
    tcpx_net_properties props{};
    std::vector<std::string> pci_segments;
    int score = -1000;
    bool cuda_supported = false;
    int numa_node = -1;
    bool numa_match = false;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(tcpx_dev_count);
  for (int dev = 0; dev < tcpx_dev_count; ++dev) {
    tcpx_net_properties props{};
    if (tcpx_get_properties(dev, &props) != 0) continue;

    Candidate cand;
    cand.dev = dev;
    cand.props = props;
    cand.cuda_supported = (props.ptr_support & NCCL_PTR_CUDA) != 0;
    std::string nic_path = props.pci_path ? props.pci_path : "";
    if (!nic_path.empty()) {
      nic_path = canonical_path(nic_path);
      cand.pci_segments = extract_pci_segments(nic_path);
      cand.score = compute_pci_score(gpu_pci_segments, cand.pci_segments);
      cand.numa_node = read_numa_node_from_path(nic_path);
    }
    if (cand.numa_node < 0 && props.pci_path) {
      cand.numa_node = read_numa_node_from_path(props.pci_path);
    }
    if (cand.numa_node < 0 && props.pci_path) {
      std::filesystem::path nic_fs(props.pci_path);
      std::string nic_bdf = nic_fs.filename().string();
      cand.numa_node = read_numa_node_from_bdf(nic_bdf);
    }
    if (gpu_numa >= 0 && cand.numa_node >= 0 && cand.numa_node == gpu_numa) {
      cand.numa_match = true;
    }
    candidates.push_back(cand);
  }

  if (candidates.empty()) {
    std::cerr << "[ChannelManager] No TCPX devices available" << std::endl;
    num_channels_ = 0;
    return;
  }

  std::vector<Candidate> sorted = candidates;
  std::sort(sorted.begin(), sorted.end(),
            [](Candidate const& a, Candidate const& b) {
              if (a.score == b.score) return a.dev < b.dev;
              return a.score > b.score;
            });

  // NCCL-style selection: prefer NICs sharing PCIe root with the GPU
  std::vector<Candidate> selected;
  selected.reserve(num_channels_);

  // Only select NICs that share the PCIe hierarchy with the GPU when known.
  for (auto const& cand : sorted) {
    if (!cand.cuda_supported) continue;
    if (!gpu_pci_segments.empty() && cand.score <= 0) continue;
    if (cand.score > 0 || gpu_pci_segments.empty()) {
      selected.push_back(cand);
      if ((int)selected.size() == num_channels_) break;
    }
  }

  if (selected.empty()) {
    std::cerr << "[ChannelManager] Warning: No GPU-direct capable NICs "
                 "detected for GPU "
              << gpu_id_ << ". Falling back to first enumerated NIC."
              << std::endl;
    selected.push_back(sorted.front());
  }

  if ((int)selected.size() < num_channels_) {
    // Phase 1 Fix: Round-robin across ALL available NICs to avoid saturating a
    // single NIC. This prevents accept stalls when multiple GPUs try to use the
    // same NIC. Build a pool of all CUDA-supported NICs for round-robin
    // distribution.
    std::vector<Candidate> pool;
    for (auto const& cand : sorted) {
      if (cand.cuda_supported) {
        pool.push_back(cand);
      }
    }

    if (pool.empty()) {
      // Fallback: if no CUDA-supported NICs, use the first available NIC
      pool.push_back(sorted.front());
    }

    std::cout << "[ChannelManager] GPU " << gpu_id_ << ": Distributing "
              << num_channels_ << " channels across " << pool.size()
              << " NICs (round-robin)" << std::endl;

    // Round-robin across all NICs in the pool
    while ((int)selected.size() < num_channels_) {
      Candidate const& src = pool[selected.size() % pool.size()];
      selected.push_back(src);
    }
  }

  channels_.resize(num_channels_);

  for (int i = 0; i < num_channels_; ++i) {
    ChannelResources& ch = channels_[i];
    auto const& cand = selected[i];

    ch.channel_id = i;
    ch.net_dev = cand.dev;

    char const* nic_name = cand.props.name ? cand.props.name : "unknown";
    char const* nic_pci = cand.props.pci_path ? cand.props.pci_path : "";
    std::cout << "[ChannelManager] Channel " << i << " → netDev " << cand.dev
              << " (" << nic_name << ", PCI=" << nic_pci
              << ", score=" << cand.score;
    if (cand.numa_node >= 0) {
      std::cout << ", numa=" << cand.numa_node;
    }
    if (cand.numa_match) {
      std::cout << ", numa-match";
    }
    std::cout << ")" << std::endl;

    ch.listen_comm = nullptr;
    ch.recv_comm = nullptr;
    ch.send_comm = nullptr;

    ch.recv_dev_handle = ch.recv_dev_handle_storage.data();
    ch.send_dev_handle = ch.send_dev_handle_storage.data();
    std::memset(ch.recv_dev_handle_storage.data(), 0,
                ch.recv_dev_handle_storage.size());
    std::memset(ch.send_dev_handle_storage.data(), 0,
                ch.send_dev_handle_storage.size());

    // Multi-memory registration maps (empty initially)
    ch.recv_mhandles.clear();
    ch.send_mhandles.clear();

    ch.bytes_transferred = 0;
    ch.chunks_processed = 0;
  }

  std::cout << "[ChannelManager] Created " << num_channels_
            << " channel(s) for GPU " << gpu_id_
            << " (TCPX devices available: " << tcpx_dev_count << ")"
            << std::endl;
}

ChannelManager::~ChannelManager() {
  // Nothing to clean up per-channel beyond sockets/memory (handled elsewhere)
}

ChannelResources& ChannelManager::get_channel(int idx) {
  // CRITICAL: Check for empty vector first (e.g., local runs without TCPX)
  if (channels_.empty()) {
    std::cerr << "[ChannelManager] FATAL: No channels available (TCPX not "
                 "initialized)"
              << std::endl;
    std::cerr << "[ChannelManager] This usually means TCPX plugin is not "
                 "loaded or no devices found"
              << std::endl;
    // Cannot return a reference to non-existent element - this is a programming
    // error
    std::abort();  // Fail fast to make misuse obvious
  }

  if (idx < 0 || idx >= num_channels_) {
    std::cerr << "[ChannelManager] FATAL: Invalid channel index " << idx
              << " (valid range: 0-" << (num_channels_ - 1) << ")" << std::endl;
    std::cerr << "[ChannelManager] This indicates a configuration bug (env "
                 "asked for more channels than available)"
              << std::endl;
    std::cerr << "[ChannelManager] Aborting to make the bug obvious instead of "
                 "silently using channel 0"
              << std::endl;
    std::abort();  // Fail fast instead of masking the bug
  }

  return channels_[idx];
}

ChannelResources& ChannelManager::get_channel_for_chunk(int chunk_idx) {
  // CRITICAL: Check for empty vector first
  if (channels_.empty()) {
    std::cerr << "[ChannelManager] FATAL: No channels available for chunk "
              << chunk_idx << std::endl;
    std::abort();  // Fail fast
  }

  int channel_idx = chunk_idx % num_channels_;
  return channels_[channel_idx];
}

int ChannelManager::server_listen_all(std::vector<ncclNetHandle_v7>& handles) {
  handles.resize(num_channels_);

  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];

    std::cout << "[ChannelManager] Channel " << ch.channel_id
              << ": Listening on netDev=" << ch.net_dev << std::endl;

    if (tcpx_listen(ch.net_dev, &handles[i], &ch.listen_comm) != 0) {
      std::cerr << "[ChannelManager] tcpx_listen failed for channel "
                << ch.channel_id << ", netDev=" << ch.net_dev << std::endl;
      return -1;
    }

    std::cout << "[ChannelManager] Channel " << ch.channel_id
              << ": listen_comm=" << ch.listen_comm << std::endl;
  }

  std::cout << "[ChannelManager] All " << num_channels_
            << " channels listening successfully" << std::endl;
  return 0;
}

int ChannelManager::server_accept_all() {
  constexpr int kMaxRetries = 100;
  constexpr int kRetryDelayMs = 100;

  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];

    std::cout << "[ChannelManager] Channel " << ch.channel_id
              << ": Accepting connection..." << std::endl;

    // Retry accept (client may not have connected yet)
    bool accepted = false;
    for (int attempt = 0; attempt < kMaxRetries; attempt++) {
      int rc =
          tcpx_accept_v5(ch.listen_comm, &ch.recv_comm, &ch.recv_dev_handle);

      // Success: recv_comm is set
      if (rc == 0 && ch.recv_comm) {
        std::cout << "[ChannelManager] Channel " << ch.channel_id
                  << ": Connection accepted, recv_comm=" << ch.recv_comm
                  << std::endl;
        accepted = true;
        break;
      }

      // Transient error or not ready yet: retry
      // rc=2 typically means "not ready" or "would block"
      if (rc != 0 || !ch.recv_comm) {
        if (attempt == 0) {
          std::cout << "[ChannelManager] Channel " << ch.channel_id
                    << ": Accept not ready (rc=" << rc << "), retrying..."
                    << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDelayMs));
        continue;
      }
    }

    if (!accepted) {
      std::cerr << "[ChannelManager] Failed to accept connection for channel "
                << ch.channel_id << " after " << kMaxRetries << " retries"
                << std::endl;
      return -1;
    }
  }

  std::cout << "[ChannelManager] All " << num_channels_
            << " channels accepted successfully" << std::endl;
  return 0;
}

int ChannelManager::client_connect_all(
    std::vector<ncclNetHandle_v7> const& handles) {
  if (handles.size() != static_cast<size_t>(num_channels_)) {
    std::cerr << "[ChannelManager] Handle count mismatch: expected "
              << num_channels_ << ", got " << handles.size() << std::endl;
    return -1;
  }

  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];

    std::cout << "[ChannelManager] Channel " << ch.channel_id
              << ": Connecting to netDev=" << ch.net_dev << std::endl;

    // Make a mutable copy of the handle (tcpx_connect_v5 may modify it)
    ncclNetHandle_v7 handle_copy = handles[i];

    if (tcpx_connect_v5(ch.net_dev, &handle_copy, &ch.send_comm,
                        &ch.send_dev_handle) != 0 ||
        !ch.send_comm) {
      std::cerr << "[ChannelManager] tcpx_connect_v5 failed for channel "
                << ch.channel_id << ", netDev=" << ch.net_dev << std::endl;
      return -1;
    }

    std::cout << "[ChannelManager] Channel " << ch.channel_id
              << ": Connected, send_comm=" << ch.send_comm << std::endl;
  }

  std::cout << "[ChannelManager] All " << num_channels_
            << " channels connected successfully" << std::endl;
  return 0;
}

int ChannelManager::register_memory(uint64_t mem_id, void* buffer, size_t size,
                                    int ptr_type, bool is_recv) {
  char const* type_str = (ptr_type == NCCL_PTR_CUDA) ? "CUDA" : "HOST";
  char const* role_str = is_recv ? "recv" : "send";

  std::cout << "[ChannelManager] Registering " << type_str << " memory for "
            << role_str << ": mem_id=" << mem_id << ", ptr=" << buffer
            << ", size=" << size << std::endl;

  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    void* comm = is_recv ? ch.recv_comm : ch.send_comm;

    if (!comm) {
      std::cerr << "[ChannelManager] Channel " << ch.channel_id
                << ": comm is null, cannot register memory" << std::endl;
      return -1;
    }

    void* mhandle = nullptr;
    if (tcpx_reg_mr(comm, buffer, size, ptr_type, &mhandle) != 0) {
      std::cerr << "[ChannelManager] tcpx_reg_mr failed for channel "
                << ch.channel_id << std::endl;
      return -1;
    }

    // Store mhandle in the appropriate map
    if (is_recv) {
      ch.recv_mhandles[mem_id] = mhandle;
    } else {
      ch.send_mhandles[mem_id] = mhandle;
    }

    std::cout << "[ChannelManager] Channel " << ch.channel_id
              << ": Memory registered, mem_id=" << mem_id
              << ", mhandle=" << mhandle << std::endl;
  }

  std::cout << "[ChannelManager] All " << num_channels_
            << " channels registered mem_id=" << mem_id << " successfully"
            << std::endl;
  return 0;
}

int ChannelManager::deregister_memory(uint64_t mem_id, bool is_recv) {
  char const* role_str = is_recv ? "recv" : "send";

  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    void* comm = is_recv ? ch.recv_comm : ch.send_comm;

    auto& mhandle_map = is_recv ? ch.recv_mhandles : ch.send_mhandles;
    auto it = mhandle_map.find(mem_id);

    if (it != mhandle_map.end()) {
      if (comm && it->second) {
        tcpx_dereg_mr(comm, it->second);
        std::cout << "[ChannelManager] Channel " << ch.channel_id
                  << ": Deregistered mem_id=" << mem_id
                  << ", mhandle=" << it->second << std::endl;
      }
      mhandle_map.erase(it);
    }
  }

  std::cout << "[ChannelManager] Deregistered mem_id=" << mem_id << " for "
            << role_str << std::endl;
  return 0;
}

void* ChannelManager::get_mhandle(uint64_t mem_id, bool is_recv,
                                  int channel_id) {
  if (channel_id < 0 || channel_id >= num_channels_) {
    return nullptr;
  }

  ChannelResources& ch = channels_[channel_id];
  auto& mhandle_map = is_recv ? ch.recv_mhandles : ch.send_mhandles;
  auto it = mhandle_map.find(mem_id);

  return (it != mhandle_map.end()) ? it->second : nullptr;
}

void ChannelManager::close_all(bool is_recv) {
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];

    if (is_recv) {
      if (ch.recv_comm) {
        tcpx_close_recv(ch.recv_comm);
        ch.recv_comm = nullptr;
      }
      if (ch.listen_comm) {
        tcpx_close_listen(ch.listen_comm);
        ch.listen_comm = nullptr;
      }
    } else {
      if (ch.send_comm) {
        tcpx_close_send(ch.send_comm);
        ch.send_comm = nullptr;
      }
    }
  }

  std::cout << "[ChannelManager] All channels closed" << std::endl;
}
