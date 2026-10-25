#include "mccl_plugin_probe.h"

#include <dlfcn.h>

#include <sstream>

namespace uccl {
namespace {

using PluginSymbol = const mcclNet_v11_t;

bool has_required_callbacks(const mcclNet_v11_t& plugin, std::string* error) {
  if (plugin.name == nullptr || plugin.name[0] == '\0') {
    *error = "mcclNetPlugin_v11 has no name";
    return false;
  }

  const void* callbacks[] = {
      reinterpret_cast<const void*>(plugin.init),
      reinterpret_cast<const void*>(plugin.devices),
      reinterpret_cast<const void*>(plugin.getProperties),
      reinterpret_cast<const void*>(plugin.listen),
      reinterpret_cast<const void*>(plugin.connect),
      reinterpret_cast<const void*>(plugin.accept),
      reinterpret_cast<const void*>(plugin.regMr),
      reinterpret_cast<const void*>(plugin.deregMr),
      reinterpret_cast<const void*>(plugin.isend),
      reinterpret_cast<const void*>(plugin.irecv),
      reinterpret_cast<const void*>(plugin.iflush),
      reinterpret_cast<const void*>(plugin.test),
      reinterpret_cast<const void*>(plugin.closeSend),
      reinterpret_cast<const void*>(plugin.closeRecv),
      reinterpret_cast<const void*>(plugin.closeListen),
  };
  for (size_t i = 0; i < sizeof(callbacks) / sizeof(callbacks[0]); ++i) {
    if (callbacks[i] == nullptr) {
      *error = "mcclNetPlugin_v11 is missing required callback " +
               std::to_string(i);
      return false;
    }
  }
  return true;
}

}  // namespace

McclPluginProbeResult probe_mccl_net_plugin(const char* path) {
  McclPluginProbeResult result;
  if (path == nullptr || path[0] == '\0') {
    result.error = "plugin path is empty";
    return result;
  }

  void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    result.error = dlerror();
    return result;
  }

  dlerror();
  void* raw_symbol = dlsym(handle, "mcclNetPlugin_v11");
  const char* symbol_error = dlerror();
  if (symbol_error != nullptr || raw_symbol == nullptr) {
    result.error = symbol_error == nullptr ? "mcclNetPlugin_v11 not found"
                                           : symbol_error;
    dlclose(handle);
    return result;
  }

  auto* plugin = reinterpret_cast<const PluginSymbol*>(raw_symbol);
  if (!has_required_callbacks(*plugin, &result.error)) {
    dlclose(handle);
    return result;
  }

  result.loaded = true;
  result.plugin = plugin;
  // The caller only needs the address during this probe. Do not expose a
  // dangling ABI pointer after the handle is closed.
  result.plugin = nullptr;
  dlclose(handle);
  return result;
}

}  // namespace uccl
