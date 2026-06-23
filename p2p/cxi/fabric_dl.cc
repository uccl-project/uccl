// Linker-level dlsym wrappers for libfabric functions.
// Provides exported fi_* symbols via dlopen/dlsym instead of linking -lfabric.

#include "fabric_dl.h"

static void* fabric_resolve(char const* name) {
  return fabric_dl::resolve(name);
}

extern "C" {

int fi_getinfo(uint32_t version, char const* node, char const* service,
               uint64_t flags, fi_info const* hints, fi_info** info) {
  using FnType = int (*)(uint32_t, char const*, char const*, uint64_t,
                         fi_info const*, fi_info**);
  static FnType fn = reinterpret_cast<FnType>(fabric_resolve("fi_getinfo"));
  return fn(version, node, service, flags, hints, info);
}

void fi_freeinfo(fi_info* info) {
  using FnType = void (*)(fi_info*);
  static FnType fn = reinterpret_cast<FnType>(fabric_resolve("fi_freeinfo"));
  return fn(info);
}

fi_info* fi_dupinfo(fi_info const* info) {
  using FnType = fi_info* (*)(fi_info const*);
  static FnType fn = reinterpret_cast<FnType>(fabric_resolve("fi_dupinfo"));
  return fn(info);
}

int fi_fabric(fi_fabric_attr* attr, fid_fabric** fabric, void* context) {
  using FnType = int (*)(fi_fabric_attr*, fid_fabric**, void*);
  static FnType fn = reinterpret_cast<FnType>(fabric_resolve("fi_fabric"));
  return fn(attr, fabric, context);
}

char const* fi_strerror(int errnum) {
  using FnType = char const* (*)(int);
  static FnType fn = reinterpret_cast<FnType>(fabric_resolve("fi_strerror"));
  return fn(errnum);
}

}  // extern "C"
