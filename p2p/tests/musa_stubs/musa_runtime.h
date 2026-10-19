// Contract-test double only. This is not a MUSA SDK compatibility header.
#pragma once
#include <cstddef>
#include <cstdint>

enum musaError_t {
  musaSuccess = 0,
  musaErrorInvalidValue = 1,
  musaErrorNotReady = 34,
  musaErrorUnknown = 999
};
enum musaMemoryType {
  musaMemoryTypeHost = 1,
  musaMemoryTypeDevice = 2,
  musaMemoryTypeManaged = 3
};
struct musaPointerAttributes {
  musaMemoryType type;
  int device;
  void* devicePointer;
  void* hostPointer;
};
struct musaStream;
using musaStream_t = musaStream*;
struct musaIpcMemHandle_t {
  char reserved[64];
};
inline void* allocated_ptr = reinterpret_cast<void*>(0x12340000);
inline musaError_t pointer_result = musaSuccess;
inline musaError_t ipc_export_result = musaSuccess;
inline void* last_ipc_export_ptr = nullptr;
inline int ipc_export_calls = 0;
inline musaPointerAttributes pointer_attributes{musaMemoryTypeDevice, 3,
                                                allocated_ptr, nullptr};
inline musaError_t musaMalloc(void** ptr, size_t) {
  *ptr = allocated_ptr;
  return musaSuccess;
}
inline musaError_t musaPointerGetAttributes(musaPointerAttributes* attr,
                                            void const*) {
  if (pointer_result == musaSuccess) *attr = pointer_attributes;
  return pointer_result;
}
inline musaError_t musaIpcGetMemHandle(musaIpcMemHandle_t* handle, void* base) {
  ++ipc_export_calls;
  last_ipc_export_ptr = base;
  if (ipc_export_result == musaSuccess) handle->reserved[0] = 42;
  return ipc_export_result;
}
