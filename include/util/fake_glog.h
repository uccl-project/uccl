#pragma once

#include <c10/util/Logging.h>

template <typename T>
inline T* uccl_check_notnull(const char* expr, T* ptr) {
  CHECK(ptr != nullptr) << "Check failed: " << expr << " != nullptr";
  return ptr;
}

#ifndef CHECK_NOTNULL
#define CHECK_NOTNULL(val) uccl_check_notnull(#val, (val))
#endif

#ifndef DCHECK_NOTNULL
#define DCHECK_NOTNULL(val) CHECK_NOTNULL(val)
#endif
