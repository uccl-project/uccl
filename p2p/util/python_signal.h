#pragma once

#include "util/debug.h"
#include <Python.h>
#include <dlfcn.h>

// Called only from Python-facing waits. Resolve the active interpreter's API
// lazily so the same core can also load in a native process without libpython.
static inline void check_python_signals() {
  struct PythonSignals {
    decltype(&Py_IsInitialized) initialized;
    decltype(&PyGILState_Ensure) ensure;
    decltype(&PyErr_CheckSignals) check;
    decltype(&PyGILState_Release) release;
  };
  static PythonSignals const api{
      reinterpret_cast<decltype(&Py_IsInitialized)>(
          dlsym(RTLD_DEFAULT, "Py_IsInitialized")),
      reinterpret_cast<decltype(&PyGILState_Ensure)>(
          dlsym(RTLD_DEFAULT, "PyGILState_Ensure")),
      reinterpret_cast<decltype(&PyErr_CheckSignals)>(
          dlsym(RTLD_DEFAULT, "PyErr_CheckSignals")),
      reinterpret_cast<decltype(&PyGILState_Release)>(
          dlsym(RTLD_DEFAULT, "PyGILState_Release"))};
  if (!api.initialized || !api.ensure || !api.check || !api.release ||
      !api.initialized()) {
    UCCL_LOG(FATAL) << "Python signal polling requires an active interpreter";
  }
  PyGILState_STATE gstate = api.ensure();
  if (api.check() != 0) {
    UCCL_LOG(FATAL) << "Python signal caught, exiting...";
  }
  api.release(gstate);
}
