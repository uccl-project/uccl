#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "context.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

namespace uccl_gin {
// Defined in tests/put_quiet_smoke.cu (compiled into this extension).
bool run_put_quiet_smoke(Context& ctx, int peer, int bytes);
double run_put_bench(Context& ctx, int peer, int bytes, int iters, int warmup,
                     int bench_lanes);
}  // namespace uccl_gin

namespace {

void ensure_mpi_initialized() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  }
}

struct PyUcclGinContext {
  PyObject_HEAD
  uccl_gin::Context* ctx;
};

void PyUcclGinContext_dealloc(PyUcclGinContext* self) {
  delete self->ctx;
  self->ctx = nullptr;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* PyUcclGinContext_new(PyTypeObject* type, PyObject*, PyObject*) {
  auto* self = reinterpret_cast<PyUcclGinContext*>(type->tp_alloc(type, 0));
  if (self != nullptr) self->ctx = nullptr;
  return reinterpret_cast<PyObject*>(self);
}

int PyUcclGinContext_init(PyUcclGinContext* self, PyObject* args, PyObject* kwargs) {
  static const char* kwlist[] = {
      "max_message_bytes",
      "local_world_size",
      "ifname",
      nullptr,
  };
  unsigned long long max_message_bytes = 1 << 20;
  int local_world_size = 8;
  const char* ifname = "enp71s0";
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Kis",
                                   const_cast<char**>(kwlist),
                                   &max_message_bytes, &local_world_size,
                                   &ifname)) {
    return -1;
  }

  ensure_mpi_initialized();
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  uccl_gin::ContextConfig cfg;
  cfg.rank = rank;
  cfg.world_size = world_size;
  cfg.local_world_size = local_world_size;
  cfg.max_message_bytes = static_cast<std::size_t>(max_message_bytes);
  cfg.ifname = ifname;

  try {
    self->ctx = new uccl_gin::Context(cfg);
  } catch (const std::bad_alloc&) {
    PyErr_NoMemory();
    return -1;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "failed to create uccl_gin.Context");
    return -1;
  }
  return 0;
}

PyObject* PyUcclGinContext_close(PyUcclGinContext* self, PyObject*) {
  delete self->ctx;
  self->ctx = nullptr;
  Py_RETURN_NONE;
}

PyObject* PyUcclGinContext_put_quiet_smoke(PyUcclGinContext* self,
                                           PyObject* args) {
  if (self->ctx == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Context is closed");
    return nullptr;
  }
  int peer = 0;
  int bytes = 0;
  if (!PyArg_ParseTuple(args, "ii", &peer, &bytes)) return nullptr;
  bool ok;
  Py_BEGIN_ALLOW_THREADS
  ok = uccl_gin::run_put_quiet_smoke(*self->ctx, peer, bytes);
  Py_END_ALLOW_THREADS
  if (ok) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* PyUcclGinContext_put_bench(PyUcclGinContext* self, PyObject* args) {
  if (self->ctx == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Context is closed");
    return nullptr;
  }
  int peer = 0;
  int bytes = 0;
  int iters = 0;
  int warmup = 0;
  int bench_lanes = 1;
  if (!PyArg_ParseTuple(args, "iii|ii", &peer, &bytes, &iters, &warmup,
                        &bench_lanes))
    return nullptr;
  double gbps;
  Py_BEGIN_ALLOW_THREADS
  gbps = uccl_gin::run_put_bench(*self->ctx, peer, bytes, iters, warmup,
                                 bench_lanes);
  Py_END_ALLOW_THREADS
  return PyFloat_FromDouble(gbps);
}

PyObject* PyUcclGinContext_num_queues(PyUcclGinContext* self, void*) {
  if (self->ctx == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Context is closed");
    return nullptr;
  }
  return PyLong_FromLong(self->ctx->num_queues());
}

PyObject* PyUcclGinContext_window_bytes(PyUcclGinContext* self, void*) {
  if (self->ctx == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Context is closed");
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong(
      static_cast<unsigned long long>(self->ctx->window_bytes()));
}

PyObject* PyUcclGinContext_max_message_bytes(PyUcclGinContext* self, void*) {
  if (self->ctx == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Context is closed");
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong(
      static_cast<unsigned long long>(self->ctx->max_message_bytes()));
}

PyObject* PyUcclGinContext_resources(PyUcclGinContext* self, PyObject*) {
  if (self->ctx == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Context is closed");
    return nullptr;
  }
  const auto& r = self->ctx->resources();
  PyObject* d = PyDict_New();
  if (d == nullptr) return nullptr;
  auto set_u64 = [&](const char* key, unsigned long long value) {
    PyObject* v = PyLong_FromUnsignedLongLong(value);
    if (v == nullptr) return -1;
    int rc = PyDict_SetItemString(d, key, v);
    Py_DECREF(v);
    return rc;
  };
  auto set_i64 = [&](const char* key, long value) {
    PyObject* v = PyLong_FromLong(value);
    if (v == nullptr) return -1;
    int rc = PyDict_SetItemString(d, key, v);
    Py_DECREF(v);
    return rc;
  };
  if (set_u64("num_queues", r.num_queues) != 0 ||
      set_u64("window_base", r.window_base) != 0 ||
      set_u64("window_bytes", r.window_bytes) != 0 ||
      set_u64("atomic_tail_base", r.atomic_tail_base) != 0 ||
      set_i64("num_scaleout_ranks", r.num_scaleout_ranks) != 0 ||
      set_i64("num_scaleup_ranks", r.num_scaleup_ranks) != 0 ||
      set_i64("scaleout_rank", r.scaleout_rank) != 0 ||
      set_i64("scaleup_rank", r.scaleup_rank) != 0 ||
      set_u64("num_lanes", r.num_lanes) != 0) {
    Py_DECREF(d);
    return nullptr;
  }
  return d;
}

PyMethodDef PyUcclGinContext_methods[] = {
    {"close", reinterpret_cast<PyCFunction>(PyUcclGinContext_close), METH_NOARGS,
     "Stop proxy threads and release registered resources."},
    {"resources", reinterpret_cast<PyCFunction>(PyUcclGinContext_resources),
     METH_NOARGS, "Return a debug dict for the device resource bundle."},
    {"put_quiet_smoke",
     reinterpret_cast<PyCFunction>(PyUcclGinContext_put_quiet_smoke), METH_VARARGS,
     "run_put_quiet_smoke(peer, bytes) -> bool. put+quiet correctness across the "
     "paired-remote peer; returns True if recv matches the peer's pattern."},
    {"put_bench", reinterpret_cast<PyCFunction>(PyUcclGinContext_put_bench),
     METH_VARARGS,
     "put_bench(peer, bytes, iters, warmup=0) -> float. Per-rank put bandwidth "
     "(GB/s) to the paired-remote peer."},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef PyUcclGinContext_getset[] = {
    {"num_queues", reinterpret_cast<getter>(PyUcclGinContext_num_queues), nullptr,
     "Number of D2H queues visible to kernels.", nullptr},
    {"window_bytes", reinterpret_cast<getter>(PyUcclGinContext_window_bytes), nullptr,
     "Registered GPU window size in bytes.", nullptr},
    {"max_message_bytes", reinterpret_cast<getter>(PyUcclGinContext_max_message_bytes),
     nullptr, "Maximum single-message payload size.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyTypeObject PyUcclGinContextType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

PyObject* mpi_rank(PyObject*, PyObject*) {
  ensure_mpi_initialized();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return PyLong_FromLong(rank);
}

PyObject* mpi_world_size(PyObject*, PyObject*) {
  ensure_mpi_initialized();
  int world = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  return PyLong_FromLong(world);
}

PyObject* mpi_finalize(PyObject*, PyObject*) {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) MPI_Finalize();
  Py_RETURN_NONE;
}

PyMethodDef module_methods[] = {
    {"mpi_rank", mpi_rank, METH_NOARGS, "Return MPI_COMM_WORLD rank."},
    {"mpi_world_size", mpi_world_size, METH_NOARGS,
     "Return MPI_COMM_WORLD size."},
    {"mpi_finalize", mpi_finalize, METH_NOARGS,
     "Finalize MPI if it has not been finalized."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_uccl_gin",
    "Standalone UCCL-GIN host context binding.",
    -1,
    module_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__uccl_gin() {
  ensure_mpi_initialized();

  PyUcclGinContextType.tp_name = "uccl_gin._uccl_gin.Context";
  PyUcclGinContextType.tp_basicsize = sizeof(PyUcclGinContext);
  PyUcclGinContextType.tp_itemsize = 0;
  PyUcclGinContextType.tp_flags = Py_TPFLAGS_DEFAULT;
  PyUcclGinContextType.tp_new = PyUcclGinContext_new;
  PyUcclGinContextType.tp_init = reinterpret_cast<initproc>(PyUcclGinContext_init);
  PyUcclGinContextType.tp_dealloc = reinterpret_cast<destructor>(PyUcclGinContext_dealloc);
  PyUcclGinContextType.tp_methods = PyUcclGinContext_methods;
  PyUcclGinContextType.tp_getset = PyUcclGinContext_getset;
  PyUcclGinContextType.tp_doc = "Standalone UCCL-GIN host context.";
  if (PyType_Ready(&PyUcclGinContextType) < 0) return nullptr;

  PyObject* m = PyModule_Create(&module_def);
  if (m == nullptr) return nullptr;
  Py_INCREF(&PyUcclGinContextType);
  if (PyModule_AddObject(m, "Context",
                         reinterpret_cast<PyObject*>(&PyUcclGinContextType)) < 0) {
    Py_DECREF(&PyUcclGinContextType);
    Py_DECREF(m);
    return nullptr;
  }
  return m;
}
