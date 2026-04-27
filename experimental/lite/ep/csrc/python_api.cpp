#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>

#include "jit/api.hpp"
#include "elastic/buffer.hpp"
#include "utils/event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    // Whether support FP8 and TMA features
    m.def("is_sm90_compiled", []() { return deep_ep::kEnableSM90Features; });

    // The integer type of top-k indices
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value);

    // JIT API
    deep_ep::jit::register_apis(m);

    // Register event handle used by ElasticBuffer APIs
    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    // Register elastic buffer (DeepEP V2) APIs
    deep_ep::elastic::register_apis(m);
}
