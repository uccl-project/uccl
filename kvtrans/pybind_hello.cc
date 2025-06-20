#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hello.h"

namespace py = pybind11;

PYBIND11_MODULE(hello_module, m) {
    m.doc() = "pybind11 hello world plugin";
    
    m.def("hello_world", &hello_world, 
          "A function that greets someone",
          py::arg("name"));
    
    m.def("add_numbers", &add_numbers, 
          "A function that adds two numbers",
          py::arg("a"), py::arg("b"));
} 