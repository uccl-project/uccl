#pragma once

#include "compiler.hpp"
#include "include_parser.hpp"
#include "kernel_runtime.hpp"

namespace deep_ep::jit {

static void init(std::string const& library_root_path,
                 std::string const& cuda_home_path_by_python,
                 std::string const& nccl_root_path_by_python) {
  Compiler::prepare_init(library_root_path, cuda_home_path_by_python,
                         nccl_root_path_by_python);
  KernelRuntime::prepare_init(cuda_home_path_by_python);
  IncludeParser::prepare_init(library_root_path);
}

static void register_apis(pybind11::module_& m) { m.def("init_jit", &init); }

}  // namespace deep_ep::jit
