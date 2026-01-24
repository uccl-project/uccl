#include "operator.h"
#include "scheduler.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // vector
#include <torch/extension.h>

namespace py = pybind11;
namespace ur = UKernel::Runtime;

void bind_reduce_kind(py::module_& m) {
  py::enum_<ur::ReduceKind>(m, "ReduceKind")
      .value("Sum", ur::ReduceKind::Sum)
      .value("Sub", ur::ReduceKind::Sub)
      .value("Avg", ur::ReduceKind::Avg)
      .value("Max", ur::ReduceKind::Max)
      .value("Min", ur::ReduceKind::Min)
      .export_values();
}

void bind_parallel_rule(py::module_& m) {
  py::class_<ur::ParallelRule>(m, "ParallelRule")
      .def(py::init<int, int>(), py::arg("num_tasks"),
           py::arg("tiles_per_task"))
      .def_readwrite("num_tasks", &ur::ParallelRule::num_tasks)
      .def_readwrite("tiles_per_task", &ur::ParallelRule::tiles_per_task);
}

void bind_operator(py::module_& m) {
  py::class_<ur::Operator>(m, "Operator")
      .def_property_readonly("id",
                             [](ur::Operator const& op) { return op.id; });
}

void bind_factory(py::module_& m) {
  // P2P
  m.def("p2p_send", &ur::OperatorFactory::P2PSend, py::arg("id"),
        py::arg("src"), py::arg("dst"), py::arg("parallel_rule"),
        py::arg("deps") = std::vector<uint64_t>{});

  m.def("p2p_recv", &ur::OperatorFactory::P2PRecv, py::arg("id"),
        py::arg("src"), py::arg("dst"), py::arg("parallel_rule"),
        py::arg("deps") = std::vector<uint64_t>{});

  // Collective
  m.def("all_reduce", &ur::OperatorFactory::AllReduce, py::arg("id"),
        py::arg("src"), py::arg("dst"), py::arg("reduce_kind"),
        py::arg("parallel_rule"), py::arg("deps") = std::vector<uint64_t>{});

  m.def("all_to_all", &ur::OperatorFactory::AllToAll, py::arg("id"),
        py::arg("src"), py::arg("dst"), py::arg("parallel_rule"),
        py::arg("deps") = std::vector<uint64_t>{});

  // Compute
  m.def("gemm", &ur::OperatorFactory::Gemm, py::arg("id"), py::arg("src"),
        py::arg("dst"), py::arg("parallel_rule"),
        py::arg("deps") = std::vector<uint64_t>{});

  // MoE
  m.def("moe_routing", &ur::OperatorFactory::MoeRouting, py::arg("id"),
        py::arg("src"), py::arg("dst"), py::arg("parallel_rule"),
        py::arg("deps") = std::vector<uint64_t>{});

  m.def("moe_expert_gemm", &ur::OperatorFactory::MoeExpertGemm, py::arg("id"),
        py::arg("src"), py::arg("dst"), py::arg("parallel_rule"),
        py::arg("deps") = std::vector<uint64_t>{});

  m.def("moe_combine", &ur::OperatorFactory::MoeCombine, py::arg("id"),
        py::arg("src"), py::arg("dst"), py::arg("parallel_rule"),
        py::arg("deps") = std::vector<uint64_t>{});
}

void run_ops(std::vector<ur::Operator> const& ops) {
  ur::Scheduler sched;
  for (auto const& op : ops) {
    sched.add_operator(op);
  }
  sched.run();
}

PYBIND11_MODULE(ukernel, m) {
  m.doc() = "UKernel runtime (scheduler + operators)";

  bind_reduce_kind(m);
  bind_parallel_rule(m);
  bind_operator(m);
  bind_factory(m);

  m.def("run", &run_ops, "Run a list of operators");
}