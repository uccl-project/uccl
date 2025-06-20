#include "hello.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(hello_module, m) {
  m.doc() = "pybind11 object-oriented hello world plugin";

  py::class_<Greeter>(m, "Greeter")
      .def(py::init<std::string const&>(),
           "Create a new Greeter with optional default greeting",
           py::arg("default_greeting") = "Hello")
      .def("greet", &Greeter::greet,
           "Greet someone with the default or custom greeting", py::arg("name"),
           py::arg("custom_greeting") = "")
      .def("add_numbers", &Greeter::add_numbers, "Add two numbers",
           py::arg("a"), py::arg("b"))
      .def("multiply_numbers", &Greeter::multiply_numbers,
           "Multiply two numbers", py::arg("a"), py::arg("b"))
      .def("get_greeting_count", &Greeter::get_greeting_count,
           "Get the number of greetings performed")
      .def("get_greeting_history", &Greeter::get_greeting_history,
           "Get the list of all greetings performed",
           py::return_value_policy::reference_internal)
      .def("set_default_greeting", &Greeter::set_default_greeting,
           "Set a new default greeting", py::arg("new_greeting"))
      .def("get_default_greeting", &Greeter::get_default_greeting,
           "Get the current default greeting",
           py::return_value_policy::reference_internal)
      .def("clear_history", &Greeter::clear_history,
           "Clear the greeting history")
      .def("get_summary", &Greeter::get_summary,
           "Get a summary of the greeter's state")
      .def("__repr__", [](Greeter const& g) {
        return "<Greeter with default greeting: '" + g.get_default_greeting() +
               "', count: " + std::to_string(g.get_greeting_count()) + ">";
      });
}