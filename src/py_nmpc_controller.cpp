#include <pybind11/pybind11.h>

#include <string>

#include "nmpc_controller.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nmpc_controller_py, m) {
  m.doc() = "Python bindings for NMPC controller";

  py::class_<dynoplan::NmpcController>(m, "Controller")
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("prob_path"), py::arg("cfg_path"))
      .def("run", [](dynoplan::NmpcController &self,
                     const std::string &mode,
                     const std::string &out_yaml,
                     const std::string &out_timing_json,
                     bool visualize) {
             self.run_with_overrides(mode, out_yaml, out_timing_json);
             if (visualize) {
               self.maybe_visualize();
             }
           },
           py::arg("mode") = "",
           py::arg("out_yaml") = "",
           py::arg("out_timing_json") = "",
           py::arg("visualize") = false)
      .def("maybe_visualize", &dynoplan::NmpcController::maybe_visualize);
}
