#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/program_options.hpp>

#include "nmpc_controller.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv) {
  try {
    std::string prob_file;
    std::string cfg_file;

    po::options_description desc("nmpc_mujoco options");
    desc.add_options()
      ("help,h", "Show help")
      ("cfg_file", po::value<std::string>(&cfg_file)->default_value(""), "optimization yaml")
      ("prob_file", po::value<std::string>(&prob_file)->default_value(""), "problem yaml");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    if (prob_file.empty()) {
      throw std::runtime_error("Missing --prob_file");
    }
    if (cfg_file.empty()) {
      throw std::runtime_error("Missing --cfg_file");
    }

    dynoplan::NmpcController controller(prob_file, cfg_file);
    controller.run();
    controller.maybe_visualize();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[nmpc_mujoco] ERROR: " << e.what() << std::endl;
    return 1;
  }
}
