// #include "pinocchio/math/fwd.hpp"
// #include "pinocchio/multibody/liegroup/liegroup.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>
#include <yaml-cpp/node/iterator.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "Eigen/Core"
#include "dyno_macros.hpp"

#include <fcl/fcl.h>

#include "general_utils.hpp"
#include "math_utils.hpp"
#include "motions.hpp"
#include "robot_models.hpp"

#include "mujoco_quadrotors_payload.hpp"
#include "mujoco_quadrotor.hpp"
namespace dynobench {

std::unique_ptr<Model_robot> robot_factory(const char *file,
                                           const Eigen::VectorXd &p_lb,
                                           const Eigen::VectorXd &p_ub) {

  // std::cout << "Robot Factory: loading file: " << file << std::endl;

  if (!std::filesystem::exists(file)) {
    ERROR_WITH_INFO((std::string("file: ") + file + " not found: ").c_str());
  }

  YAML::Node node = YAML::LoadFile(file);

  assert(node["dynamics"]);
  std::string dynamics = node["dynamics"].as<std::string>();
  // std::cout << STR_(dynamics) << std::endl;

  if (dynamics == "mujocoquadspayload") {
  return std::make_unique<Model_MujocoQuadsPayload>(file, p_lb, p_ub);
  } else if (dynamics == "mujocoquad") {
  return std::make_unique<Model_MujocoQuad>(file, p_lb, p_ub);
  } else {
    ERROR_WITH_INFO("dynamics not implemented: " + dynamics);
  }
}

// std::unique_ptr<Model_robot>
// robot_factory_with_env(const std::string &robot_name,
//                        const std::string &problem_name) {

//   auto robot = robot_factory(robot_name.c_str());
//   Problem problem(problem_name);
//   load_env(*robot, problem);
//   return robot;
// }
// std::unique_ptr<Model_robot>
// joint_robot_factory(const std::vector<std::string> &robot_types,
//                     const std::string &base_path, const Eigen::VectorXd &p_lb,
//                     const Eigen::VectorXd &p_ub) {

//   std::vector<std::string> robotParams;
//   std::vector<std::shared_ptr<Model_robot>> jointRobot;
//   for (auto robot_type : robot_types) {
//     jointRobot.push_back(
//         robot_factory((base_path + robot_type + ".yaml").c_str(), p_lb, p_ub));
//   }
//   return std::make_unique<Joint_robot>(jointRobot, p_lb, p_ub);
// }
// bool check_edge_at_resolution(const Eigen::VectorXd &start,
//                               const Eigen::VectorXd &goal,
//                               std::shared_ptr<dynobench::Model_robot> &robot,
//                               double resolution) {

//   using Segment = std::pair<Eigen::VectorXd, Eigen::VectorXd>;

//   if (!robot->collision_check(start)) {
//     return false;
//   }
//   if (!robot->collision_check(goal)) {
//     return false;
//   }

//   std::queue<Segment> queue;
//   queue.push(Segment{start, goal});
//   Eigen::VectorXd x(robot->nx);

//   while (!queue.empty()) {

//     auto [si, gi] = queue.front();
//     queue.pop();

//     if (robot->distance(si, gi) > resolution) {
//       // check mid points
//       robot->interpolate(x, si, gi, 0.5);

//       if (robot->collision_check(x)) {
//         // collision free.
//         queue.push({si, x});
//         queue.push({x, gi});
//       } else {
//         // collision!
//         return false;
//       }

//       ;
//     }
//   }
//   return true;
// }

} // namespace dynobench
