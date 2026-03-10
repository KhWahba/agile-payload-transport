#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <yaml-cpp/node/node.h>

namespace dynoplan {
namespace po = boost::program_options;

struct Options_trajopt {

  // Core NMPC/DDP settings
  bool states_reg = false;
  double th_stop = 1e-2;
  double init_reg = 1e2;
  double th_acceptnegstep = .3;
  size_t max_iter = 50;
  double u_bound_scale = 1;

  double weight_goal = 1000.;
  double collision_weight = 350.;

  // Legacy shared control-tracking weight (kept for compatibility).
  double policy_control_tracking_weight = 20.0;

  // Running state tracking weight for reference-tracking modes.
  double ref_state_tracking_weight = 100.0;
  // Running control tracking weight when reference comes from planner.
  double planner_ref_control_tracking_weight = 20.0;
  // Running control tracking weight when reference comes from policy rollout.
  double policy_ref_control_tracking_weight = 20.0;
  // Running control regularization in track_goal mode (u_hover reference).
  double goal_control_regularization_weight = 50.0;

  // When true, multiply running state tracking weight by goal_weight so that
  // dimensions irrelevant to the goal (payload quaternion, payload angular
  // velocity) get zero running cost.  This matches standard quadrotor NMPC
  // practice where only physically relevant dimensions are penalised in the
  // running cost.  Default false for backward compatibility.
  bool running_cost_goal_weight_mask = false;

  // Solve cadence for NMPC. 1 means solve every step.
  size_t solve_every_k_steps = 1;

  // Runtime mode for controller:
  // - track_goal
  // - track_reference_nmpc_standard
  // - track_reference_nmpc_refwarm
  // - track_reference_policy
  // - track_policy_warmstart_goal
  // - track_linear_hover (TODO behavior)
  std::string nmpc_mode = "track_goal";

  // Payload disturbance (impulse-like force profile over a duration).
  bool disturbance_enable = false;
  double disturbance_start_s = 0.0;
  double disturbance_duration_s = 0.0;
  double disturbance_force_x = 0.0;
  double disturbance_force_y = 0.0;
  double disturbance_force_z = 0.0;
  double disturbance_payload_mass = 1.0;

  void add_options(po::options_description &desc);

  void __read_from_node(const YAML::Node &node);

  void print(std::ostream &out, const std::string &be = "",
             const std::string &af = ": ") const;

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);
};

void PrintVariableMap(const boost::program_options::variables_map &vm,
                      std::ostream &out);

// TODO: check which methods are deprecated!!
enum class SOLVER {
  traj_opt = 0,
};

} // namespace dynoplan
