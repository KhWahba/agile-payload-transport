#pragma once

#include <memory>
#include <fstream>
#include <string>
#include <vector>

#include <yaml-cpp/node/node.h>

#include "policy_onnx.hpp"
#include "croco_models.hpp"
#include "motions.hpp"
#include "options.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"

namespace dynoplan {

enum class NmpcMode {
  TrackGoal,
  TrackReferenceNmpcStandard,
  TrackReferenceNmpcRefWarm,
  TrackReferencePolicy,
  TrackLinearHover,
};

struct NmpcStepInfo {
  bool did_solve = false;
  bool reached_goal = false;
  bool failed = false;
  double goal_distance = 0.0;
  double solve_time_ms = 0.0;
  double step_time_ms = 0.0;
};

class NmpcController {
 public:
  // Construct a stateful NMPC controller from problem/config yaml files.
  NmpcController(const std::string &prob_file, const std::string &cfg_file);

  // Run closed-loop NMPC for max_steps_ and store stitched trajectory in solution().
  void run();
  // Convenience wrapper for Python/file-based workflows.
  void run_with_overrides(const std::string &mode,
                          const std::string &out_yaml,
                          const std::string &out_timing_json);
  // Optional visualization pass based on prob_file rendering settings.
  void maybe_visualize();

  const dynobench::Trajectory &solution() const { return sol_; }
  const std::string &results_path() const { return results_path_; }

 private:
  // Parse yaml files, initialize model/problem objects and runtime options.
  void setup_from_files(const std::string &prob_file, const std::string &cfg_file);
  // Prepare warm-start/reference windows for the current mode at step k.
  void prepare_window_for_step(int k);
  // Execute one simulation step (solve-if-needed + apply first/open-loop control).
  NmpcStepInfo step_once(int k);
  // Inject optional payload disturbance at configured times.
  void apply_payload_disturbance(Eigen::VectorXd &xnext, int k) const;
  // Build NMPC Crocoddyl structure once. No per-step structural rebuilds.
  void build_problem_once();
  // Build warm-start seed shared by non-policy modes.
  void prepare_common_warm_start(int k, bool allow_init_bootstrap);
  // Fill warm_start_N_.states by rolling model from current measured state.
  void rollout_warm_start_states_from_actions();
  // Build policy learner-features vector (must match scripts/payload_env.py).
  Eigen::VectorXd build_policy_features(const Eigen::VectorXd &state) const;
  void write_policy_debug_step(int k, bool did_solve, const Eigen::VectorXd &x_before,
                               const Eigen::VectorXd &u_applied, const Eigen::VectorXd &x_after,
                               double goal_distance);
  // Update per-step references/weights and x0 in the already built problem.
  void update_problem_references();
  // Solve using cached solver + cached problem with updated warm-start.
  void solve_with_cached_problem();

  static void ensure_horizon(dynobench::Trajectory &traj, std::size_t N,
                             std::size_t nx, std::size_t nu);
  static dynobench::Trajectory shift_and_pad(const dynobench::Trajectory &solved_window,
                                             std::size_t N, std::size_t nx, std::size_t nu);
  static void slice_window(dynobench::Trajectory &w, const dynobench::Trajectory &traj,
                           std::size_t offset, std::size_t N,
                           Eigen::VectorXd &u_hover, Eigen::VectorXd &goal);
  static NmpcMode parse_mode(const std::string &mode);

  // Parsed from input yaml.
  bool do_optimize_ = false;
  bool do_visualize_ = false;
  bool view_ref_ = false;
  int repeats_ = 1;
  std::vector<std::string> views_{"auto"};
  std::string env_file_;
  std::string init_file_;
  std::string ref_file_;
  std::string models_dir_;
  std::string models_dir_abs_;
  std::string results_path_;
  std::string video_prefix_;
  std::string timing_output_path_override_;
  std::string last_timing_output_path_;

  bool use_policy_onnx_ = false;
  std::string policy_onnx_path_;
  bool policy_rollout_as_ref_ = true;
  double policy_u_clip_min_ = -1e30;
  double policy_u_clip_max_ = 1e30;
  int policy_threads_ = 1;
  double planner_act_low_ = 0.0;
  double planner_act_high_ = 1.4;
  bool debug_policy_loop_ = false;
  std::string debug_policy_path_;
  double control_noise_ = 1e-3;
  double fail_threshold_ = 5.0;
  double goal_tol_ = 0.05;
  std::size_t N_ = 0;
  std::size_t max_steps_ = 0;

  Options_trajopt options_trajopt_;
  NmpcMode mode_ = NmpcMode::TrackGoal;
  std::size_t solve_every_k_steps_ = 1;
  bool track_reference_active_ = false;

  // Disturbance params.
  bool disturbance_enable_ = false;
  double disturbance_start_s_ = 0.0;
  double disturbance_duration_s_ = 0.0;
  Eigen::Vector3d disturbance_force_ = Eigen::Vector3d::Zero();
  double disturbance_payload_mass_ = 1.0;

  // Runtime state.
  std::unique_ptr<dynobench::Problem> problem_;
  std::shared_ptr<dynobench::Model_robot> robot_;
  ptr<Dynamics> dynamics_;
  ptr<crocoddyl::ShootingProblem> shooting_problem_;
  std::unique_ptr<crocoddyl::SolverBoxFDDP> ddp_solver_;
  std::vector<Eigen::VectorXd> xs_ws_;
  std::vector<Eigen::VectorXd> us_ws_;
  std::vector<ptr<State_cost>> run_state_costs_;
  std::vector<ptr<Control_cost>> run_control_costs_;
  ptr<State_cost> terminal_goal_cost_;
  bool problem_built_ = false;
  std::unique_ptr<PolicyOnnx> policy_onnx_;
  dynobench::Trajectory init_guess_;
  dynobench::Trajectory ref_traj_;
  dynobench::Trajectory warm_start_N_;
  dynobench::Trajectory ref_traj_N_;
  dynobench::Trajectory sol_window_;
  dynobench::Trajectory sol_broken_;
  dynobench::Trajectory sol_;
  dynobench::Trajectory last_solved_window_;

  Eigen::VectorXd u_prev_exec_;
  Eigen::VectorXd prev_action_policy_;
  Eigen::VectorXd act_mid_;
  Eigen::VectorXd act_half_;
  Eigen::VectorXd last_policy_features_;
  Eigen::VectorXd last_policy_chunk_raw_;
  std::size_t last_policy_horizon_ = 0;
  Eigen::VectorXd x_init_;
  int k_goal_ = 0;
  std::size_t nx_ = 0;
  std::size_t nu_ = 0;
  std::size_t n_bodies_ = 0;
  std::size_t n_quads_ = 0;
  std::size_t planned_window_idx_ = 0;
  bool have_valid_window_ = false;
  bool init_bootstrap_used_ = false;
  bool warm_start_is_feasible_ = false;  // true when ws states come from rollout
  std::ofstream debug_policy_stream_;
};

}  // namespace dynoplan
