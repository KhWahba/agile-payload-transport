#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <chrono>

#include <boost/program_options.hpp>
#include "opt_simulate_mujoco.hpp"
#include "general_utils.hpp"
#include "ocp.hpp"
#include "mujoco_quadrotors_payload.hpp"

#include "bc_policy_onnx.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;

using namespace dynobench;
using namespace dynoplan;

static inline void ensure_horizon(Trajectory& traj, std::size_t N,
                                  std::size_t nx, std::size_t nu)
{
  // Truncate first
  if (traj.actions.size() > N)   traj.actions.resize(N);
  if (traj.states.size()  > N+1) traj.states.resize(N+1);

  // Pad actions to N
  while (traj.actions.size() < N) {
    Eigen::VectorXd u = traj.actions.empty()
                        ? Eigen::VectorXd::Zero(nu)
                         : Eigen::VectorXd(traj.actions.back()); // force a plain copy
    if (u.size() != (Eigen::Index)nu) u.setZero(nu); // safety
    traj.actions.push_back(u);
  }

  // Pad states to N+1
  while (traj.states.size() < N+1) {
    Eigen::VectorXd x = traj.states.empty()
                        ? Eigen::VectorXd::Zero(nx)
                        : Eigen::VectorXd(traj.states.back()); // force a plain copy
    if (x.size() != (Eigen::Index)nx) x.setZero(nx); // safety
    traj.states.push_back(x);
  }
}

static inline Trajectory shift_and_pad(const Trajectory &solved_window,
                                       std::size_t N,
                                       std::size_t nx,
                                       std::size_t nu)
{
  Trajectory warm;

  // If window is empty, just pad
  if (solved_window.actions.empty() || solved_window.states.size() < 2) {
    warm = solved_window; // copy whatever there is
    ensure_horizon(warm, N, nx, nu);
    return warm;
  }

  // Drop the first knot and copy the rest (deep copies)
  warm.actions.reserve(N);
  for (std::size_t i = 1; i < solved_window.actions.size(); ++i) {
    Eigen::VectorXd u = solved_window.actions[i];
    warm.actions.push_back(u);
  }

  warm.states.reserve(N + 1);
  for (std::size_t i = 1; i < solved_window.states.size(); ++i) {
    Eigen::VectorXd x = solved_window.states[i];
    warm.states.push_back(x);
  }

  // Pad to full horizon
  ensure_horizon(warm, N, nx, nu);
  return warm;
}

static inline void slice_window(Trajectory& w, const Trajectory& traj, std::size_t offset, std::size_t N, Eigen::VectorXd& u_hover, Eigen::VectorXd& goal)
{
  w.actions.clear();
  w.states.clear();
  if (offset < traj.actions.size()) {
    auto a0 = traj.actions.begin() + offset;
    auto a1 = (offset + N <= traj.actions.size()) ? a0 + N : traj.actions.end();
    w.actions.assign(a0, a1);
  } 
  if (w.actions.size() < N) {
    w.actions.resize(N, u_hover);
  } else if (w.actions.size() > N) {
    w.actions.resize(N);
  }
  if (offset < traj.states.size()) {
    auto s0 = traj.states.begin() + offset;
    auto s1 = (offset + N + 1 <= traj.states.size()) ? s0 + (N + 1) : traj.states.end();
    w.states.assign(s0, s1);
  } 
  // pad states to N+1 with goal
  if (w.states.size() < N + 1) {
    w.states.resize(N + 1, goal);
  } else if (w.states.size() > N + 1) {
    w.states.resize(N + 1);
  }
}


int main(int argc, char **argv)
{
  // --- CLI ---
  std::string prob_file, cfg_file;
  // std::string env_file, init_file, ref_file, models_base, results_path, cfg_file;

  po::options_description desc("main_mujoco_opt_simulate options");
  desc.add_options()("help,h", "Show help")
  ("cfg_file", po::value<std::string>(&cfg_file)->default_value(""), "optimization parameters, see optimization/options.hpp")
  ("prob_file", po::value<std::string>(&prob_file)->default_value(""), "optimization parameters, see optimization/options.hpp");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 0;
  }
  if (prob_file.empty()) throw std::runtime_error("Missing --prob_file");
  if (cfg_file.empty())  throw std::runtime_error("Missing --cfg_file");

  YAML::Node problem_file = YAML::LoadFile(prob_file);
  YAML::Node opt_file     = YAML::LoadFile(cfg_file);

  bool do_optimize               = problem_file["optimize"] ? problem_file["optimize"].as<bool>() : false;
  bool do_visualize              = problem_file["visualize"] ? problem_file["visualize"].as<bool>() : false;
  std::string env_file           = problem_file["env_file"] ? problem_file["env_file"].as<std::string>() : "";
  std::string init_file          = problem_file["init_file"] ? problem_file["init_file"].as<std::string>() : "";
  std::string ref_file           = problem_file["ref_file"] ? problem_file["ref_file"].as<std::string>() : "";
  std::string models_dir         = problem_file["models_dir"] ? problem_file["models_dir"].as<std::string>() : "";
  std::string results_path       = problem_file["results_path"] ? problem_file["results_path"].as<std::string>() : "../result_opt.yaml";  
  bool view_ref                  = problem_file["view_ref"] ? problem_file["view_ref"].as<bool>() : false;
  std::string base               = problem_file["video_prefix"] ? problem_file["video_prefix"].as<std::string>() : "";
  int repeats                    = problem_file["repeats"] ? problem_file["repeats"].as<int>() : 1;
  std::vector<std::string> views = problem_file["views"] ? problem_file["views"].as<std::vector<std::string>>() : std::vector<std::string>{"auto"};

  bool use_bc_policy       = opt_file["use_bc_policy"] ? opt_file["use_bc_policy"].as<bool>() : false;
  std::string bc_onnx_path = opt_file["bc_onnx_path"] ? opt_file["bc_onnx_path"].as<std::string>() : "";
  bool bc_rollout_as_ref   = opt_file["bc_rollout_as_ref"] ? opt_file["bc_rollout_as_ref"].as<bool>() : true;
  double bc_u_clip_min     = opt_file["bc_u_clip_min"] ? opt_file["bc_u_clip_min"].as<double>() : -1e30;
  double bc_u_clip_max     = opt_file["bc_u_clip_max"] ? opt_file["bc_u_clip_max"].as<double>() :  1e30;
  int bc_threads           = opt_file["bc_threads"] ? opt_file["bc_threads"].as<int>() : 1;
  double control_noise     = opt_file["control_noise"] ? opt_file["control_noise"].as<double>() : 1e-3;
  double fail_threshold    = opt_file["fail_threshold"] ? opt_file["fail_threshold"].as<double>() : 5.0;
  double goal_tol          = opt_file["goal_tol"]       ? opt_file["goal_tol"].as<double>() : 0.05;
  size_t N                 = static_cast<std::size_t>(opt_file["N"].as<int>()); // horizon
  size_t max_steps         = static_cast<std::size_t>(opt_file["max_steps"].as<int>());
  
  Options_trajopt options_trajopt;
  options_trajopt.read_from_yaml(opt_file);
  // If neither optimize nor visualize requested, do nothing (as specified)
  if (!do_optimize && !do_visualize) {
    std::cout << "Nothing to do: neither -o/--optimize nor -v/--visualize specified.\n";
    std::cout << "Run with --help to see options.\n";
    return 0;
  }

  // Validate required inputs when actions are requested
  auto need = [&](const char *name, const std::string &val)
  {
    if (val.empty())
    {
      throw std::runtime_error(std::string("Missing required option --") + name);
    }
  };

  // initialize problem AND robot model
  fs::path build_dir = fs::current_path(); // e.g., /path/to/agile-payload-transport/build
  std::string models_dir_path = build_dir.parent_path().string() + "/"+ models_dir +"/";
  Problem problem(env_file.c_str());
  std::string models_dir_abs = fs::absolute(models_dir_path).string();
  problem.models_base_path = models_dir_abs;
  dynoplan::Result_opti result;
  std::shared_ptr<dynobench::Model_robot> robot;
  robot = dynobench::robot_factory((models_dir_abs + problem.robotType + ".yaml").c_str(),
                                   problem.p_lb, problem.p_ub);

  if (!robot) {
    throw std::runtime_error("Failed to create robot model for type: " + problem.robotType);
  }
  load_env(*robot, problem);
  // known sizes from the robot model
  const std::size_t nx = static_cast<std::size_t>(robot->nx);
  const std::size_t nu = static_cast<std::size_t>(robot->nu);

  // initialize BC policy if needed
  std::unique_ptr<BCPolicyOnnx> bc_policy;
  Eigen::VectorXd u_prev_exec = Eigen::VectorXd::Zero(nu);   // previous executed control (for BC warm start)
  Trajectory init_guess, warm_start_N, ref_traj, ref_traj_N; // initialize full and windowed trajectories

  if (use_bc_policy) {
    if (bc_onnx_path.empty()) {
      throw std::runtime_error("use_bc_policy=true but bc_onnx_path is empty");
    }
    bc_policy = std::make_unique<BCPolicyOnnx>(bc_onnx_path, bc_threads);
    std::cout << "[BC] Enabled. Loaded ONNX policy from: " << bc_onnx_path << std::endl;
  } else {  
    // NO POLICY: load init and ref trajectories from files
    // initilize guess and solution trajectories
    if (!init_file.empty()) {
      init_guess.read_from_yaml(init_file.c_str());
    } else {
      std::cout << "No init_file specified.\n";
      warm_start_N.num_time_steps = N;
    }
    if (!ref_file.empty()) {
      ref_traj.read_from_yaml(ref_file.c_str());
    } else {
      options_trajopt.track_reference = false;
      options_trajopt.track_goal = true;
      ref_traj_N.num_time_steps = N;
      ref_traj_N.states.resize(N + 1);
      std::for_each(ref_traj_N.states.begin(), ref_traj_N.states.end(),
                    [&](auto &x) {
                        x = problem.goal;
                    });
      std::cout << "No ref_file specified.\n";
    }
  }

  // main NMPC loop
  Trajectory sol, sol_window, sol_broken; 
  sol.states.clear();
  sol.actions.clear();
  sol.states.push_back(problem.start);

  Eigen::VectorXd x_init = problem.start;
  int k_goal = 0; //  log step when goal was reached

  for (int k = 0; k < max_steps; ++k) {
    std::cout << "=== NMPC Step " << k << " ===\n";    
    if (use_bc_policy && bc_policy) {
      bc_policy->predict_rollout(
          N,
          robot.get(),
          x_init,
          u_prev_exec,
          &warm_start_N.states,
          &warm_start_N.actions,
          bc_u_clip_min,
          bc_u_clip_max
      );
      // use rollout as reference too 
      ref_traj_N = warm_start_N;
      if (k == 0) { 
        std::cout << "[BC] warm_start_N states=" << warm_start_N.states.size()
                  << " actions=" << warm_start_N.actions.size() << std::endl;
        std::cout << "[BC] first u=" << warm_start_N.actions.front().transpose() << std::endl;
      }
    } else {
      if (!init_file.empty()) {
        const std::size_t start_idx = std::min<std::size_t>(k, init_guess.actions.size() ? init_guess.actions.size()-1 : 0);
        slice_window(warm_start_N, init_guess, k, N, robot->u_0, problem.goal);
      } else {
        // TODO: debug shift and pad
        warm_start_N = shift_and_pad(sol, N, nx, nu);
        ensure_horizon(warm_start_N, N, nx, nu);
      }
      if (!ref_file.empty()) {
        std::size_t max_k_u = (ref_traj.actions.size() > 0) ? (ref_traj.actions.size() - 1) : 0;
        std::size_t start_idx = std::min<std::size_t>(k, max_k_u);
        std::cout << "Slicing reference trajectory from index " << start_idx << ", N: " << N << "\n";
        slice_window(ref_traj_N, ref_traj, k, N, robot->u_0, problem.goal);
      } 
    }
      
    problem.start = x_init;
    execute_nmpc_mujoco(problem, warm_start_N, ref_traj_N, sol_window, sol_broken, options_trajopt);    
    Eigen::VectorXd u = sol_window.actions.front();
    Eigen::VectorXd xnext(nx);
    u += control_noise * Eigen::VectorXd::Random(nu);
    // propagate one step in the real robot
    robot->step(xnext, x_init.head(nx), u.head(nu), robot->ref_dt);
    // log results
    sol.states.push_back(xnext);
    sol.actions.push_back(u);
    // log control for next BC warm start (if used)
    u_prev_exec = u;
    // prepare for next iteration
    x_init = xnext;
    // compute distance to goal
    if (robot->distance(sol.states.back(), problem.goal) <= goal_tol) {
      if (k_goal == 0 && k != 0) {
        k_goal = k;
        std::cout << "Goal reached at step " << k << " with distance " << robot->distance(sol.states.back(), problem.goal) << "\n";
        // break;
      }
      else if (k == 0 && k_goal == 0) {
        k_goal = 0;
        std::cout << "Goal reached at step 0\n";
      }
    } else if (robot->distance(sol.states.back(), problem.goal) > fail_threshold) {
      double dist = robot->distance(sol.states.back(), problem.goal);
      std::cout << "Tracking failed: distance (" << dist << ") exceeded threshold (" << fail_threshold << ") at step " << k << "\n";
      break;
    } else {
      std::cout << "Goal distance: " << robot->distance(sol.states.back(), problem.goal) << std::endl;
    }  
  }

  if (max_steps < 2) {
    // This is for debugging the optimization on the full problem
    sol = sol_window; // if only one step, show ghost of init
  }
  std::cout << "states size: " << sol.states.size() << std::endl;
  std::cout << "actions size: " << sol.actions.size() << std::endl;
  
  Problem problem_final(env_file.c_str());
  sol.cost = sol.actions.size() * robot->ref_dt; // time cost
  sol.start = problem_final.start;
  sol.goal = problem_final.goal;
  sol.check(robot, true);
  
  if (!results_path.empty()) {
    sol.to_yaml_format(results_path.c_str());
    std::cout << "Saved NMPC stitched trajectory to: " << results_path << "\n";
  } else {
    std::cout << "No results_path specified, not saving trajectory.\n";
    return 0;
  }

  // --- VISUALIZATION ---
  if (!do_visualize) {
    return 0;
  }

  if (base.empty()) {
    fs::path init_p(init_file);
    base = (init_p.parent_path() / (init_p.stem().string() + "_viz")).string();
  }
  bool view_ghost = view_ref && !ref_file.empty();
  
  Trajectory empty_traj;
  // Remove .yaml, add random number, then append .yaml
  std::string tmp_ref_padded_file = init_file;

  if (!tmp_ref_padded_file.empty()) {
    size_t pos = tmp_ref_padded_file.rfind(".yaml");
    if (pos != std::string::npos) {
      tmp_ref_padded_file = tmp_ref_padded_file.substr(0, pos);
    }
    // Generate a random number
    int rand_num = std::rand();
    tmp_ref_padded_file += std::to_string(rand_num) + ".yaml";
    empty_traj.read_from_yaml(init_file.c_str());
  }
  empty_traj.start = problem.start;
  empty_traj.goal = problem.goal;

  // Pad empty_traj states and actions to match sol
  if (!empty_traj.states.empty() && !sol.states.empty()) {
    while (empty_traj.states.size() < sol.states.size()) {
      empty_traj.states.push_back(empty_traj.states.back());
    }
  }
  if (!empty_traj.actions.empty() && !sol.actions.empty()) {
    while (empty_traj.actions.size() < sol.actions.size()) {
      empty_traj.actions.push_back(empty_traj.actions.back());
    }
  }
  std::cout << "Writing temporary empty trajectory to: " << tmp_ref_padded_file << "\n";
  empty_traj.to_yaml_format(tmp_ref_padded_file.c_str());
  if (views.size() == 1 && views[0] == "auto") {
    std::string video_path = base + ".mp4"; // AUTO strips .mp4 and appends suffixes
    std::cout << "Writing videos to base: " << base << "_{side,top,front,diag}.mp4 (repeats=" << repeats << ")\n";
    execute_simMujoco(env_file, tmp_ref_padded_file, sol, models_dir_abs,
                      video_path, "auto", repeats, view_ghost, true /*feasible*/);
  } else {
    for (const auto &v : views) {
      std::string out = base + "_" + v + ".mp4";
      std::cout << "Writing: " << out << " (repeats=" << repeats << ")\n";
      execute_simMujoco(env_file, tmp_ref_padded_file, sol, models_dir_abs,
                        out, v, repeats, view_ghost, true /*feasible*/);
    }
  }
  std::cout << "Done.\n";
  return 0;
}
