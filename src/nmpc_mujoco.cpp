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

namespace po = boost::program_options;
namespace fs = std::filesystem;

using namespace dynobench;

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
                        :  Eigen::VectorXd::Zero(nu);
                        //  : Eigen::VectorXd(traj.actions.back()); // force a plain copy
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

static inline Trajectory slice_window(const Trajectory& traj,
                                      std::size_t offset, std::size_t N)
{
  Trajectory w;

  if (offset < traj.actions.size()) {
    auto a0 = traj.actions.begin() + offset;
    auto a1 = (offset + N <= traj.actions.size()) ? a0 + N : traj.actions.end();
    w.actions.assign(a0, a1); // copies into plain Eigen::VectorXd
  }
  if (offset < traj.states.size()) {
    auto s0 = traj.states.begin() + offset;
    auto s1 = (offset + N + 1 <= traj.states.size()) ? s0 + (N + 1) : traj.states.end();
    w.states.assign(s0, s1);
  }
  return w;
}

// Append one applied (x1, u0) to the stitched trajectory
static inline void append_step(Trajectory &stitched,
                               const Eigen::VectorXd &x1,
                               const Eigen::VectorXd &u0)
{
  stitched.actions.push_back(u0);
  stitched.states.push_back(x1);
}

int main(int argc, char **argv)
{
  // --- CLI ---
  std::string prob_file, cfg_file;
  // std::string env_file, init_file, ref_file, dynobench_base, results_path, cfg_file;

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

  YAML::Node problem_file = YAML::LoadFile(prob_file);
  bool do_optimize = problem_file["optimize"] ? problem_file["optimize"].as<bool>() : false;
  bool do_visualize = problem_file["visualize"] ? problem_file["visualize"].as<bool>() : false;
  std::string env_file = problem_file["env_file"] ? problem_file["env_file"].as<std::string>() : "";
  std::string init_file = problem_file["init_file"] ? problem_file["init_file"].as<std::string>() : "";
  std::string ref_file = problem_file["ref_file"] ? problem_file["ref_file"].as<std::string>() : "";
  std::string dynobench_base = problem_file["dynobench_base"] ? problem_file["dynobench_base"].as<std::string>() : "";
  std::string results_path = problem_file["results_path"] ? problem_file["results_path"].as<std::string>() : "../result_opt.yaml";  
  bool view_ref = problem_file["view_ref"] ? problem_file["view_ref"].as<bool>() : false;
  std::string base = problem_file["video_prefix"] ? problem_file["video_prefix"].as<std::string>() : "";
  int repeats = problem_file["repeats"] ? problem_file["repeats"].as<int>() : 1;
  std::vector<std::string> views = problem_file["views"] ? problem_file["views"].as<std::vector<std::string>>() : std::vector<std::string>{"auto"};


  // If neither optimize nor visualize requested, do nothing (as specified)
  if (!do_optimize && !do_visualize)
  {
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
  need("env_file", env_file);
  need("dynobench_base", dynobench_base);

  // Normalize paths
  auto as_abs = [](const std::string &p) -> std::string
  {
    try
    {
      return fs::weakly_canonical(fs::path(p)).string();
    }
    catch (...)
    {
      return p;
    }
  };
  env_file = as_abs(env_file);
  init_file = as_abs(init_file);
  dynobench_base = as_abs(dynobench_base);

  YAML::Node opt_file = YAML::LoadFile(cfg_file);
  std::size_t N = static_cast<std::size_t>(opt_file["N"].as<int>()); // horizon
  std::size_t max_steps = static_cast<std::size_t>(opt_file["max_steps"].as<int>());

  // initialize problem
  std::string models_base_path = dynobench_base + "/models/";
  Problem problem(env_file.c_str());
  problem.models_base_path = models_base_path;
  dynoplan::Result_opti result;
  std::shared_ptr<dynobench::Model_robot> robot;
  robot = dynobench::robot_factory((models_base_path + problem.robotType + ".yaml").c_str(),
                                   problem.p_lb, problem.p_ub);

  if (!robot)
  {
    throw std::runtime_error("Failed to create robot model for type: " + problem.robotType);
  }


  if (opt_file["init_file"])
  {
    init_file = opt_file["init_file"].as<std::string>();
  }

  // initilize guess and solution trajectories
  Trajectory init_guess, warm_start, ref_traj, ref_traj_N;
  if (!init_file.empty()) {
    init_guess.read_from_yaml(init_file.c_str());
  } else {
    std::cout << "No init_file specified.\n";
    warm_start.num_time_steps = N;
    
  }
  if (!ref_file.empty())
  {
    ref_traj.read_from_yaml(ref_file.c_str());
  }



  // main NMPC loop
  Trajectory sol, sol_broken, sol_window;
  sol.states.clear();
  sol.actions.clear();
  sol.states.push_back(problem.start);


  int num_bodies = int((robot->nx) / 13);
  int nv = 6 * (num_bodies);
  int nq = 7 * (num_bodies);
  // double x_weightb_val = 350.0;
  if (opt_file["x_weightb"])
  {
    double x_weightb_val = opt_file["x_weightb"].as<double>();
    robot->x_weightb.head(nq + nv) = Eigen::VectorXd::Ones(robot->nx) * x_weightb_val;
    robot->x_weightb.segment(3, 4) = Eigen::VectorXd::Zero(4);                    // paylaod quat
    robot->x_weightb.segment(7 * (num_bodies) + 3, 3) = Eigen::VectorXd::Zero(3); // ang vel payload
  }

  Eigen::VectorXd x_init = problem.start;
  Eigen::VectorXd x;
  double noise_level = opt_file["control_noise"] ? opt_file["control_noise"].as<double>() : 1e-3;
  double fail_threshold = opt_file["fail_threshold"].as<double>() ? opt_file["fail_threshold"].as<double>() : 5.0;
  double goal_tol = opt_file["goal_tol"].as<double>() ? opt_file["goal_tol"].as<double>() : 0.05;
  int k_goal = 0;
  // known sizes from the robot model
  const std::size_t nx = static_cast<std::size_t>(robot->nx);
  const std::size_t nu = static_cast<std::size_t>(robot->nu);
  
  for (int k = 0; k < max_steps; ++k) {
    
    if (!init_file.empty()) {
      const std::size_t start_idx =
      std::min<std::size_t>(k, init_guess.states.size() ? init_guess.states.size()-1 : 0);
      warm_start = slice_window(init_guess, start_idx, N);
      ensure_horizon(warm_start, N, nx, nu);
    } else {
      warm_start = shift_and_pad(sol, N, nx, nu);
      ensure_horizon(warm_start, N, nx, nu);
    }
    
    if (!ref_file.empty()) {
      const std::size_t start_idx =
      std::min<std::size_t>(k, ref_traj.states.size() ? ref_traj.states.size()-1 : 0);
      ref_traj_N = slice_window(ref_traj, start_idx, N);
      ensure_horizon(ref_traj_N, N, nx, nu);
    }
    
    problem.start = x_init;
    auto t_start = std::chrono::high_resolution_clock::now();
    execute_nmpc_mujoco(problem, warm_start, ref_traj_N, sol_window, sol_broken, cfg_file);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "execute_nmpc_mujoco took " << duration_ms << " ms\n";
    Eigen::VectorXd x = x_init;
    Eigen::VectorXd u = sol_window.actions.front();
    Eigen::VectorXd xnext(robot->nx);
    u += noise_level * Eigen::VectorXd::Random(nu);
    robot->step(xnext, x.head(robot->nx), u.head(robot->nu), robot->ref_dt);
    sol.states.push_back(xnext);
    sol.actions.push_back(u);

    x_init = xnext;
    if (robot->distance(sol.states.back(), problem.goal) <= goal_tol) {
      if (k_goal == 0 && k != 0) {
        k_goal = k;
        std::cout << "Goal reached at step " << k << "\n";
        // break;
      }
      else if (k == 0 && k_goal == 0) {
        k_goal = 0;
        std::cout << "Goal reached at step 0\n";
      }
    }
    else if (robot->distance(sol.states.back(), problem.goal) > fail_threshold)
    {
      double dist = robot->distance(sol.states.back(), problem.goal);
      std::cout << "Tracking failed: distance (" << dist << ") exceeded threshold (" << fail_threshold << ") at step " << k << "\n";
      break;
    }
    else
    {
      std::cout << "Goal distance: " << robot->distance(sol.states.back(), problem.goal) << std::endl;
    }
  }
  if (max_steps < 2) {
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
  }

  // std::string base = vm["video_prefix"].as<std::string>();
  if (base.empty()) {
    fs::path init_p(init_file);
    base = (init_p.parent_path() / (init_p.stem().string() + "_viz")).string();
  }
  bool view_ghost = view_ref && !ref_file.empty();
  
  Trajectory empty_traj;
  // Remove .yaml, add random number, then append .yaml
  std::string tmp_init_guess_file = init_file;

  if (!tmp_init_guess_file.empty()) {
    size_t pos = tmp_init_guess_file.rfind(".yaml");
    if (pos != std::string::npos) {
      tmp_init_guess_file = tmp_init_guess_file.substr(0, pos);
    }
    // Generate a random number
    int rand_num = std::rand();
    tmp_init_guess_file += std::to_string(rand_num) + ".yaml";
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
  std::cout << "Writing temporary empty trajectory to: " << tmp_init_guess_file << "\n";
  empty_traj.to_yaml_format(tmp_init_guess_file.c_str());
  if (views.size() == 1 && views[0] == "auto")
  {
    std::string video_path = base + ".mp4"; // AUTO strips .mp4 and appends suffixes
    std::cout << "Writing videos to base: " << base << "_{side,top,front,diag}.mp4 (repeats=" << repeats << ")\n";
    execute_simMujoco(env_file, tmp_init_guess_file, sol, dynobench_base,
                      video_path, "auto", repeats, view_ghost, true /*feasible*/);
  }
  else
  {
    for (const auto &v : views)
    {
      std::string out = base + "_" + v + ".mp4";
      std::cout << "Writing: " << out << " (repeats=" << repeats << ")\n";
      execute_simMujoco(env_file, tmp_init_guess_file, sol, dynobench_base,
                        out, v, repeats, view_ghost, true /*feasible*/);
    }
  }
  std::cout << "Done.\n";
  return 0;
}
