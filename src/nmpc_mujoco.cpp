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

static inline void ensure_horizon(Trajectory &traj, std::size_t N)
{
  if (traj.actions.size() == N && traj.states.size() == N + 1)
    return;

  const std::size_t nx = traj.states.empty() ? 0 : traj.states.front().size();
  const std::size_t nu = traj.actions.empty() ? 0 : traj.actions.front().size();

  // truncate
  if (traj.actions.size() > N)
  {
    traj.actions.resize(N);
    traj.states.resize(N + 1);
    return;
  }
  // pad
  Eigen::VectorXd a_last = traj.actions.empty() ? Eigen::VectorXd::Zero(nu) : traj.actions.back();
  Eigen::VectorXd x_last = traj.states.empty() ? Eigen::VectorXd::Zero(nx) : traj.states.back();

  if (traj.actions.empty())
    traj.actions.assign(N, a_last);
  else
    traj.actions.resize(N, a_last);

  if (traj.states.size() < N + 1)
    traj.states.resize(N + 1, x_last);
  if (traj.states.size() > N + 1)
    traj.states.resize(N + 1);
}

static inline Trajectory shift_and_pad(const Trajectory &solved_window, std::size_t N)
{
  Trajectory warm;
  if (solved_window.actions.empty() || solved_window.states.size() < 2)
  {
    warm = solved_window;
    ensure_horizon(warm, N);
    return warm;
  }
  // drop first knot
  warm.actions.assign(solved_window.actions.begin() + 1, solved_window.actions.end());
  warm.states.assign(solved_window.states.begin() + 1, solved_window.states.end());
  // pad back to N/N+1
  ensure_horizon(warm, N);
  return warm;
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
  std::string env_file, init_file, dynobench_base, results_path, cfg_file;
  bool do_optimize = false;
  bool do_visualize = false;
  bool view_init = false;

  po::options_description desc("main_mujoco_opt_simulate options");
  desc.add_options()("help,h", "Show help")("env_file", po::value<std::string>(&env_file), "Environment YAML")("init_file", po::value<std::string>(&init_file)->default_value(""), "Initial guess YAML")("results_path", po::value<std::string>(&results_path)->default_value("../result_opt.yaml"), "Path to save optimized solution YAML (written only if -o succeeds)")("dynobench_base", po::value<std::string>(&dynobench_base), "DynoBench base directory (contains models/)")("cfg_file", po::value<std::string>(&cfg_file)->default_value(""), "optimization parameters, see optimization/options.hpp")("visualize,v", po::bool_switch(&do_visualize)->default_value(false), "Save videos; does not require -o")("view_init,i", po::bool_switch(&view_init)->default_value(false), "view the initial guess, does not require -o")("views", po::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"auto"}, "auto"), "Views: 'auto' or list of side top front diag")("repeats", po::value<int>()->default_value(2), "Number of repeats inside each video (default 2)")("video_prefix", po::value<std::string>()->default_value(""), "Optional video base; outputs base_<view>.mp4");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 0;
  }

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

  if (opt_file["init_file"])
  {
    init_file = opt_file["init_file"].as<std::string>();
  }

  // initilize guess and solution trajectories
  Trajectory init_guess, warm_start;
  if (!init_file.empty())
  {
    init_guess.read_from_yaml(init_file.c_str());
    warm_start.read_from_yaml(init_file.c_str());
    ensure_horizon(warm_start, N);
  }
  else
  {
    std::cout << "No init_file specified.\n";
    warm_start.num_time_steps = N;
  }

  Trajectory sol, sol_broken, sol_window;
  sol.states.clear();
  sol.actions.clear();
  sol.states.push_back(problem.start);

  dynoplan::Result_opti result;
  std::shared_ptr<dynobench::Model_robot> robot;
  robot = dynobench::robot_factory((models_base_path + problem.robotType + ".yaml").c_str(),
                                   problem.p_lb, problem.p_ub);

  if (!robot)
  {
    throw std::runtime_error("Failed to create robot model for type: " + problem.robotType);
  }
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

  double fail_threshold = 1e6;
  if (opt_file["fail_threshold"])
  {
    fail_threshold = opt_file["fail_threshold"].as<double>();
  }
  double goal_tol = 1e-1;
  if (opt_file["goal_tol"])
  {
    goal_tol = opt_file["goal_tol"].as<double>();
  }

  for (int k = 0; k < max_steps; ++k)
  {
    problem.start = x_init;
    auto t_start = std::chrono::high_resolution_clock::now();
    execute_nmpc_mujoco(problem, warm_start, sol_window, sol_broken, cfg_file);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "execute_nmpc_mujoco took " << duration_ms << " ms\n";
    Eigen::VectorXd x = x_init;
    Eigen::VectorXd u = sol_window.actions.front();
    Eigen::VectorXd xnext(robot->nx);
    robot->step(xnext, x.head(robot->nx), u.head(robot->nu), robot->ref_dt);
    sol.states.push_back(xnext);
    sol.actions.push_back(u);
    if (!init_file.empty()) {
      warm_start = shift_and_pad(init_guess, N);
    } else { 
      warm_start = shift_and_pad(sol_window, N);
    }
    ensure_horizon(warm_start, N);
    x_init = xnext;
    if (robot->distance(sol.states.back(), problem.goal) < goal_tol)
    {
      std::cout << "Goal reached at step " << k << "\n";
      break;
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
    std::cout << "Step " << k << " done. Current position: " << xnext.transpose() << std::endl;
  }
  if (max_steps < 2)
    sol = sol_window; // if only one step, show ghost of init
  std::cout << "states size: " << sol.states.size() << std::endl;
  std::cout << "actions size: " << sol.actions.size() << std::endl;
  Problem problem_final(env_file.c_str());
  sol.cost = sol.actions.size() * robot->ref_dt; // time cost
  sol.start = problem_final.start;
  sol.goal = problem_final.goal;
  sol.check(robot, true);
  if (!results_path.empty())
  {
    sol.to_yaml_format(results_path.c_str());
    std::cout << "Saved NMPC stitched trajectory to: " << results_path << "\n";
  }

  std::string base = vm["video_prefix"].as<std::string>();
  if (base.empty())
  {
    fs::path init_p(init_file);
    base = (init_p.parent_path() / (init_p.stem().string() + "_viz")).string();
  }
  bool view_ghost = view_init && !init_file.empty();
  auto views = vm["views"].as<std::vector<std::string>>();
  int repeats = vm["repeats"].as<int>();
  if (views.size() == 1 && views[0] == "auto")
  {
    std::string video_path = base + ".mp4"; // AUTO strips .mp4 and appends suffixes
    std::cout << "Writing videos to base: " << base << "_{side,top,front,diag}.mp4 (repeats=" << repeats << ")\n";
    execute_simMujoco(env_file, init_file, sol, dynobench_base,
                      video_path, "auto", repeats, view_ghost, true /*feasible*/);
  }
  else
  {
    for (const auto &v : views)
    {
      std::string out = base + "_" + v + ".mp4";
      std::cout << "Writing: " << out << " (repeats=" << repeats << ")\n";
      execute_simMujoco(env_file, init_file, sol, dynobench_base,
                        out, v, repeats, view_ghost, true /*feasible*/);
    }
  }
  std::cout << "Done.\n";
  return 0;
}
