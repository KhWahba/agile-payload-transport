#include "ocp.hpp"
#include "dyno_macros.hpp"
#include "math_utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <filesystem>

#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"

#include "general_utils.hpp"
#include "robot_models.hpp"
#include "croco_models.hpp"

using vstr = std::vector<std::string>;
using V2d = Eigen::Vector2d;
using V3d = Eigen::Vector3d;
using V4d = Eigen::Vector4d;
using Vxd = Eigen::VectorXd;
using V1d = Eigen::Matrix<double, 1, 1>;

using dynobench::Trajectory;

namespace dynoplan {

using dynobench::enforce_bounds;
using dynobench::FMT;

class CallVerboseDyno : public crocoddyl::CallbackAbstract {
public:
  using Traj =
      std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>;
  std::vector<Traj> trajs;

  explicit CallVerboseDyno() = default;
  ~CallVerboseDyno() override = default;

  void operator()(crocoddyl::SolverAbstract &solver) override {
    std::cout << "adding trajectory" << std::endl;
    trajs.push_back(std::make_pair(solver.get_xs(), solver.get_us()));
  }

  void store() {
    std::string timestamp = get_time_stamp();
    std::string folder = "iterations/" + timestamp;
    std::filesystem::create_directories(folder);

    for (size_t i = 0; i < trajs.size(); i++) {
      auto &traj = trajs.at(i);
      dynobench::Trajectory tt;
      tt.states = traj.first;
      tt.actions = traj.second;

      std::stringstream ss;
      ss << std::setfill('0') << std::setw(4) << i;
      std::ofstream out(folder + "/it" + ss.str() + ".yaml");
      tt.to_yaml_format(out);
    }
  }
};

void write_states_controls(const std::vector<Eigen::VectorXd> &xs,
                           const std::vector<Eigen::VectorXd> &us,
                           std::shared_ptr<dynobench::Model_robot> model_robot,
                           const dynobench::Problem &problem,
                           const char *filename) {

  // store the init guess:
  dynobench::Trajectory __traj;
  __traj.actions = us;
  __traj.states = xs;

  {
    std::ofstream out(filename + std::string(".raw.yaml"));
    out << "states:" << std::endl;
    for (auto &x : xs) {
      out << "- " << x.format(FMT) << std::endl;
    }
    out << "actions:" << std::endl;
    for (auto &u : us) {
      out << "- " << u.format(FMT) << std::endl;
    }
  }

  if (__traj.actions.front().size() > model_robot->nu) {
    for (size_t i = 0; i < __traj.actions.size(); i++) {
      __traj.actions.at(i) =
          Eigen::VectorXd(__traj.actions.at(i).head(model_robot->nu));
    }
  }
  if (__traj.states.front().size() > model_robot->nx) {
    for (size_t i = 0; i < __traj.states.size(); i++) {
      __traj.states.at(i) =
          Eigen::VectorXd(__traj.states.at(i).head(model_robot->nx));
    }
  }

  // }

  __traj.start = problem.start;
  __traj.goal = problem.goal;

  // create directory if necessary
  if (const std::filesystem::path path =
          std::filesystem::path(filename).parent_path();
      !path.empty()) {
    std::filesystem::create_directories(path);
  }

  std::ofstream init_guess(filename);
  CSTR_(filename);

  std::cout << "Check traj in controls " << std::endl;
  __traj.check(model_robot, true);
  std::cout << "Check traj in controls -- DONE " << std::endl;
  __traj.to_yaml_format(init_guess);
}



void add_extra_time_rate(std::vector<Eigen::VectorXd> &us_init) {
  std::vector<Vxd> us_init_time(us_init.size());
  size_t nu = us_init.front().size();
  for (size_t i = 0; i < us_init.size(); i++) {
    Vxd u(nu + 1);
    u.head(nu) = us_init.at(i);
    u(nu) = 1.;
    us_init_time.at(i) = u;
  }
  us_init = us_init_time;
};

void add_extra_state_time_rate(std::vector<Eigen::VectorXd> &xs_init,
                               Eigen::VectorXd &start) {
  std::vector<Vxd> xs_init_time(xs_init.size());
  size_t nx = xs_init.front().size();
  for (size_t i = 0; i < xs_init_time.size(); i++) {
    Vxd x(nx + 1);
    x.head(nx) = xs_init.at(i);
    x(nx) = 1.;
    xs_init_time.at(i) = x;
  }
  xs_init = xs_init_time;
  Eigen::VectorXd old_start = start;
  start.resize(nx + 1);
  start << old_start, 1.;
};

void check_problem_with_finite_diff(
    Options_trajopt options, Generate_params gen_args,
    ptr<crocoddyl::ShootingProblem> problem_croco, const std::vector<Vxd> &xs,
    const std::vector<Vxd> &us) {
  std::cout << "Checking with finite diff " << std::endl;
  options.use_finite_diff = true;
  options.disturbance = 1e-5;
  std::cout << "gen problem " << STR_(AT) << std::endl;
  size_t nx, nu;
  ptr<crocoddyl::ShootingProblem> problem_fdiff =
      generate_problem(gen_args, options);
  check_problem(problem_croco, problem_fdiff, xs, us);
};

void add_noise(double noise_level, std::vector<Eigen::VectorXd> &xs,
               std::vector<Eigen::VectorXd> &us,
               std::shared_ptr<dynobench::Model_robot> model_robot) {
  size_t nx = xs.at(0).size();
  size_t nu = us.at(0).size();
  for (size_t i = 0; i < xs.size(); i++) {
    DYNO_CHECK_EQ(static_cast<size_t>(xs.at(i).size()), nx, AT);
    xs.at(i) += noise_level * Vxd::Random(nx);
    model_robot->ensure(xs.at(i));
  }

  for (size_t i = 0; i < us.size(); i++) {
    DYNO_CHECK_EQ(static_cast<size_t>(us.at(i).size()), nu, AT);
    us.at(i) += noise_level * Vxd::Random(nu);
  }
};

void mpc_adaptative_warmstart(
    size_t counter, size_t window_optimize_i, std::vector<Vxd> &xs,
    std::vector<Vxd> &us, std::vector<Vxd> &xs_warmstart,
    std::vector<Vxd> &us_warmstart,
    std::shared_ptr<dynobench::Model_robot> model_robot, bool shift_repeat,
    ptr<dynobench::Interpolator> path, ptr<dynobench::Interpolator> path_u,
    double max_alpha) {
  size_t _nx = model_robot->nx;
  size_t _nu = model_robot->nu;
  size_t nu = us_warmstart.front().size();
  double dt = model_robot->ref_dt;

  if (counter) {
    std::cout << "new warmstart" << std::endl;
    xs = xs_warmstart;
    us = us_warmstart;
    DYNO_CHECK_GE(nu, 0, AT);
    size_t missing_steps = window_optimize_i - us.size();

    Vxd u_last = Vxd::Zero(nu);

    u_last.head(model_robot->nu) = model_robot->u_0;

    Vxd x_last = xs.back();

    // TODO: Sample the interpolator to get new init guess.

    if (shift_repeat) {
      for (size_t i = 0; i < missing_steps; i++) {
        us.push_back(u_last);
        xs.push_back(x_last);
      }
    } else {

      std::cout << "filling window by sampling the trajectory" << std::endl;
      Vxd last = xs_warmstart.back().head(_nx);

      auto it = std::min_element(path->x.begin(), path->x.end(),
                                 [&](const auto &a, const auto &b) {
                                   return model_robot->distance(a, last) <
                                          model_robot->distance(b, last);
                                 });

      size_t last_index = std::distance(path->x.begin(), it);
      double alpha_of_last = path->times(last_index);
      std::cout << STR_(last_index) << std::endl;
      // now I

      Vxd out(_nx);
      Vxd J(_nx);

      Vxd out_u(_nu);
      Vxd J_u(_nu);

      for (size_t i = 0; i < missing_steps; i++) {
        {
          path->interpolate(std::min(alpha_of_last + (i + 1) * dt, max_alpha),
                            out, J);
          xs.push_back(out);
        }

        {
          path_u->interpolate(std::min(alpha_of_last + i * dt, max_alpha - dt),
                              out_u, J_u);
          us.push_back(out_u);
        }
      }
    }

  } else {
    std::cout << "first iteration -- using first" << std::endl;

    if (window_optimize_i + 1 < xs_warmstart.size()) {
      xs = std::vector<Vxd>(xs_warmstart.begin(),
                            xs_warmstart.begin() + window_optimize_i + 1);
      us = std::vector<Vxd>(us_warmstart.begin(),
                            us_warmstart.begin() + window_optimize_i);
    } else {
      std::cout << "Optimizing more steps than required" << std::endl;
      xs = xs_warmstart;
      us = us_warmstart;

      size_t missing_steps = window_optimize_i - us.size();
      Vxd u_last = Vxd::Zero(nu);

      u_last.head(model_robot->nu) = model_robot->u_0;

      Vxd x_last = xs.back();

      // TODO: Sample the interpolator to get new init guess.

      for (size_t i = 0; i < missing_steps; i++) {
        us.push_back(u_last);
        xs.push_back(x_last);
      }
    }
  }
};


void solve_for_fixed_penalty(
    Generate_params &gen_args, Options_trajopt &options_trajopt_local,
    const std::vector<Eigen::VectorXd> &xs_init,
    const std::vector<Eigen::VectorXd> &us_init, bool check_with_finite_diff,
    size_t N, const std::string &name, size_t &ddp_iterations, double &ddp_time,
    std::vector<Eigen::VectorXd> &xs_out, std::vector<Eigen::VectorXd> &us_out,
    std::shared_ptr<dynobench::Model_robot> model_robot,
    const dynobench::Problem &problem, const std::string folder_tmptraj,
    bool store_iterations, boost::shared_ptr<CallVerboseDyno> callback_dyno) {
  // generate problem
  ptr<crocoddyl::ShootingProblem> problem_croco =
      generate_problem(gen_args, options_trajopt_local);

  size_t nu = model_robot->nu;
  if (gen_args.free_time) {
    nu++;
  }

  // warmstart
  std::vector<Vxd> xs, us;
  if (options_trajopt_local.use_warmstart) {
    xs = xs_init;
    us = us_init;
  } else {
    xs = std::vector<Eigen::VectorXd>(N + 1, gen_args.start);
    Eigen::VectorXd u0 = Vxd(nu);
    u0 << model_robot->u_0, 1;
    us = std::vector<Vxd>(N, u0);
  }

  // add noise
  if (options_trajopt_local.noise_level > 0.) {
    add_noise(options_trajopt_local.noise_level, xs, us, model_robot);
  }

  // store init guess
  // report_problem(problem_croco, xs, us, "/tmp/dynoplan/report-0.yaml");
  std::cout << "solving with croco " << AT << std::endl;

  std::string random_id = gen_random(6);
  {
    std::string filename = folder_tmptraj + "init_guess_" + random_id + ".yaml";
    // write_states_controls(xs, us, model_robot, problem, filename.c_str());
  }

  // solve
  crocoddyl::SolverBoxFDDP ddp(problem_croco);
  ddp.set_th_stop(options_trajopt_local.th_stop);
  ddp.set_th_acceptnegstep(options_trajopt_local.th_acceptnegstep);

  if (options_trajopt_local.CALLBACKS) {
    std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
    cbs.push_back(mk<crocoddyl::CallbackVerbose>());
    if (store_iterations) {
      cbs.push_back(callback_dyno);
    }
    ddp.setCallbacks(cbs);
  }

  std::cout << "CROCO optimize " << AT << std::endl;
  crocoddyl::Timer timer;
  ddp.solve(xs, us, options_trajopt_local.max_iter, false,
            options_trajopt_local.init_reg);
  std::cout << "time: " << timer.get_duration() << std::endl;

  if (store_iterations)
    callback_dyno->store();
  std::cout << "CROCO optimize -- DONE" << std::endl;
  ddp_iterations += ddp.get_iter();
  ddp_time += timer.get_duration();
  xs_out = ddp.get_xs();
  us_out = ddp.get_us();

  // report after
  std::string filename = folder_tmptraj + "opt_" + random_id + ".yaml";

  for (auto &x : xs_out) {
    model_robot->ensure(x);
  }

  // write_states_controls(xs_out, us_out, model_robot, problem, filename.c_str());
  // report_problem(problem_croco, xs_out, us_out, "/tmp/dynoplan/report-1.yaml");
};

void __trajectory_optimization(
    const dynobench::Problem &t_problem,
    std::shared_ptr<dynobench::Model_robot> &model_robot,
    const dynobench::Trajectory &init_guess,
    const Options_trajopt &options_trajopt, dynobench::Trajectory &traj,
    Result_opti &opti_out) {

  dynobench::Problem problem = t_problem;

  model_robot->ensure(problem.start); // precission issues with quaternions
  model_robot->ensure(problem.goal);

  Options_trajopt options_trajopt_local = options_trajopt;

  std::vector<SOLVER> solvers{SOLVER::traj_opt,
                              SOLVER::traj_opt_free_time_proxi,
                              SOLVER::mpc,
                              SOLVER::mpcc,
                              SOLVER::mpcc2,
                              SOLVER::mpcc_linear,
                              SOLVER::mpc_adaptative,
                              SOLVER::traj_opt_free_time_proxi_linear};

  CHECK(__in_if(solvers,
                [&](const SOLVER &s) {
                  return s ==
                         static_cast<SOLVER>(options_trajopt_local.solver_id);
                }),
        "solver_id not in solvers");

  const bool modify_to_match_goal_start = false;
  const bool store_iterations = false;
  const std::string folder_tmptraj = "/tmp/dynoplan/";

  // std::cout
  //     << "WARNING: "
  //     << "Cleaning data in opti_out at beginning of __trajectory_optimization"
  //     << std::endl;
  opti_out.data.clear();

  auto callback_dyno = mk<CallVerboseDyno>();

  {
    dynobench::Trajectory __init_guess = init_guess;
    __init_guess.start = problem.start;
    __init_guess.goal = problem.goal;
    std::cout << "checking traj input of __trajectory_optimization "
              << std::endl;
    __init_guess.check(model_robot, false);
    std::cout << "checking traj input of __trajectory_optimization -- DONE "
              << std::endl;
  }

  size_t ddp_iterations = 0;
  double ddp_time = 0;

  bool check_with_finite_diff = true;
  // std::string name = problem.robotType;
  std::string name = model_robot->name;
  size_t _nx = model_robot->nx;
  size_t _nu = model_robot->nu;

  bool verbose = false;
  auto xs_init = init_guess.states;
  auto us_init = init_guess.actions;
  DYNO_CHECK_EQ(xs_init.size(), us_init.size() + 1, AT);
  size_t N = init_guess.actions.size();
  auto goal = problem.goal;
  auto start = problem.start;
  double dt = model_robot->ref_dt;

  SOLVER solver = static_cast<SOLVER>(options_trajopt_local.solver_id);

  if (modify_to_match_goal_start) {
    std::cout << "WARNING: " << "i modify last state to match goal"
              << std::endl;
    xs_init.back() = goal;
    xs_init.front() = start;
  }

  // write_states_controls(xs_init, us_init, model_robot, problem,
  //                       (folder_tmptraj + "init_guess.yaml").c_str());

  size_t num_smooth_iterations =
      dt > .05 ? 3 : 5; // TODO: put this as an option in command line

  std::vector<Eigen::VectorXd> xs_init__ = xs_init;

  if (options_trajopt_local.smooth_traj) {
    for (size_t i = 0; i < num_smooth_iterations; i++) {
      xs_init = smooth_traj2(xs_init, *model_robot->state);
    }

    for (size_t i = 0; i < num_smooth_iterations; i++) {
      us_init = smooth_traj2(us_init, dynobench::Rn(us_init.front().size()));
    }
  }

  // for (auto &x : xs_init) {
  //   model_robot->ensure(x);
  // }

  // write_states_controls(xs_init, us_init, model_robot, problem,
  //                       (folder_tmptraj + "init_guess_smooth.yaml").c_str());

  bool success = false;
  std::vector<Vxd> xs_out, us_out;

  create_dir_if_necessary(options_trajopt_local.debug_file_name.c_str());
  std::ofstream debug_file_yaml(options_trajopt_local.debug_file_name);
  {
    debug_file_yaml << "robotType: " << problem.robotType << std::endl;
    debug_file_yaml << "N: " << N << std::endl;
    debug_file_yaml << "start: " << start.format(FMT) << std::endl;
    debug_file_yaml << "goal: " << goal.format(FMT) << std::endl;
    debug_file_yaml << "xs0: " << std::endl;
    for (auto &x : xs_init)
      debug_file_yaml << "  - " << x.format(FMT) << std::endl;

    debug_file_yaml << "us0: " << std::endl;
    for (auto &x : us_init)
      debug_file_yaml << "  - " << x.format(FMT) << std::endl;
  }

  bool __free_time_mode = solver == SOLVER::traj_opt_free_time_proxi ||
                          solver == SOLVER::traj_opt_free_time_proxi_linear;

  // WINDOW APPROACH
  if (solver == SOLVER::mpc || solver == SOLVER::mpcc ||
      solver == SOLVER::mpcc_linear || solver == SOLVER::mpc_adaptative) {

    DYNO_CHECK_GEQ(options_trajopt_local.window_optimize,
                   options_trajopt_local.window_shift, AT);

    bool finished = false;

    std::vector<Vxd> xs_opt, us_opt;
    std::vector<Vxd> xs, us;

    std::vector<Vxd> xs_init_rewrite = xs_init;
    std::vector<Vxd> us_init_rewrite = us_init;

    std::vector<Vxd> xs_warmstart, us_warmstart;
    xs_warmstart = xs_init;
    us_warmstart = us_init;

    xs_opt.push_back(start);
    xs_init_rewrite.at(0) = start;

    debug_file_yaml << "opti:" << std::endl;

    auto times = Vxd::LinSpaced(xs_init.size(), 0, (xs_init.size() - 1) * dt);

    double max_alpha = times(times.size() - 1);

    ptr<dynobench::Interpolator> path =
        mk<dynobench::Interpolator>(times, xs_init, model_robot->state);
    ptr<dynobench::Interpolator> path_u =
        mk<dynobench::Interpolator>(times.head(times.size() - 1), us_init);

    Vxd previous_state = start;
    ptr<crocoddyl::ShootingProblem> problem_croco;

    Vxd goal_mpc(_nx);

    bool is_last = false;

    double total_time = 0;
    size_t counter = 0;
    size_t total_iterations = 0;
    size_t window_optimize_i = 0;

    bool close_to_goal = false;

    Vxd goal_with_alpha(_nx + 1);
    goal_with_alpha.head(_nx) = goal;
    goal_with_alpha(_nx) = max_alpha;

    bool last_reaches_ = false;
    size_t index_first_goal = 0;

    Generate_params gen_args;

    auto fun_is_goal = [&](const auto &x) {
      return model_robot->distance(x.head(_nx), goal) < 1e-2;
    };

    while (!finished) {
      if (solver == SOLVER::mpc) {
        DYNO_CHECK_GEQ(
            int(N) - int(counter * options_trajopt_local.window_shift), 0, "");
        size_t remaining_steps =
            N - counter * options_trajopt_local.window_shift;

        const bool adaptative_last_window = true;

        window_optimize_i = options_trajopt_local.window_optimize;
        if (adaptative_last_window) {
          window_optimize_i = std::min(window_optimize_i, remaining_steps);
        }

        if (options_trajopt_local.window_optimize > remaining_steps)
          goal_mpc = goal;
        else
          goal_mpc = xs_init.at(counter * options_trajopt_local.window_shift +
                                window_optimize_i);

        std::cout << "goal_i:" << goal_mpc.transpose() << std::endl;
        std::cout << "start_i:" << previous_state.transpose() << std::endl;

        gen_args = Generate_params{
            .name = name,
            .N = window_optimize_i,
            .goal = goal_mpc,
            .start = previous_state,
            .model_robot = model_robot,
            .collisions = options_trajopt_local.collision_weight > 1e-3};

        problem_croco = generate_problem(gen_args, options_trajopt_local);
        is_last = options_trajopt_local.window_optimize > remaining_steps;

        if (options_trajopt_local.use_warmstart) {
          warmstart_mpc(xs, us, xs_init, us_init, counter, window_optimize_i,
                        options_trajopt_local.window_shift);
        } else {
          xs = std::vector<Vxd>(window_optimize_i + 1, gen_args.start);
          us = std::vector<Vxd>(window_optimize_i, Vxd::Zero(model_robot->nu));
        }

      }

      else if (solver == SOLVER::mpc_adaptative) {

        std::cout << "previous state is " << previous_state.format(FMT)
                  << std::endl;
        auto it = std::min_element(
            path->x.begin(), path->x.end(), [&](const auto &a, const auto &b) {
              return model_robot->distance(a, previous_state) <
                     model_robot->distance(b, previous_state);
            });

        size_t index = std::distance(path->x.begin(), it);
        std::cout << "starting with index " << index << std::endl;
        std::cout << "Non adaptative index would be: "
                  << counter * options_trajopt_local.window_shift << std::endl;

        window_optimize_i = options_trajopt_local.window_optimize;
        // next goal:
        size_t goal_index = index + window_optimize_i;
        CSTR_(goal_index);
        if (goal_index > xs_init.size() - 1) {
          std::cout << "trying to reach the goal " << std::endl;
          goal_mpc = goal;
        } else {
          goal_mpc = xs_init.at(goal_index);
        }

        std::cout << "goal_i:" << goal_mpc.transpose() << std::endl;
        std::cout << "start_i:" << previous_state.transpose() << std::endl;

        gen_args = Generate_params{
            .name = name,
            .N = window_optimize_i,
            .goal = goal_mpc,
            .start = previous_state,
            .model_robot = model_robot,
            .collisions = options_trajopt_local.collision_weight > 1e-3};

        size_t nx, nu;
        problem_croco = generate_problem(gen_args, options_trajopt_local);

        if (options_trajopt_local.use_warmstart) {
          mpc_adaptative_warmstart(counter, window_optimize_i, xs, us,
                                   xs_warmstart, us_warmstart, model_robot,
                                   options_trajopt_local.shift_repeat, path,
                                   path_u, max_alpha);

        } else {
          xs = std::vector<Vxd>(window_optimize_i + 1, gen_args.start);
          us = std::vector<Vxd>(window_optimize_i, Vxd::Zero(nu));
        }
      } else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {

        window_optimize_i =
            std::min(options_trajopt_local.window_optimize, us_init.size());

        std::cout << "previous state " << previous_state.format(FMT)
                  << std::endl;
        auto it = std::min_element(
            path->x.begin(), path->x.end(), [&](const auto &a, const auto &b) {
              return model_robot->distance(a, previous_state) <=
                     model_robot->distance(b, previous_state);
            });
        size_t first_index = std::distance(path->x.begin(), it);
        double alpha_of_first = path->times(first_index);

        size_t expected_last_index = first_index + window_optimize_i;

        Vxd alpha_refs =
            Vxd::LinSpaced(window_optimize_i + 1, alpha_of_first,
                           alpha_of_first + window_optimize_i * dt);

        double expected_final_alpha =
            std::min(alpha_of_first + window_optimize_i * dt, max_alpha);

        std::cout << STR(first_index, ":") << std::endl;
        std::cout << STR(expected_final_alpha, ":") << std::endl;
        std::cout << STR(expected_last_index, ":") << std::endl;
        std::cout << STR(alpha_of_first, ":") << std::endl;
        std::cout << STR(max_alpha, ":") << std::endl;

        size_t nx, nu;

        Vxd start_ic(_nx + 1);
        start_ic.head(_nx) = previous_state.head(_nx);
        start_ic(_nx) = alpha_of_first;

        bool goal_cost = false;

        if (expected_final_alpha > max_alpha - 1e-3 || close_to_goal) {
          std::cout << "alpha_refs > max_alpha || close to goal" << std::endl;
          goal_cost = true; // new
        }

        std::cout << "goal " << goal_with_alpha.format(FMT) << std::endl;
        std::cout << STR_(goal_cost) << std::endl;

        std::vector<Vxd> state_weights;
        std::vector<Vxd> _states(window_optimize_i);

        int try_faster = 5;
        if (last_reaches_) {
          std::cout << "last_reaches_ adds goal cost special" << std::endl;
          std::cout << "try_faster: " << try_faster << std::endl;

          state_weights.resize(window_optimize_i);
          _states.resize(window_optimize_i);

          for (size_t t = 0; t < window_optimize_i; t++) {
            if (t > index_first_goal - options_trajopt_local.window_shift -
                        try_faster)
              state_weights.at(t) = 1. * Vxd::Ones(_nx + 1);
            else
              state_weights.at(t) = Vxd::Zero(_nx + 1);
          }

          for (size_t t = 0; t < window_optimize_i; t++) {
            _states.at(t) = goal_with_alpha;
          }
        }

        gen_args = Generate_params{
            .free_time = false,
            .name = name,
            .N = window_optimize_i,
            .goal = goal,
            .start = start_ic,
            .model_robot = model_robot,
            .states = _states,
            .states_weights = state_weights,
            .actions = {},
            .contour_control = true,
            .interpolator = path,
            .max_alpha = max_alpha,
            .linear_contour = solver == SOLVER::mpcc_linear,
            .goal_cost = goal_cost,
            .collisions = options_trajopt_local.collision_weight > 1e-3

        };

        std::cout << "gen problem " << STR_(AT) << std::endl;
        problem_croco = generate_problem(gen_args, options_trajopt_local);

        if (options_trajopt_local.use_warmstart) {

          warmstart_mpcc(xs_warmstart, us_warmstart, counter, window_optimize_i,
                         xs, us, model_robot,
                         options_trajopt_local.shift_repeat, path, path_u, dt,
                         max_alpha, xs_init, us_init);

        } else {
          std::cout << "no warmstart " << std::endl;
          Vxd u0c(_nu + 1);
          u0c.head(_nu) = model_robot->u_0;
          u0c(_nu) = dt;
          xs = std::vector<Vxd>(window_optimize_i + 1, gen_args.start);
          us = std::vector<Vxd>(window_optimize_i, u0c);
        }
      }

      DYNO_CHECK_EQ(xs.size(), window_optimize_i + 1, AT);
      DYNO_CHECK_EQ(us.size(), window_optimize_i, AT);

      if (!options_trajopt_local.use_finite_diff &&
          options_trajopt_local.check_with_finite_diff) {
        check_problem_with_finite_diff(options_trajopt_local, gen_args,
                                       problem_croco, xs, us);
      }

      // report problem

      // auto models = problem->get_runningModels();

      crocoddyl::SolverBoxFDDP ddp(problem_croco);
      ddp.set_th_stop(options_trajopt_local.th_stop);
      ddp.set_th_acceptnegstep(options_trajopt_local.th_acceptnegstep);

      if (options_trajopt_local.CALLBACKS) {
        std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
        cbs.push_back(mk<crocoddyl::CallbackVerbose>());
        if (store_iterations) {
          cbs.push_back(callback_dyno);
        }
        ddp.setCallbacks(cbs);
      }

      if (options_trajopt_local.noise_level > 1e-8) {
        add_noise(options_trajopt_local.noise_level, xs, us, model_robot);
      }

      report_problem(problem_croco, xs, us, "/tmp/dynoplan/report-0.yaml");

      std::string random_id = gen_random(6);

      {
        std::string filename =
            folder_tmptraj + "init_guess_" + random_id + ".yaml";
        write_states_controls(xs, us, model_robot, problem, filename.c_str());
      }

      std::cout << "CROCO optimize" << AT << std::endl;
      crocoddyl::Timer timer;
      ddp.solve(xs, us, options_trajopt_local.max_iter, false,
                options_trajopt_local.init_reg);
      std::cout << "CROCO optimize -- DONE" << std::endl;

      if (store_iterations) {
        callback_dyno->store();
      }

      {
        std::string filename = folder_tmptraj + "opt_" + random_id + ".yaml";
        write_states_controls(ddp.get_xs(), ddp.get_us(), model_robot, problem,
                              filename.c_str());

        std::string filename_raw =
            folder_tmptraj + "opt_" + random_id + ".raw.yaml";
        dynobench::Trajectory traj;
        traj.states = ddp.get_xs();
        traj.actions = ddp.get_us();
        traj.to_yaml_format(filename_raw.c_str());
      }

      double time_i = timer.get_duration();
      size_t iterations_i = ddp.get_iter();
      ddp_iterations += ddp.get_iter();
      ddp_time += timer.get_duration();
      report_problem(problem_croco, ddp.get_xs(), ddp.get_us(),
                     "/tmp/dynoplan/report-1.yaml");

      std::cout << "time: " << time_i << std::endl;
      std::cout << "iterations: " << iterations_i << std::endl;
      total_time += time_i;
      total_iterations += iterations_i;
      std::vector<Vxd> xs_i_sol = ddp.get_xs();
      std::vector<Vxd> us_i_sol = ddp.get_us();

      previous_state =
          xs_i_sol.at(options_trajopt_local.window_shift).head(_nx);

      size_t copy_steps = 0;

      // After optimization, I check If I am reaching the goal
      if (solver == SOLVER::mpc_adaptative) {

        size_t final_index = window_optimize_i;
        Vxd x_last = ddp.get_xs().at(final_index);
        std::cout << "**\n" << std::endl;
        std::cout << "checking as final index: " << final_index << std::endl;
        std::cout << "last state: " << x_last.format(FMT) << std::endl;
        std::cout << "true last state: " << ddp.get_xs().back().format(FMT)
                  << std::endl;
        std::cout << "distance to goal: "
                  << model_robot->distance(x_last.head(_nx), goal) << std::endl;

        if (fun_is_goal(x_last)) {
          std::cout << " x last " << x_last.format(FMT) << "reaches the goal"
                    << std::endl;

          std::cout << "setting x last to true " << std::endl;
          is_last = true;
        }

      } else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {

        std::cout << "if final reaches the goal, i stop" << std::endl;
        std::cout << "ideally, I should check if I can check the goal "
                     "faster, "
                     "with a small linear search "
                  << std::endl;
        size_t final_index = window_optimize_i;
        std::cout << "final index is " << final_index << std::endl;

        double alpha_mpcc = ddp.get_xs().at(final_index)(_nx);
        Vxd x_last = ddp.get_xs().at(final_index);
        last_reaches_ = fun_is_goal(ddp.get_xs().back());

        std::cout << "**\n" << std::endl;
        std::cout << "checking as final index: " << final_index << std::endl;
        std::cout << "alpha_mpcc:" << alpha_mpcc << std::endl;
        std::cout << "last state: " << x_last.format(FMT) << std::endl;
        std::cout << "true last state: " << ddp.get_xs().back().format(FMT)
                  << std::endl;
        std::cout << "distance to goal: "
                  << model_robot->distance(x_last.head(_nx), goal) << std::endl;

        std::cout << "last_reaches_: " << last_reaches_ << std::endl;
        std::cout << "\n**\n";

        if (last_reaches_) {

          auto it = std::find_if(ddp.get_xs().begin(), ddp.get_xs().end(),
                                 [&](const auto &x) { return fun_is_goal(x); });

          bool __flag = it != ddp.get_xs().end();
          CHECK(__flag, AT);

          index_first_goal = std::distance(ddp.get_xs().begin(), it);
          std::cout << "index first goal " << index_first_goal << std::endl;
        }

        if (std::fabs(alpha_mpcc - times(times.size() - 1)) < 1. &&
            fun_is_goal(x_last)) {

          is_last = true;

          std::cout << "checking first state that reaches the goal "
                    << std::endl;

          auto it = std::find_if(ddp.get_xs().begin(), ddp.get_xs().end(),
                                 [&](const auto &x) { return fun_is_goal(x); });

          assert(it != ddp.get_xs().end());

          window_optimize_i = std::distance(ddp.get_xs().begin(), it);
          std::cout << "changing the number of steps to optimize(copy) to "
                    << window_optimize_i << std::endl;
        }

        std::cout << "checking if i am close to the goal " << std::endl;

        for (size_t i = 0; i < ddp.get_xs().size(); i++) {
          auto &x = ddp.get_xs().at(i);
          if (model_robot->distance(x.head(_nx), goal) < 1e-1) {
            std::cout << "one state is close to goal! " << std::endl;
            close_to_goal = true;
          }

          if (std::fabs(x(_nx) - max_alpha) < 1e-1) {
            std::cout << "alpha is close to final " << std::endl;
            close_to_goal = true;
          }
        }

        std::cout << "done" << std::endl;
      }

      if (is_last)
        copy_steps = window_optimize_i;
      else
        copy_steps = options_trajopt_local.window_shift;

      for (size_t i = 0; i < copy_steps; i++)
        xs_opt.push_back(xs_i_sol.at(i + 1).head(_nx));

      for (size_t i = 0; i < copy_steps; i++)
        us_opt.push_back(us_i_sol.at(i).head(_nu));

      if (solver == SOLVER::mpc) {
        for (size_t i = 0; i < window_optimize_i; i++) {
          xs_init_rewrite.at(1 + counter * options_trajopt_local.window_shift +
                             i) = xs_i_sol.at(i + 1).head(_nx);

          us_init_rewrite.at(counter * options_trajopt_local.window_shift + i) =
              us_i_sol.at(i).head(_nu);
        }
      } else if (solver == SOLVER::mpc_adaptative || solver == SOLVER::mpcc ||
                 solver == SOLVER::mpcc_linear) {

        xs_warmstart.clear();
        us_warmstart.clear();

        for (size_t i = copy_steps; i < window_optimize_i; i++) {
          xs_warmstart.push_back(xs_i_sol.at(i));
          us_warmstart.push_back(us_i_sol.at(i));
        }
        xs_warmstart.push_back(xs_i_sol.back());
      }

      // DEBUGGING
      debug_file_yaml << "  - xs0:" << std::endl;
      for (auto &x : xs)
        debug_file_yaml << "    - " << x.format(FMT) << std::endl;

      debug_file_yaml << "    us0:" << std::endl;
      for (auto &u : us)
        debug_file_yaml << "    - " << u.format(FMT) << std::endl;

      debug_file_yaml << "    xsOPT:" << std::endl;
      for (auto &x : xs_i_sol)
        debug_file_yaml << "    - " << x.format(FMT) << std::endl;

      debug_file_yaml << "    usOPT:" << std::endl;
      for (auto &u : us_i_sol)
        debug_file_yaml << "    - " << u.format(FMT) << std::endl;

      debug_file_yaml << "    start: " << xs.front().format(FMT) << std::endl;

      if (solver == SOLVER::mpc || solver == SOLVER::mpc_adaptative) {
        debug_file_yaml << "    goal: " << goal_mpc.format(FMT) << std::endl;
      } else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {
        double alpha_mpcc = ddp.get_xs().back()(_nx);
        Vxd out(_nx);
        Vxd Jout(_nx);
        path->interpolate(alpha_mpcc, out, Jout);
        debug_file_yaml << "    alpha: " << alpha_mpcc << std::endl;
        debug_file_yaml << "    state_alpha: " << out.format(FMT) << std::endl;
      }

      DYNO_CHECK_EQ(us_i_sol.size() + 1, xs_i_sol.size(), AT);

      // copy results

      if (is_last) {
        finished = true;
        std::cout << "finished: " << "is_last=TRUE" << std::endl;
      }

      counter++;

      if (counter > options_trajopt_local.max_mpc_iterations) {
        finished = true;
        std::cout << "finished: " << "max mpc iterations" << std::endl;
      }
    }
    std::cout << "Total TIME: " << total_time << std::endl;
    std::cout << "Total Iterations: " << total_iterations << std::endl;

    xs_out = xs_opt;
    us_out = us_opt;

    debug_file_yaml << "xsOPT: " << std::endl;
    for (auto &x : xs_out)
      debug_file_yaml << "  - " << x.format(FMT) << std::endl;

    debug_file_yaml << "usOPT: " << std::endl;
    for (auto &u : us_out)
      debug_file_yaml << "  - " << u.format(FMT) << std::endl;

    // checking feasibility
    Trajectory traj;
    traj.states.resize(xs_out.size());
    traj.actions.resize(us_out.size());
    for (size_t i = 0; i < xs_out.size(); i++)
      traj.states.at(i) = xs_out.at(i).head(model_robot->nx);
    for (size_t i = 0; i < us_out.size(); i++)
      traj.actions.at(i) = us_out.at(i).head(model_robot->nu);
    traj.start = problem.start;
    traj.goal = problem.goal;
    traj.check(model_robot, true);
    traj.update_feasibility(dynobench::Feasibility_thresholds(), false);
    success = traj.feasible;

  } else if (solver == SOLVER::traj_opt || __free_time_mode) {

    if (solver == SOLVER::traj_opt_free_time_proxi) {
      add_extra_time_rate(us_init);
    }

    if (solver == SOLVER::traj_opt_free_time_proxi_linear) {
      add_extra_time_rate(us_init);
      add_extra_state_time_rate(xs_init, start);
    }

    std::vector<Vxd> regs;
    if (options_trajopt_local.states_reg && solver == SOLVER::traj_opt) {
      double state_reg_weight = 100.;
      regs = std::vector<Vxd>(xs_init.size() - 1,
                              state_reg_weight * Vxd::Ones(_nx));
    }

    Generate_params gen_args{
        .free_time = __free_time_mode,
        .free_time_linear = solver == SOLVER::traj_opt_free_time_proxi_linear,
        .name = name,
        .N = N,
        .goal = goal,
        .start = start,
        .model_robot = model_robot,
        .states = {xs_init.begin(), xs_init.end() - 1},
        .states_weights = regs,
        .actions = us_init,
        .collisions = options_trajopt_local.collision_weight > 1e-3

    };

    // std::cout << "gen problem " << STR_(AT) << std::endl;

    std::vector<Eigen::VectorXd> _xs_out, _us_out, xs_init_p, us_init_p;

    xs_init_p = xs_init;
    us_init_p = us_init;
    const size_t penalty_iterations = 1;
    for (size_t i = 0; i < penalty_iterations; i++) {
      // std::cout << "PENALTY iteration " << i << std::endl;
      gen_args.penalty = std::pow(10., double(i) / 2.);

      if (i > 0) {
        options_trajopt_local.noise_level = 0;
      }

      solve_for_fixed_penalty(gen_args, options_trajopt_local, xs_init, us_init,
                              options_trajopt_local.check_with_finite_diff, N,
                              name, ddp_iterations, ddp_time, _xs_out, _us_out,
                              model_robot, problem, folder_tmptraj,
                              store_iterations, callback_dyno);

      xs_init_p = _xs_out;
      us_init_p = _us_out;
    }

    Trajectory traj;
    traj.start = start;
    traj.goal = goal;
    traj.states.resize(_xs_out.size());
    traj.actions.resize(_us_out.size());

    for (size_t i = 0; i < traj.states.size(); i++)
      traj.states.at(i) = _xs_out.at(i).head(model_robot->nx);

    for (size_t i = 0; i < traj.actions.size(); i++)
      traj.actions.at(i) = _us_out.at(i).head(model_robot->nu);

    if (__free_time_mode) {
      traj.times.resize(_xs_out.size());
      traj.times(0) = 0.;
      for (size_t i = 1; i < static_cast<size_t>(traj.times.size()); i++)
        traj.times(i) = traj.times(i - 1) +
                        _us_out.at(i - 1).tail<1>()(0) * model_robot->ref_dt;
    }

    // std::cout << "CHECK traj with non uniform time " << std::endl;
    traj.check(model_robot, false);
    traj.update_feasibility(dynobench::Feasibility_thresholds(), false);
    std::cout << "CHECK traj with non uniform time -- DONE " << std::endl;

    success = traj.feasible;

    CSTR_(success);

    if (__free_time_mode) {

      Trajectory traj_resample = traj.resample(model_robot);

      for (auto &s : traj_resample.states) {
        model_robot->ensure(s);
      }

      std::cout << "check traj after resample " << std::endl;
      traj_resample.check(model_robot, false);
      traj_resample.update_feasibility(dynobench::Feasibility_thresholds(),
                                       false);

      xs_out = traj_resample.states;
      us_out = traj_resample.actions;
    } else {
      xs_out = _xs_out;
      us_out = _us_out;
    }

    // write out the solution
    // {
    //   debug_file_yaml << "xsOPT: " << std::endl;
    //   for (auto &x : xs_out)
    //     debug_file_yaml << "  - " << x.format(FMT) << std::endl;

    //   debug_file_yaml << "usOPT: " << std::endl;
    //   for (auto &u : us_out)
    //     debug_file_yaml << "  - " << u.format(FMT) << std::endl;
    // }
  }

  // END OF Optimization

  // std::ofstream file_out_debug("/tmp/dynoplan/out.yaml");

  opti_out.success = success;
  // in some s
  // opti_out.feasible = feasible;

  opti_out.xs_out = xs_out;
  opti_out.us_out = us_out;
  opti_out.cost = us_out.size() * dt;
  traj.states = xs_out;
  traj.actions = us_out;

  // TODO: check if this is actually necessary!!
  // if (traj.actions.front().size() > model_robot->nu) {
  //   for (size_t i = 0; i < traj.actions.size(); i++) {
  //     Eigen::VectorXd tmp = traj.actions.at(i).head(model_robot->nu);
  //     traj.actions.at(i) = tmp;
  //   }
  // }
  // if (traj.states.front().size() > model_robot->nx) {
  //   for (size_t i = 0; i < traj.states.size(); i++) {
  //     Eigen::VectorXd tmp = traj.states.at(i).head(model_robot->nx);
  //     traj.states.at(i) = tmp;
  //   }
  // }

  traj.start = problem.start;
  traj.goal = problem.goal;
  traj.cost = traj.actions.size() * model_robot->ref_dt;
  traj.info = "\"ddp_iterations=" + std::to_string(ddp_iterations) +
              ";"
              "ddp_time=" +
              std::to_string(ddp_time) + "\"";

  // traj.to_yaml_format(file_out_debug);

  opti_out.data.insert({"ddp_time", std::to_string(ddp_time)});

  if (opti_out.success) {
    double traj_tol = 1e-2;
    double goal_tol = 1e-1;
    double col_tol = 1e-2;
    double x_bound_tol = 1e-2;
    double u_bound_tol = 1e-2;

    // traj.to_yaml_format(std::cout);

    // std::cout << "Final CHECK" << std::endl;
    // CSTR_(model_robot->name);

    traj.check(model_robot, false);
    std::cout << "Final CHECK -- DONE" << std::endl;

    dynobench::Feasibility_thresholds thresholds{.traj_tol = traj_tol,
                                                 .goal_tol = goal_tol,
                                                 .col_tol = col_tol,
                                                 .x_bound_tol = x_bound_tol,
                                                 .u_bound_tol = u_bound_tol};

    traj.update_feasibility(thresholds);

    opti_out.feasible = traj.feasible;

    if (!traj.feasible) {
      std::cout << "WARNING: "
                << "why first feas and now infeas? (could happen using the "
                   "time proxi) "
                << std::endl;

      if (!__free_time_mode &&
          options_trajopt_local.u_bound_scale <= 1 + 1e-8) {
        // ERROR_WITH_INFO("why?");
        std::cout << "WARNING"
                  << "solver says feasible, but check says infeasible!"
                  << std::endl;
        traj.feasible = false;
        opti_out.feasible = false;
        opti_out.success = false;
      }
    }
  } else {
    traj.feasible = false;
    opti_out.feasible = false;
  }
}

void trajectory_optimization(const dynobench::Problem &problem,
                             const Trajectory &init_guess,
                             const Options_trajopt &options_trajopt,
                             Trajectory &traj, Result_opti &opti_out) {

  double time_ddp_total = 0;
  Stopwatch watch;
  Options_trajopt options_trajopt_local = options_trajopt;
  // std::string _base_path = "../../models/";

  std::shared_ptr<dynobench::Model_robot> model_robot;
    // robotType.empty()) {
  model_robot = dynobench::robot_factory(
      (problem.models_base_path + problem.robotType + ".yaml").c_str(),
      problem.p_lb, problem.p_ub);


  load_env(*model_robot, problem);

  size_t _nx = model_robot->nx; // state
  size_t _nu = model_robot->nu;

  Trajectory tmp_init_guess(init_guess), tmp_solution;


  for (auto &s : tmp_init_guess.states)
    model_robot->ensure(s);

  // CSTR_(model_robot->ref_dt);

  if (!tmp_init_guess.states.size() && tmp_init_guess.num_time_steps == 0) {
    ERROR_WITH_INFO("define either xs_init or num time steps");
  }

  if (!tmp_init_guess.states.size() && !tmp_init_guess.actions.size()) {

    std::cout << "Warning: no xs_init or us_init has been provided. "
              << std::endl;

    tmp_init_guess.states.resize(init_guess.num_time_steps + 1);

    std::for_each(tmp_init_guess.states.begin(), tmp_init_guess.states.end(),
                  [&](auto &x) {
                    if (options_trajopt_local.ref_x0)
                      x = model_robot->get_x0(problem.start);
                    else
                      x = problem.start;
                  });

    tmp_init_guess.actions.resize(tmp_init_guess.states.size() - 1);
    std::for_each(tmp_init_guess.actions.begin(), tmp_init_guess.actions.end(),
                  [&](auto &x) { x = model_robot->u_0; });

    CSTR_V(tmp_init_guess.states.front());
    CSTR_V(tmp_init_guess.actions.front());
  }

  if (tmp_init_guess.states.size() && !tmp_init_guess.actions.size()) {

    std::cout << "Warning: no us_init has been provided -- using u_0: "
              << model_robot->u_0.format(FMT) << std::endl;

    tmp_init_guess.actions.resize(tmp_init_guess.states.size() - 1);

    std::for_each(tmp_init_guess.actions.begin(), tmp_init_guess.actions.end(),
                  [&](auto &x) { x = model_robot->u_0; });
  }


  // check the init guess trajectory

  std::cout << "Report on the init guess " << std::endl;
  WARN_WITH_INFO("should I copy the first state in the init guess? -- now yes");
  tmp_init_guess.start = problem.start;
  // tmp_init_guess.check(model_robot, true);
  std::cout << "Report on the init guess -- DONE " << std::endl;

  switch (static_cast<SOLVER>(options_trajopt.solver_id)) {

  case SOLVER::traj_opt_free_time: {
    // SOLVER 1
    bool do_final_repair_step = true;
    options_trajopt_local.solver_id =
        static_cast<int>(SOLVER::traj_opt_free_time_proxi);
    options_trajopt_local.debug_file_name =
        "/tmp/dynoplan/debug_file_trajopt_freetime_proxi.yaml";

    __trajectory_optimization(problem, model_robot, tmp_init_guess,
                              options_trajopt_local, traj, opti_out);
    time_ddp_total += std::stod(opti_out.data.at("ddp_time"));
    CSTR_(time_ddp_total);

    if (!opti_out.success) {
      std::cout << "warning" << " " << "infeasible" << std::endl;
      do_final_repair_step = false;
    }

    if (do_final_repair_step) {

      std::cout << "time proxi was feasible, doing final step " << std::endl;
      options_trajopt_local.solver_id = static_cast<int>(SOLVER::traj_opt);
      options_trajopt_local.debug_file_name =
          "/tmp/dynoplan/debug_file_trajopt_after_freetime_proxi.yaml";

      __trajectory_optimization(problem, model_robot, tmp_solution,
                                options_trajopt_local, traj, opti_out);

      time_ddp_total += std::stod(opti_out.data.at("ddp_time"));
      CSTR_(time_ddp_total);
    }
    DYNO_CHECK_EQ(traj.feasible, opti_out.feasible, AT);
  } break;

  default: {
    __trajectory_optimization(problem, model_robot, tmp_init_guess,
                              options_trajopt_local, traj, opti_out);
    time_ddp_total += std::stod(opti_out.data.at("ddp_time"));
    CSTR_(time_ddp_total);
    DYNO_CHECK_EQ(traj.feasible, opti_out.feasible, AT);
  }
  }

  // convert the format if necessary

  double time_raw = watch.elapsed_ms();
  opti_out.data.insert({"time_raw", std::to_string(time_raw)});
  opti_out.data.insert({"time_ddp_total", std::to_string(time_ddp_total)});
}

void Result_opti::write_yaml(std::ostream &out) {
  out << "feasible: " << feasible << std::endl;
  out << "success: " << success << std::endl;
  out << "cost: " << cost << std::endl; // TODO: why 2*cost is joint_robot?
  if (data.size()) {
    out << "info:" << std::endl;
    for (const auto &[k, v] : data) {
      out << "  " << k << ": " << v << std::endl;
    }
  }
  // TODO: @QUIM @AKMARAL Clarify this!!!
  out << "result:" << std::endl;
  // out << "xs_out: " << std::endl;
  out << "  - states:" << std::endl;
  for (auto &x : xs_out)
    out << "      - " << x.format(FMT) << std::endl;

  // out << "us_out: " << std::endl;
  out << "    actions:" << std::endl;
  for (auto &u : us_out)
    out << "      - " << u.format(FMT) << std::endl;
}


std::vector<Eigen::VectorXd>
smooth_traj2(const std::vector<Eigen::VectorXd> &xs_init,
             const dynobench::StateDyno &state) {
  size_t n = xs_init.front().size();
  size_t ndx = state.ndx;
  DYNO_CHECK_EQ(n, state.nx, AT);
  std::vector<Vxd> xs_out(xs_init.size(), Eigen::VectorXd::Zero(n));

  // compute diff vectors

  Eigen::VectorXd diffA = Eigen::VectorXd::Zero(ndx);
  Eigen::VectorXd diffB = Eigen::VectorXd::Zero(ndx);
  Eigen::VectorXd diffC = Eigen::VectorXd::Zero(ndx);

  xs_out.front() = xs_init.front();
  xs_out.back() = xs_init.back();
  for (size_t i = 1; i < xs_init.size() - 1; i++) {
    state.diff(xs_init.at(i - 1), xs_init.at(i), diffA);
    state.diff(xs_init.at(i - 1), xs_init.at(i + 1), diffB);

    if (i == xs_init.size() - 2) {
      state.integrate(xs_init.at(i - 1), (diffA + diffB / 2.) / 2.,
                      xs_out.at(i));
    } else {
      state.diff(xs_init.at(i - 1), xs_init.at(i + 2), diffC);
      state.integrate(xs_init.at(i - 1), (diffA + diffB / 2. + diffC / 3.) / 3.,
                      xs_out.at(i));
    }
  }
  // smooth the diffs
  return xs_out;
}

} // namespace dynoplan
