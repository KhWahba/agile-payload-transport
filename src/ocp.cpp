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

  // warmstart
  std::vector<Vxd> xs, us;
  if (options_trajopt_local.use_warmstart) {
    xs = xs_init;
    us = us_init;
  } else {
    xs = std::vector<Eigen::VectorXd>(N + 1, gen_args.start);
    Eigen::VectorXd u0 = Vxd(nu);
    u0 << model_robot->u_0;
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

  std::vector<SOLVER> solvers{SOLVER::traj_opt};

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

  bool check_with_finite_diff = false;
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
  
  if (solver == SOLVER::traj_opt) {

    std::vector<Vxd> regs;
    if (options_trajopt_local.states_reg && solver == SOLVER::traj_opt) {
      double state_reg_weight = 100.;
      regs = std::vector<Vxd>(xs_init.size() - 1,
                              state_reg_weight * Vxd::Ones(_nx));
    }

    Generate_params gen_args{
        .free_time = false,
        .free_time_linear = false,
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

    // std::cout << "CHECK traj with non uniform time " << std::endl;
    traj.check(model_robot, false);
    traj.update_feasibility(dynobench::Feasibility_thresholds(), false);
    std::cout << "CHECK traj with non uniform time -- DONE " << std::endl;

    success = traj.feasible;

    CSTR_(success);
    xs_out = _xs_out;
    us_out = _us_out;
  }

  // END OF Optimization


  opti_out.success = success;

  opti_out.xs_out = xs_out;
  opti_out.us_out = us_out;
  opti_out.cost = us_out.size() * dt;
  traj.states = xs_out;
  traj.actions = us_out;


  traj.start = problem.start;
  traj.goal = problem.goal;
  traj.cost = traj.actions.size() * model_robot->ref_dt;
  traj.info = "\"ddp_iterations=" + std::to_string(ddp_iterations) +
              ";"
              "ddp_time=" +
              std::to_string(ddp_time) + "\"";


  opti_out.data.insert({"ddp_time", std::to_string(ddp_time)});

  if (opti_out.success) {
    double traj_tol = 1e-2;
    double goal_tol = 1e-1;
    double col_tol = 1e-2;
    double x_bound_tol = 1e-2;
    double u_bound_tol = 1e-2;

    traj.check(model_robot, false);
    std::cout << "Final CHECK -- DONE" << std::endl;

    dynobench::Feasibility_thresholds thresholds{.traj_tol = traj_tol,
                                                 .goal_tol = goal_tol,
                                                 .col_tol = col_tol,
                                                 .x_bound_tol = x_bound_tol,
                                                 .u_bound_tol = u_bound_tol};

    traj.update_feasibility(thresholds);

    opti_out.feasible = traj.feasible;
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

    case SOLVER::traj_opt: {
      // SOLVER 1
      options_trajopt_local.debug_file_name =
          "/tmp/dynoplan/debug_file_trajopt.yaml";

      __trajectory_optimization(problem, model_robot, tmp_init_guess,
                                options_trajopt_local, traj, opti_out);
      time_ddp_total += std::stod(opti_out.data.at("ddp_time"));
      CSTR_(time_ddp_total);

      if (!opti_out.success) {
        std::cout << "warning" << " " << "infeasible" << std::endl;
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
