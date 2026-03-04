#include "nmpc_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "croco_models.hpp"
#include "mujoco_quadrotors_payload.hpp"
#include "mujoco_quadrotor.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include <GLFW/glfw3.h>

namespace fs = std::filesystem;

namespace dynoplan {

using dynobench::Trajectory;
using dynobench::Model_MujocoQuad;
using dynobench::Model_MujocoQuadsPayload;

namespace {

void repaint_model_geoms(mjModel* m, float rr, float gg, float bb, float aa) {
  for (int i = 0; i < m->ngeom; ++i) {
    float* c = m->geom_rgba + 4 * i;
    c[0] = rr;
    c[1] = gg;
    c[2] = bb;
    c[3] = aa;
  }
}

void apply_camera_preset(mjvCamera& cam, const Eigen::Vector3d& env_center,
                         const Eigen::Vector3d& env_size, const std::string& camera_view) {
  cam.lookat[0] = env_center.x();
  cam.lookat[1] = env_center.y();
  cam.lookat[2] = env_center.z();

  const double xy_diag = std::sqrt(env_size.x() * env_size.x() + env_size.y() * env_size.y());
  const double base_dist = std::max(1.0, xy_diag * 0.65);

  if (camera_view == "top") {
    cam.azimuth = 0;
    cam.elevation = -90;
    cam.distance = std::max(env_size.x(), env_size.y()) * 0.8;
  } else if (camera_view == "front") {
    cam.azimuth = 180;
    cam.elevation = -15;
    cam.distance = base_dist;
  } else if (camera_view == "side") {
    cam.azimuth = 90;
    cam.elevation = -15;
    cam.distance = base_dist;
  } else if (camera_view == "diag") {
    cam.azimuth = 202.5;
    cam.elevation = -45;
    cam.distance = base_dist;
  } else {
    cam.azimuth = 45;
    cam.elevation = -35;
    cam.distance = base_dist;
  }
}

template <class ModelT>
void render_video_for_view(ModelT* live, ModelT* ghost, const dynobench::Trajectory& sol,
                           const dynobench::Trajectory* ref_traj, const std::string& video_path,
                           const std::string& camera_view, const Eigen::Vector3d& env_center,
                           const Eigen::Vector3d& env_size, int repeats) {
  if (!glfwInit()) throw std::runtime_error("GLFW init failed in maybe_visualize");
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  GLFWwindow* win = glfwCreateWindow(1280, 720, "nmpc_video", nullptr, nullptr);
  if (!win) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window for visualization");
  }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  live->init_mujoco_viewer();
  if (ghost) {
    ghost->init_mujoco_viewer();
    for (int g = 0; g < 6; ++g) ghost->opt_.geomgroup[g] = 0;
    ghost->opt_.geomgroup[1] = 1;
    ghost->opt_.geomgroup[2] = 1;
    ghost->opt_.geomgroup[3] = 1;
    ghost->opt_.geomgroup[5] = 1;
    repaint_model_geoms(ghost->m, 1.f, 0.f, 0.f, 0.5f);
  }

  apply_camera_preset(live->cam_, env_center, env_size, camera_view);

  int w = 0, h = 0;
  glfwGetFramebufferSize(win, &w, &h);
  w = (w / 2) * 2;
  h = (h / 2) * 2;
  std::vector<unsigned char> rgb(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
  std::vector<float> depth(static_cast<size_t>(w) * static_cast<size_t>(h));

  fs::create_directories(fs::path(video_path).parent_path());
  const std::string cmd =
      "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size " + std::to_string(w) + "x" +
      std::to_string(h) +
      " -framerate 60 -i - -vf \"vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
      "-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \"" +
      video_path + "\"";
  FILE* ffmpeg = popen(cmd.c_str(), "w");
  if (!ffmpeg) {
    glfwDestroyWindow(win);
    glfwTerminate();
    throw std::runtime_error("Failed to open FFmpeg pipe");
  }

  const size_t T = sol.states.size();
  for (int r = 0; r < repeats; ++r) {
    for (size_t k = 0; k < T; ++k) {
      live->get_x0(sol.states[k]);
      if (ghost && ref_traj && k < ref_traj->states.size()) {
        ghost->get_x0(ref_traj->states[k]);
      }

      mjv_updateScene(live->m, live->d, &live->opt_, nullptr, &live->cam_, mjCAT_ALL, &live->scn_);
      if (ghost && ref_traj && k < ref_traj->states.size()) {
        mjv_updateScene(ghost->m, ghost->d, &ghost->opt_, nullptr, &live->cam_, mjCAT_ALL, &ghost->scn_);
        mjv_addGeoms(ghost->m, ghost->d, &ghost->opt_, nullptr, mjCAT_ALL, &live->scn_);
      }
      mjr_render({0, 0, w, h}, &live->scn_, &live->con_);
      mjr_readPixels(rgb.data(), depth.data(), {0, 0, w, h}, &live->con_);
      fwrite(rgb.data(), 3, static_cast<size_t>(w) * static_cast<size_t>(h), ffmpeg);
      glfwSwapBuffers(win);
      glfwPollEvents();
    }
  }

  pclose(ffmpeg);
  glfwDestroyWindow(win);
  glfwTerminate();
}

}  // namespace

void NmpcController::build_problem_once() {
  std::map<std::string, double> additional_params;
  dynamics_ = create_dynamics(robot_, Control_Mode::default_mode, additional_params);
  const std::size_t nx = dynamics_->nx;
  const std::size_t nu = dynamics_->nu;

  run_state_costs_.assign(N_, nullptr);
  run_control_costs_.assign(N_, nullptr);
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> running_models;
  running_models.reserve(N_);

  for (std::size_t t = 0; t < N_; ++t) {
    std::vector<ptr<Cost>> feats;

    auto st = mk<State_cost>(nx, nu, nx, Eigen::VectorXd::Constant(nx, 100.0), problem_->goal);
    auto ct = mk<Control_cost>(nx, nu, nu, Eigen::VectorXd::Ones(nu), robot_->u_0);
    run_state_costs_[t] = st;
    run_control_costs_[t] = ct;
    feats.push_back(st);
    feats.push_back(ct);

    if (options_trajopt_.collision_weight > 1e-3 && robot_->env) {
      feats.push_back(mk<Col_cost>(nx, nu, 1, robot_, options_trajopt_.collision_weight));
    }

    if (options_trajopt_.states_reg) {
      auto payload_model =
          std::dynamic_pointer_cast<dynobench::Model_MujocoQuadsPayload>(robot_);
      if (payload_model) {
        feats.push_back(
            mk<State_cost>(nx, nu, nx, payload_model->state_weights, payload_model->state_ref));
      }
      feats.push_back(mk<mujoco_quads_payload_acc>(robot_, robot_->k_acc));
    }

    if (dynamics_->x_lb.size() && dynamics_->x_weightb.sum() > 1e-10) {
      feats.push_back(mk<State_bounds>(nx, nu, nx, dynamics_->x_lb, -dynamics_->x_weightb));
    }
    if (dynamics_->x_ub.size() && dynamics_->x_weightb.sum() > 1e-10) {
      feats.push_back(mk<State_bounds>(nx, nu, nx, dynamics_->x_ub, dynamics_->x_weightb));
    }

    auto am = boost::static_pointer_cast<crocoddyl::ActionModelAbstract>(
        mk<ActionModelDyno>(dynamics_, feats));
    am->set_u_lb(options_trajopt_.u_bound_scale * dynamics_->u_lb);
    am->set_u_ub(options_trajopt_.u_bound_scale * dynamics_->u_ub);
    running_models.push_back(am);
  }

  terminal_goal_cost_ = mk<State_cost>(
      nx, nu, nx, options_trajopt_.weight_goal * robot_->goal_weight, problem_->goal);
  std::vector<ptr<Cost>> terminal_feats{terminal_goal_cost_};
  auto terminal_model = boost::static_pointer_cast<crocoddyl::ActionModelAbstract>(
      mk<ActionModelDyno>(dynamics_, terminal_feats));

  shooting_problem_ = mk<crocoddyl::ShootingProblem>(problem_->start, running_models, terminal_model);
  ddp_solver_ = std::make_unique<crocoddyl::SolverBoxFDDP>(shooting_problem_);
  ddp_solver_->set_th_stop(options_trajopt_.th_stop);
  ddp_solver_->set_th_acceptnegstep(options_trajopt_.th_acceptnegstep);

  xs_ws_.assign(N_ + 1, problem_->start);
  us_ws_.assign(N_, robot_->u_0);
  problem_built_ = true;
}

void NmpcController::prepare_common_warm_start(int k, bool allow_init_bootstrap) {
  const bool can_use_init_bootstrap =
      allow_init_bootstrap && !init_bootstrap_used_ && !init_file_.empty();
  if (can_use_init_bootstrap) {
    slice_window(warm_start_N_, init_guess_, k, N_, robot_->u_0, problem_->goal);
    init_bootstrap_used_ = true;
  } else if (have_valid_window_) {
    warm_start_N_ = shift_and_pad(last_solved_window_, N_, nx_, nu_);
  } else {
    warm_start_N_.num_time_steps = N_;
    warm_start_N_.actions.assign(N_, robot_->u_0);
    warm_start_N_.states.assign(N_ + 1, x_init_);
  }
  ensure_horizon(warm_start_N_, N_, nx_, nu_);
}

void NmpcController::rollout_warm_start_states_from_actions() {
  ensure_horizon(warm_start_N_, N_, nx_, nu_);
  warm_start_N_.states.resize(N_ + 1);
  warm_start_N_.states[0] = x_init_;
  for (std::size_t i = 0; i < N_; ++i) {
    Eigen::VectorXd xn(nx_);
    robot_->step(xn, warm_start_N_.states[i], warm_start_N_.actions[i], robot_->ref_dt);
    warm_start_N_.states[i + 1] = xn;
  }
}

void NmpcController::update_problem_references() {
  if (!problem_built_) {
    throw std::runtime_error("update_problem_references called before build_problem_once");
  }
  shooting_problem_->set_x0(problem_->start);
  const std::size_t nx = dynamics_->nx;
  const std::size_t nu = dynamics_->nu;
  const double w_x_ref = options_trajopt_.ref_state_tracking_weight;
  const double w_u_ref_planner = options_trajopt_.planner_ref_control_tracking_weight;
  const double w_u_ref_policy = options_trajopt_.policy_ref_control_tracking_weight;
  const double w_u_goal = options_trajopt_.goal_control_regularization_weight;
  for (std::size_t t = 0; t < N_; ++t) {
    if (!run_state_costs_[t] || !run_control_costs_[t]) continue;
    if (track_reference_active_ && t < ref_traj_N_.states.size()) {
      run_state_costs_[t]->ref = ref_traj_N_.states[t];
      run_state_costs_[t]->x_weight = Eigen::VectorXd::Constant(nx, w_x_ref);
      Eigen::VectorXd uref = (t < ref_traj_N_.actions.size()) ? ref_traj_N_.actions[t] : robot_->u_0;
      run_control_costs_[t]->set_u_ref(uref);
      const bool policy_ref_mode = (mode_ == NmpcMode::TrackReferencePolicy);
      const double w_u_ref = policy_ref_mode ? w_u_ref_policy : w_u_ref_planner;
      run_control_costs_[t]->set_u_weight(w_u_ref * Eigen::VectorXd::Ones(nu));
    } else {
      run_state_costs_[t]->ref = problem_->goal;
      run_state_costs_[t]->x_weight = Eigen::VectorXd::Constant(nx, w_x_ref);
      run_control_costs_[t]->set_u_ref(robot_->u_0);
      run_control_costs_[t]->set_u_weight(w_u_goal * Eigen::VectorXd::Ones(nu));
    }
  }
  if (terminal_goal_cost_) {
    terminal_goal_cost_->ref = problem_->goal;
    terminal_goal_cost_->x_weight = options_trajopt_.weight_goal * robot_->goal_weight;
  }
}

void NmpcController::solve_with_cached_problem() {
  if (!problem_built_ || !ddp_solver_) {
    throw std::runtime_error("solve_with_cached_problem called before problem setup");
  }
  ensure_horizon(warm_start_N_, N_, nx_, nu_);
  xs_ws_ = warm_start_N_.states;
  us_ws_ = warm_start_N_.actions;
  update_problem_references();
  ddp_solver_->solve(xs_ws_, us_ws_, options_trajopt_.max_iter, false, options_trajopt_.init_reg);
  xs_ws_ = ddp_solver_->get_xs();
  us_ws_ = ddp_solver_->get_us();
  sol_window_.states = xs_ws_;
  sol_window_.actions = us_ws_;
  sol_window_.start = problem_->start;
  sol_window_.goal = problem_->goal;
}

NmpcMode NmpcController::parse_mode(const std::string &mode) {
  if (mode == "track_reference_nmpc_standard") {
    return NmpcMode::TrackReferenceNmpcStandard;
  }
  if (mode == "track_reference_nmpc_refwarm") {
    return NmpcMode::TrackReferenceNmpcRefWarm;
  }
  if (mode == "track_reference_policy") {
    return NmpcMode::TrackReferencePolicy;
  }
  if (mode == "track_linear_hover") {
    return NmpcMode::TrackLinearHover;
  }
  if (mode == "track_goal") {
    return NmpcMode::TrackGoal;
  }
  throw std::runtime_error(
      "Unknown nmpc_mode '" + mode +
      "'. Valid modes: track_goal, track_reference_nmpc_standard, "
      "track_reference_nmpc_refwarm, track_reference_policy, track_linear_hover");
}

void NmpcController::ensure_horizon(Trajectory &traj, std::size_t N,
                                    std::size_t nx, std::size_t nu) {
  if (traj.actions.size() > N) traj.actions.resize(N);
  if (traj.states.size() > N + 1) traj.states.resize(N + 1);

  while (traj.actions.size() < N) {
    Eigen::VectorXd u = traj.actions.empty() ? Eigen::VectorXd::Zero(nu)
                                             : Eigen::VectorXd(traj.actions.back());
    if (u.size() != static_cast<Eigen::Index>(nu)) u.setZero(nu);
    traj.actions.push_back(u);
  }
  while (traj.states.size() < N + 1) {
    Eigen::VectorXd x = traj.states.empty() ? Eigen::VectorXd::Zero(nx)
                                            : Eigen::VectorXd(traj.states.back());
    if (x.size() != static_cast<Eigen::Index>(nx)) x.setZero(nx);
    traj.states.push_back(x);
  }
}

Trajectory NmpcController::shift_and_pad(const Trajectory &solved_window, std::size_t N,
                                         std::size_t nx, std::size_t nu) {
  Trajectory warm;
  if (solved_window.actions.empty() || solved_window.states.size() < 2) {
    warm = solved_window;
    ensure_horizon(warm, N, nx, nu);
    return warm;
  }
  warm.actions.reserve(N);
  for (std::size_t i = 1; i < solved_window.actions.size(); ++i) {
    warm.actions.push_back(Eigen::VectorXd(solved_window.actions[i]));
  }
  warm.states.reserve(N + 1);
  for (std::size_t i = 1; i < solved_window.states.size(); ++i) {
    warm.states.push_back(Eigen::VectorXd(solved_window.states[i]));
  }
  ensure_horizon(warm, N, nx, nu);
  return warm;
}

void NmpcController::slice_window(Trajectory &w, const Trajectory &traj,
                                  std::size_t offset, std::size_t N,
                                  Eigen::VectorXd &u_hover, Eigen::VectorXd &goal) {
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
  if (w.states.size() < N + 1) {
    w.states.resize(N + 1, goal);
  } else if (w.states.size() > N + 1) {
    w.states.resize(N + 1);
  }
}

NmpcController::NmpcController(const std::string &prob_file, const std::string &cfg_file) {
  setup_from_files(prob_file, cfg_file);
}

void NmpcController::setup_from_files(const std::string &prob_file, const std::string &cfg_file) {
  YAML::Node problem_file = YAML::LoadFile(prob_file);
  YAML::Node opt_file = YAML::LoadFile(cfg_file);

  do_optimize_ = problem_file["optimize"] ? problem_file["optimize"].as<bool>() : false;
  do_visualize_ = problem_file["visualize"] ? problem_file["visualize"].as<bool>() : false;
  env_file_ = problem_file["env_file"] ? problem_file["env_file"].as<std::string>() : "";
  init_file_ = problem_file["init_file"] ? problem_file["init_file"].as<std::string>() : "";
  ref_file_ = problem_file["ref_file"] ? problem_file["ref_file"].as<std::string>() : "";
  models_dir_ = problem_file["models_dir"] ? problem_file["models_dir"].as<std::string>() : "";
  results_path_ = problem_file["results_path"] ? problem_file["results_path"].as<std::string>()
                                               : "../result_opt.yaml";
  view_ref_ = problem_file["view_ref"] ? problem_file["view_ref"].as<bool>() : false;
  video_prefix_ = problem_file["video_prefix"] ? problem_file["video_prefix"].as<std::string>() : "";
  repeats_ = problem_file["repeats"] ? problem_file["repeats"].as<int>() : 1;
  views_ = problem_file["views"] ? problem_file["views"].as<std::vector<std::string>>()
                                 : std::vector<std::string>{"auto"};

  use_policy_onnx_ = opt_file["use_policy_onnx"] ? opt_file["use_policy_onnx"].as<bool>() : false;
  policy_onnx_path_ = opt_file["policy_onnx_path"] ? opt_file["policy_onnx_path"].as<std::string>() : "";
  policy_rollout_as_ref_ = opt_file["policy_rollout_as_ref"] ? opt_file["policy_rollout_as_ref"].as<bool>() : true;
  policy_u_clip_min_ = opt_file["policy_u_clip_min"] ? opt_file["policy_u_clip_min"].as<double>() : -1e30;
  policy_u_clip_max_ = opt_file["policy_u_clip_max"] ? opt_file["policy_u_clip_max"].as<double>() : 1e30;
  policy_threads_ = opt_file["policy_threads"] ? opt_file["policy_threads"].as<int>() : 1;
  control_noise_ = opt_file["control_noise"] ? opt_file["control_noise"].as<double>() : 1e-3;
  fail_threshold_ = opt_file["fail_threshold"] ? opt_file["fail_threshold"].as<double>() : 5.0;
  goal_tol_ = opt_file["goal_tol"] ? opt_file["goal_tol"].as<double>() : 0.05;
  N_ = static_cast<std::size_t>(opt_file["N"].as<int>());
  max_steps_ = static_cast<std::size_t>(opt_file["max_steps"].as<int>());

  options_trajopt_.read_from_yaml(opt_file);
  mode_ = parse_mode(options_trajopt_.nmpc_mode);
  solve_every_k_steps_ = std::max<std::size_t>(1, options_trajopt_.solve_every_k_steps);
  disturbance_enable_ = options_trajopt_.disturbance_enable;
  disturbance_start_s_ = options_trajopt_.disturbance_start_s;
  disturbance_duration_s_ = options_trajopt_.disturbance_duration_s;
  disturbance_force_ = Eigen::Vector3d(options_trajopt_.disturbance_force_x,
                                       options_trajopt_.disturbance_force_y,
                                       options_trajopt_.disturbance_force_z);
  disturbance_payload_mass_ = std::max(1e-6, options_trajopt_.disturbance_payload_mass);

  if (!do_optimize_ && !do_visualize_) {
    throw std::runtime_error("Nothing to do: both optimize and visualize are false");
  }
  if (env_file_.empty()) {
    throw std::runtime_error("env_file is required in prob_file");
  }

  fs::path build_dir = fs::current_path();
  fs::path models_path_cfg(models_dir_);
  if (models_path_cfg.is_absolute()) {
    models_dir_abs_ = models_path_cfg.string();
  } else {
    std::string models_dir_path = build_dir.parent_path().string() + "/" + models_dir_ + "/";
    models_dir_abs_ = fs::absolute(models_dir_path).string();
  }
  if (!models_dir_abs_.empty() && models_dir_abs_.back() != '/') {
    models_dir_abs_ += "/";
  }

  problem_ = std::make_unique<dynobench::Problem>(env_file_.c_str());
  problem_->models_base_path = models_dir_abs_;

  robot_ = dynobench::robot_factory((models_dir_abs_ + problem_->robotType + ".yaml").c_str(),
                                    problem_->p_lb, problem_->p_ub);
  if (!robot_) {
    throw std::runtime_error("Failed to create robot model for type: " + problem_->robotType);
  }
  load_env(*robot_, *problem_);

  nx_ = static_cast<std::size_t>(robot_->nx);
  nu_ = static_cast<std::size_t>(robot_->nu);
  u_prev_exec_ = Eigen::VectorXd::Zero(nu_);

  if (use_policy_onnx_ || mode_ == NmpcMode::TrackReferencePolicy) {
    if (policy_onnx_path_.empty()) {
      throw std::runtime_error("track_reference_policy mode requires policy_onnx_path");
    }
    policy_onnx_ = std::make_unique<PolicyOnnx>(policy_onnx_path_, policy_threads_);
  }

  if (!init_file_.empty()) {
    init_guess_.read_from_yaml(init_file_.c_str());
  }
  if (!ref_file_.empty()) {
    ref_traj_.read_from_yaml(ref_file_.c_str());
  }

  build_problem_once();
}

void NmpcController::prepare_window_for_step(int k) {
  switch (mode_) {
    case NmpcMode::TrackReferencePolicy: {
      // New chunk policy contract: output is [H*nu].
      Eigen::VectorXd flat =
          policy_onnx_->predict_chunk(x_init_, u_prev_exec_, static_cast<int>(N_), static_cast<int>(nu_));
      if (flat.size() != static_cast<Eigen::Index>(N_ * nu_)) {
        throw std::runtime_error("PolicyChunk: ONNX output size mismatch, expected H*nu");
      }
      warm_start_N_.actions.resize(N_);
      for (std::size_t i = 0; i < N_; ++i) {
        Eigen::VectorXd u = flat.segment(i * nu_, nu_);
        for (std::size_t j = 0; j < nu_; ++j) {
          u(static_cast<Eigen::Index>(j)) =
              std::min(std::max(u(static_cast<Eigen::Index>(j)), policy_u_clip_min_), policy_u_clip_max_);
        }
        warm_start_N_.actions[i] = u;
      }
      // Rollout warm-start states from measured x_init_ with policy controls.
      rollout_warm_start_states_from_actions();
      ref_traj_N_ = warm_start_N_;
      track_reference_active_ = true;
      break;
    }
    case NmpcMode::TrackReferenceNmpcRefWarm: {
      if (ref_file_.empty()) {
        throw std::runtime_error("track_reference_nmpc_refwarm mode requires non-empty ref_file");
      }
      slice_window(ref_traj_N_, ref_traj_, k, N_, robot_->u_0, problem_->goal);
      warm_start_N_ = ref_traj_N_;
      ensure_horizon(warm_start_N_, N_, nx_, nu_);
      track_reference_active_ = true;
      break;
    }
    case NmpcMode::TrackReferenceNmpcStandard: {
      prepare_common_warm_start(k, true);
      if (!ref_file_.empty()) {
        slice_window(ref_traj_N_, ref_traj_, k, N_, robot_->u_0, problem_->goal);
        track_reference_active_ = true;
      } else {
        throw std::runtime_error("track_reference_nmpc_standard mode requires non-empty ref_file");
      }
      break;
    }
    case NmpcMode::TrackLinearHover:
    case NmpcMode::TrackGoal: {
      prepare_common_warm_start(k, true);
      ref_traj_N_.num_time_steps = N_;
      ref_traj_N_.states.assign(N_ + 1, problem_->goal);
      ref_traj_N_.actions.assign(N_, robot_->u_0);
      track_reference_active_ = false;
      break;
    }
  }
}

void NmpcController::apply_payload_disturbance(Eigen::VectorXd &xnext, int k) const {
  if (!disturbance_enable_) return;
  const double t = static_cast<double>(k) * robot_->ref_dt;
  if (t < disturbance_start_s_ || t >= disturbance_start_s_ + disturbance_duration_s_) {
    return;
  }
  if (xnext.size() < 24) return;
  Eigen::Vector3d dv = (disturbance_force_ / disturbance_payload_mass_) * robot_->ref_dt;
  xnext.segment<3>(21) += dv;
}

NmpcStepInfo NmpcController::step_once(int k) {
  NmpcStepInfo info;
  const auto step_t0 = std::chrono::steady_clock::now();
  const bool need_solve =
      (!have_valid_window_) || (planned_window_idx_ >= last_solved_window_.actions.size()) ||
      (static_cast<std::size_t>(k) % solve_every_k_steps_ == 0);
  if (need_solve) {
    const auto solve_t0 = std::chrono::steady_clock::now();
    prepare_window_for_step(k);
    problem_->start = x_init_;
    solve_with_cached_problem();
    const auto solve_t1 = std::chrono::steady_clock::now();
    last_solved_window_ = sol_window_;
    planned_window_idx_ = 0;
    have_valid_window_ = !last_solved_window_.actions.empty();
    info.did_solve = true;
    info.solve_time_ms =
        std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
  }
  if (!have_valid_window_) {
    throw std::runtime_error("No valid control window available after solve");
  }

  Eigen::VectorXd u = last_solved_window_.actions.at(planned_window_idx_);
  planned_window_idx_++;

  const double umax = 1.4;
  u += 0.5 * control_noise_ * umax * (Eigen::VectorXd::Random(nu_).array() + 1.0).matrix();

  Eigen::VectorXd xnext(nx_);
  robot_->step(xnext, x_init_.head(nx_), u.head(nu_), robot_->ref_dt);
  apply_payload_disturbance(xnext, k);

  sol_.states.push_back(xnext);
  sol_.actions.push_back(u);
  u_prev_exec_ = u;
  x_init_ = xnext;

  info.goal_distance = robot_->distance(sol_.states.back(), problem_->goal);
  if (info.goal_distance <= goal_tol_) {
    if (k_goal_ == 0 && k != 0) {
      k_goal_ = k;
    }
    info.reached_goal = true;
  } else if (info.goal_distance > fail_threshold_) {
    info.failed = true;
  }
  const auto step_t1 = std::chrono::steady_clock::now();
  info.step_time_ms = std::chrono::duration<double, std::milli>(step_t1 - step_t0).count();
  return info;
}

void NmpcController::run() {
  sol_.states.clear();
  sol_.actions.clear();
  sol_.states.push_back(problem_->start);
  x_init_ = problem_->start;
  k_goal_ = 0;
  have_valid_window_ = false;
  planned_window_idx_ = 0;
  init_bootstrap_used_ = false;
  std::size_t executed_steps = 0;
  std::size_t solve_count = 0;
  double total_step_time_ms = 0.0;
  double total_solve_time_ms = 0.0;
  double max_step_time_ms = 0.0;
  double max_solve_time_ms = 0.0;
  std::vector<double> step_hz_samples;
  std::vector<double> solve_hz_samples;
  step_hz_samples.reserve(max_steps_);
  solve_hz_samples.reserve(max_steps_);
  const auto run_t0 = std::chrono::steady_clock::now();

  if (!do_optimize_) return;
  for (int k = 0; k < static_cast<int>(max_steps_); ++k) {
    std::cout << "=== NMPC Step " << k << " ===\n";
    NmpcStepInfo info = step_once(k);
    executed_steps++;
    total_step_time_ms += info.step_time_ms;
    max_step_time_ms = std::max(max_step_time_ms, info.step_time_ms);
    if (info.step_time_ms > 0.0) {
      step_hz_samples.push_back(1000.0 / info.step_time_ms);
    }
    if (info.did_solve) {
      solve_count++;
      total_solve_time_ms += info.solve_time_ms;
      max_solve_time_ms = std::max(max_solve_time_ms, info.solve_time_ms);
      if (info.solve_time_ms > 0.0) {
        solve_hz_samples.push_back(1000.0 / info.solve_time_ms);
      }
    }
    if (info.reached_goal) {
      std::cout << "Goal distance: " << info.goal_distance << std::endl;
    }
    if (info.failed) {
      std::cout << "Tracking failed: distance (" << info.goal_distance
                << ") exceeded threshold (" << fail_threshold_ << ") at step " << k << "\n";
      break;
    }
    if (!info.reached_goal) {
      std::cout << "Goal distance: " << info.goal_distance << std::endl;
    }
  }

  if (max_steps_ < 2 && !sol_window_.states.empty()) {
    sol_ = sol_window_;
  }
  dynobench::Problem problem_final(env_file_.c_str());
  sol_.cost = sol_.actions.size() * robot_->ref_dt;
  sol_.start = problem_final.start;
  sol_.goal = problem_final.goal;
  sol_.check(robot_, true);
  if (!results_path_.empty()) {
    sol_.to_yaml_format(results_path_.c_str());
    std::cout << "Saved NMPC stitched trajectory to: " << results_path_ << "\n";
  }

  const auto run_t1 = std::chrono::steady_clock::now();
  const double run_wall_time_ms =
      std::chrono::duration<double, std::milli>(run_t1 - run_t0).count();
  const double mean_step_time_ms =
      executed_steps ? (total_step_time_ms / static_cast<double>(executed_steps)) : 0.0;
  const double mean_solve_time_ms =
      solve_count ? (total_solve_time_ms / static_cast<double>(solve_count)) : 0.0;
  const double mean_step_hz = (mean_step_time_ms > 0.0) ? (1000.0 / mean_step_time_ms) : 0.0;
  const double mean_solve_hz = (mean_solve_time_ms > 0.0) ? (1000.0 / mean_solve_time_ms) : 0.0;
  const double run_hz = (run_wall_time_ms > 0.0)
                            ? (1000.0 * static_cast<double>(executed_steps) / run_wall_time_ms)
                            : 0.0;
  auto mean_std = [](const std::vector<double>& v) -> std::pair<double, double> {
    if (v.empty()) return {0.0, 0.0};
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (double x : v) {
      const double d = x - mean;
      var += d * d;
    }
    var /= static_cast<double>(v.size());
    return {mean, std::sqrt(var)};
  };
  const auto [step_hz_mean_samples, step_hz_std_samples] = mean_std(step_hz_samples);
  const auto [solve_hz_mean_samples, solve_hz_std_samples] = mean_std(solve_hz_samples);

  fs::path timing_path = "nmpc_timing.json";
  if (!results_path_.empty()) {
    fs::path rp(results_path_);
    timing_path = rp.parent_path() / (rp.stem().string() + "_timing.json");
  }
  if (timing_path.has_parent_path()) {
    fs::create_directories(timing_path.parent_path());
  }
  std::ofstream timing_out(timing_path);
  if (timing_out) {
    timing_out << "{\n"
               << "  \"mode\": \"" << options_trajopt_.nmpc_mode << "\",\n"
               << "  \"solve_every_k_steps\": " << solve_every_k_steps_ << ",\n"
               << "  \"max_steps_configured\": " << max_steps_ << ",\n"
               << "  \"steps_executed\": " << executed_steps << ",\n"
               << "  \"solve_count\": " << solve_count << ",\n"
               << "  \"run_wall_time_ms\": " << run_wall_time_ms << ",\n"
               << "  \"total_step_time_ms\": " << total_step_time_ms << ",\n"
               << "  \"mean_step_time_ms\": " << mean_step_time_ms << ",\n"
               << "  \"mean_step_hz\": " << mean_step_hz << ",\n"
               << "  \"mean_step_hz_samples\": " << step_hz_mean_samples << ",\n"
               << "  \"std_step_hz_samples\": " << step_hz_std_samples << ",\n"
               << "  \"max_step_time_ms\": " << max_step_time_ms << ",\n"
               << "  \"total_solve_time_ms\": " << total_solve_time_ms << ",\n"
               << "  \"mean_solve_time_ms\": " << mean_solve_time_ms << ",\n"
               << "  \"mean_solve_hz\": " << mean_solve_hz << ",\n"
               << "  \"mean_solve_hz_samples\": " << solve_hz_mean_samples << ",\n"
               << "  \"std_solve_hz_samples\": " << solve_hz_std_samples << ",\n"
               << "  \"max_solve_time_ms\": " << max_solve_time_ms << ",\n"
               << "  \"run_hz\": " << run_hz << "\n"
               << "}\n";
    std::cout << "Saved NMPC timing summary to: " << timing_path.string() << "\n";
  }
}

void NmpcController::maybe_visualize() {
  if (!do_visualize_) return;
  if (sol_.states.empty()) {
    std::cout << "[nmpc] skip visualize: solution trajectory is empty\n";
    return;
  }
  if (video_prefix_.empty()) {
    std::cout << "[nmpc] skip visualize: video_prefix is empty\n";
    return;
  }

  YAML::Node env = YAML::LoadFile(env_file_);
  auto maxNode = env["environment"]["max"];
  auto minNode = env["environment"]["min"];
  Eigen::Vector3d env_max(maxNode[0].as<double>(), maxNode[1].as<double>(), maxNode[2].as<double>());
  Eigen::Vector3d env_min(minNode[0].as<double>(), minNode[1].as<double>(), minNode[2].as<double>());
  auto startNode = env["robots"][0]["start"];
  auto goalNode = env["robots"][0]["goal"];
  Eigen::Vector3d start_pos(startNode[0].as<double>(), startNode[1].as<double>(), startNode[2].as<double>());
  Eigen::Vector3d goal_pos(goalNode[0].as<double>(), goalNode[1].as<double>(), goalNode[2].as<double>());
  env_min = env_min.cwiseMin(start_pos).cwiseMin(goal_pos);
  env_max = env_max.cwiseMax(start_pos).cwiseMax(goal_pos);
  const Eigen::Vector3d env_center = 0.5 * (env_max + env_min);
  const Eigen::Vector3d env_size = env_max - env_min;

  std::vector<std::string> local_views = views_;
  if (local_views.empty()) local_views = {"auto"};
  if (local_views.size() == 1 && local_views[0] == "auto") {
    local_views = {"side", "top", "front", "diag"};
  }

  const dynobench::Trajectory* maybe_ref = (view_ref_ && !ref_traj_.states.empty()) ? &ref_traj_ : nullptr;
  const int local_repeats = std::max(1, repeats_);
  const fs::path base(video_prefix_);
  fs::create_directories(base.parent_path());

  auto* payload_live = dynamic_cast<Model_MujocoQuadsPayload*>(robot_.get());
  auto* quad_live = dynamic_cast<Model_MujocoQuad*>(robot_.get());

  try {
    if (payload_live) {
    std::shared_ptr<dynobench::Model_robot> ghost_base;
    Model_MujocoQuadsPayload* payload_ghost = nullptr;
    if (maybe_ref) {
      ghost_base = dynobench::robot_factory(
          (models_dir_abs_ + problem_->robotType + ".yaml").c_str(), problem_->p_lb, problem_->p_ub);
      payload_ghost = dynamic_cast<Model_MujocoQuadsPayload*>(ghost_base.get());
      if (!payload_ghost) {
        throw std::runtime_error("Failed to cast ghost model to Model_MujocoQuadsPayload");
      }
    }
    for (const auto& v : local_views) {
      std::string out = (base.string() + "_" + v + ".mp4");
      render_video_for_view(payload_live, payload_ghost, sol_, maybe_ref, out, v, env_center, env_size,
                            local_repeats);
      std::cout << "[nmpc] wrote video: " << out << "\n";
    }
    return;
  }

    if (quad_live) {
    std::shared_ptr<dynobench::Model_robot> ghost_base;
    Model_MujocoQuad* quad_ghost = nullptr;
    if (maybe_ref) {
      ghost_base = dynobench::robot_factory(
          (models_dir_abs_ + problem_->robotType + ".yaml").c_str(), problem_->p_lb, problem_->p_ub);
      quad_ghost = dynamic_cast<Model_MujocoQuad*>(ghost_base.get());
      if (!quad_ghost) {
        throw std::runtime_error("Failed to cast ghost model to Model_MujocoQuad");
      }
    }
    for (const auto& v : local_views) {
      std::string out = (base.string() + "_" + v + ".mp4");
      render_video_for_view(quad_live, quad_ghost, sol_, maybe_ref, out, v, env_center, env_size, local_repeats);
      std::cout << "[nmpc] wrote video: " << out << "\n";
    }
    return;
  }

    std::cout << "[nmpc] visualize unsupported robot type: " << problem_->robotType << "\n";
  } catch (const std::exception& e) {
    std::cout << "[nmpc] visualize failed: " << e.what() << "\n";
  }
}

}  // namespace dynoplan
