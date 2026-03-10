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
#include <sstream>
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

fs::path resolve_with_base(const fs::path &base, const std::string &value) {
  if (value.empty()) return {};
  fs::path p(value);
  if (p.is_absolute()) return p;
  return fs::weakly_canonical(base / p);
}

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

Eigen::Vector4d quat_xyzw_mul(const Eigen::Vector4d &q1, const Eigen::Vector4d &q2) {
  const double x1 = q1(0), y1 = q1(1), z1 = q1(2), w1 = q1(3);
  const double x2 = q2(0), y2 = q2(1), z2 = q2(2), w2 = q2(3);
  Eigen::Vector4d out;
  out << (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2),
      (w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2),
      (w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2),
      (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2);
  return out;
}

Eigen::Vector4d quat_xyzw_inv(const Eigen::Vector4d &q) {
  const double n2 = q.squaredNorm();
  if (n2 <= 1e-12) {
    return Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  }
  return Eigen::Vector4d(-q(0), -q(1), -q(2), q(3)) / n2;
}

void json_write_vec(std::ostream &os, const Eigen::VectorXd &v) {
  os << "[";
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    if (i) os << ",";
    os << v(i);
  }
  os << "]";
}

void json_write_traj_vec(std::ostream &os, const std::vector<Eigen::VectorXd> &xs) {
  os << "[";
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i) os << ",";
    json_write_vec(os, xs[i]);
  }
  os << "]";
}

void json_write_payload_pv(std::ostream &os, const std::vector<Eigen::VectorXd> &xs) {
  os << "[";
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i) os << ",";
    const auto &x = xs[i];
    os << "[";
    if (x.size() >= 24) {
      os << x(0) << "," << x(1) << "," << x(2) << ","
         << x(21) << "," << x(22) << "," << x(23);
    } else if (x.size() >= 3) {
      os << x(0) << "," << x(1) << "," << x(2) << ",0,0,0";
    } else {
      os << "0,0,0,0,0,0";
    }
    os << "]";
  }
  os << "]";
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

  // Precompute per-dimension running state weight vector.
  if (options_trajopt_.running_cost_goal_weight_mask) {
    running_state_weight_vec_ = robot_->goal_weight;
  } else {
    running_state_weight_vec_ = Eigen::VectorXd::Ones(nx);
  }

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

Eigen::VectorXd NmpcController::build_policy_features(const Eigen::VectorXd &state) const {
  // Matches scripts/payload_env.py::_get_learner_features exactly:
  // [payload_pos_err(3), payload_vel(3), per-quad(12), prev_action(nu)]
  if (n_bodies_ == 0 || n_quads_ + 1 != n_bodies_) {
    throw std::runtime_error("build_policy_features: invalid body/quad metadata");
  }
  if (state.size() != static_cast<Eigen::Index>(nx_)) {
    throw std::runtime_error("build_policy_features: state has wrong size");
  }
  if (problem_->goal.size() != static_cast<Eigen::Index>(nx_)) {
    throw std::runtime_error("build_policy_features: goal has wrong size");
  }

  const Eigen::VectorXd &goal = problem_->goal;
  const std::size_t n = n_bodies_;
  const std::size_t poses_dim = 7 * n;
  const std::size_t vels_dim = 6 * n;
  if (state.size() != static_cast<Eigen::Index>(poses_dim + vels_dim)) {
    throw std::runtime_error("build_policy_features: expected poses+vels layout (13*n)");
  }

  std::vector<double> feat;
  feat.reserve(6 + n_quads_ * 12 + nu_);

  const auto pL = state.segment<3>(0);
  const auto pL_goal = goal.segment<3>(0);
  const auto vL = state.segment<3>(static_cast<Eigen::Index>(poses_dim + 0));

  // payload position error and payload linear velocity
  for (int i = 0; i < 3; ++i) feat.push_back(pL(i) - pL_goal(i));
  for (int i = 0; i < 3; ++i) feat.push_back(vL(i));

  // Goal velocities are zero by design in Python learner features
  for (std::size_t qi = 0; qi < n_quads_; ++qi) {
    const std::size_t body = 1 + qi;
    const std::size_t pose_base = 7 * body;
    const std::size_t vel_base = 6 * body;

    const auto p_i = state.segment<3>(static_cast<Eigen::Index>(pose_base + 0));
    const auto q_i = state.segment<4>(static_cast<Eigen::Index>(pose_base + 3));   // xyzw
    const auto v_i = state.segment<3>(static_cast<Eigen::Index>(poses_dim + vel_base + 0));
    const auto w_i = state.segment<3>(static_cast<Eigen::Index>(poses_dim + vel_base + 3));

    const auto p_i_goal = goal.segment<3>(static_cast<Eigen::Index>(pose_base + 0));
    const auto q_i_goal = goal.segment<4>(static_cast<Eigen::Index>(pose_base + 3)); // xyzw

    const Eigen::Vector3d e_p_rel = (p_i - pL) - (p_i_goal - pL_goal);
    const Eigen::Vector3d e_v_rel = (v_i - vL); // goal-relative velocities are zero

    Eigen::Vector4d q_err = quat_xyzw_mul(quat_xyzw_inv(q_i_goal), q_i);
    if (q_err(3) < 0.0) q_err = -q_err;
    const Eigen::Vector3d e_q = q_err.head<3>();
    const Eigen::Vector3d e_w = w_i; // goal angular velocities are zero

    for (int i = 0; i < 3; ++i) feat.push_back(e_p_rel(i));
    for (int i = 0; i < 3; ++i) feat.push_back(e_v_rel(i));
    for (int i = 0; i < 3; ++i) feat.push_back(e_q(i));
    for (int i = 0; i < 3; ++i) feat.push_back(e_w(i));
  }

  if (prev_action_policy_.size() != static_cast<Eigen::Index>(nu_)) {
    throw std::runtime_error("build_policy_features: prev_action_policy_ has wrong size");
  }
  for (std::size_t i = 0; i < nu_; ++i) feat.push_back(prev_action_policy_(static_cast<Eigen::Index>(i)));

  Eigen::VectorXd out(static_cast<Eigen::Index>(feat.size()));
  for (std::size_t i = 0; i < feat.size(); ++i) out(static_cast<Eigen::Index>(i)) = feat[i];
  return out;
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
  const Eigen::VectorXd w_x_vec = w_x_ref * running_state_weight_vec_;
  for (std::size_t t = 0; t < N_; ++t) {
    if (!run_state_costs_[t] || !run_control_costs_[t]) continue;
    if (track_reference_active_ && t < ref_traj_N_.states.size()) {
      run_state_costs_[t]->ref = ref_traj_N_.states[t];
      run_state_costs_[t]->x_weight = w_x_vec;
      Eigen::VectorXd uref = (t < ref_traj_N_.actions.size()) ? ref_traj_N_.actions[t] : robot_->u_0;
      run_control_costs_[t]->set_u_ref(uref);
      const bool policy_ref_mode = (mode_ == NmpcMode::TrackReferencePolicy);
      const double w_u_ref = policy_ref_mode ? w_u_ref_policy : w_u_ref_planner;
      run_control_costs_[t]->set_u_weight(w_u_ref * Eigen::VectorXd::Ones(nu));
    } else {
      run_state_costs_[t]->ref = problem_->goal;
      run_state_costs_[t]->x_weight = w_x_vec;
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
  if (mode == "track_policy_warmstart_goal") {
    return NmpcMode::TrackPolicyWarmstartGoal;
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
      "track_reference_nmpc_refwarm, track_reference_policy, "
      "track_policy_warmstart_goal, track_linear_hover");
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

void NmpcController::run_with_overrides(const std::string &mode,
                                        const std::string &out_yaml,
                                        const std::string &out_timing_json) {
  if (!mode.empty()) {
    mode_ = parse_mode(mode);
    options_trajopt_.nmpc_mode = mode;
  }
  if (!out_yaml.empty()) {
    results_path_ = out_yaml;
  }
  timing_output_path_override_ = out_timing_json;
  run();
}

void NmpcController::setup_from_files(const std::string &prob_file, const std::string &cfg_file) {
  const fs::path prob_path = fs::absolute(prob_file);
  const fs::path cfg_path = fs::absolute(cfg_file);
  const fs::path prob_dir = prob_path.parent_path();
  const fs::path cfg_dir = cfg_path.parent_path();

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
  policy_u_clip_min_ = opt_file["policy_u_clip_min"] ? opt_file["policy_u_clip_min"].as<double>() : 0.0;
  policy_u_clip_max_ = opt_file["policy_u_clip_max"] ? opt_file["policy_u_clip_max"].as<double>() : 1.4;
  policy_threads_ = opt_file["policy_threads"] ? opt_file["policy_threads"].as<int>() : 1;
  planner_act_low_ = opt_file["planner_act_low"] ? opt_file["planner_act_low"].as<double>() : 0.0;
  planner_act_high_ = opt_file["planner_act_high"] ? opt_file["planner_act_high"].as<double>() : 1.4;
  debug_policy_loop_ = opt_file["debug_policy_loop"] ? opt_file["debug_policy_loop"].as<bool>() : false;
  debug_policy_path_ = opt_file["debug_policy_path"] ? opt_file["debug_policy_path"].as<std::string>() : "";
  control_noise_ = opt_file["control_noise"] ? opt_file["control_noise"].as<double>() : 1e-3;
  fail_threshold_ = opt_file["fail_threshold"] ? opt_file["fail_threshold"].as<double>() : 5.0;
  goal_tol_ = opt_file["goal_tol"] ? opt_file["goal_tol"].as<double>() : 0.05;
  N_ = static_cast<std::size_t>(opt_file["N"].as<int>());
  max_steps_ = static_cast<std::size_t>(opt_file["max_steps"].as<int>());

  options_trajopt_.read_from_yaml(opt_file);
  mode_ = parse_mode(options_trajopt_.nmpc_mode);
  if (mode_ == NmpcMode::TrackReferencePolicy && !policy_rollout_as_ref_) {
    // Backward-compatibility path for older configs that selected the policy
    // mode and then disabled reference tracking with a separate flag.
    mode_ = NmpcMode::TrackPolicyWarmstartGoal;
    options_trajopt_.nmpc_mode = "track_policy_warmstart_goal";
  }
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
  env_file_ = resolve_with_base(prob_dir, env_file_).string();
  if (!init_file_.empty()) init_file_ = resolve_with_base(prob_dir, init_file_).string();
  if (!ref_file_.empty()) ref_file_ = resolve_with_base(prob_dir, ref_file_).string();
  if (!results_path_.empty()) results_path_ = resolve_with_base(prob_dir, results_path_).string();
  if (!video_prefix_.empty()) video_prefix_ = resolve_with_base(prob_dir, video_prefix_).string();
  if (!policy_onnx_path_.empty()) policy_onnx_path_ = resolve_with_base(cfg_dir, policy_onnx_path_).string();
  if (!debug_policy_path_.empty()) debug_policy_path_ = resolve_with_base(cfg_dir, debug_policy_path_).string();

  fs::path build_dir = fs::current_path();
  fs::path models_path_cfg(models_dir_);
  if (models_path_cfg.is_absolute()) {
    models_dir_abs_ = models_path_cfg.string();
  } else {
    const fs::path from_prob = prob_dir / models_path_cfg;
    if (fs::exists(from_prob)) {
      models_dir_abs_ = fs::weakly_canonical(from_prob).string();
    } else {
      std::string models_dir_path = build_dir.parent_path().string() + "/" + models_dir_ + "/";
      models_dir_abs_ = fs::absolute(models_dir_path).string();
    }
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
  n_bodies_ = nx_ / 13;
  n_quads_ = (n_bodies_ > 0) ? (n_bodies_ - 1) : 0;
  u_prev_exec_ = Eigen::VectorXd::Zero(nu_);
  prev_action_policy_ = Eigen::VectorXd::Zero(nu_);
  act_mid_ = Eigen::VectorXd::Constant(nu_, 0.5 * (planner_act_high_ + planner_act_low_));
  act_half_ = Eigen::VectorXd::Constant(nu_, 0.5 * (planner_act_high_ - planner_act_low_));

  if (use_policy_onnx_ || mode_ == NmpcMode::TrackReferencePolicy ||
      mode_ == NmpcMode::TrackPolicyWarmstartGoal) {
    if (policy_onnx_path_.empty()) {
      throw std::runtime_error(
          "policy-driven NMPC modes require policy_onnx_path");
    }
    policy_onnx_ = std::make_unique<PolicyOnnx>(policy_onnx_path_, policy_threads_);
    const int expect_dim = 6 + static_cast<int>(n_quads_ * 12) + static_cast<int>(nu_);
    const int model_in = policy_onnx_->input_dim();
    if (model_in > 0 && model_in != expect_dim) {
      throw std::runtime_error(
          "Policy ONNX input dim mismatch: model expects " + std::to_string(model_in) +
          " but NMPC features are " + std::to_string(expect_dim));
    }
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
    case NmpcMode::TrackReferencePolicy:
    case NmpcMode::TrackPolicyWarmstartGoal: {
      // New chunk policy contract: output is [H*nu].
      const Eigen::VectorXd obs_feat = build_policy_features(x_init_);
      Eigen::VectorXd flat =
          policy_onnx_->predict_chunk(obs_feat, prev_action_policy_, static_cast<int>(N_), static_cast<int>(nu_));
      if (flat.size() % static_cast<Eigen::Index>(nu_) != 0) {
        throw std::runtime_error(
            "PolicyChunk: ONNX output size must be multiple of nu. got " +
            std::to_string(flat.size()) + " and nu=" + std::to_string(nu_));
      }
      last_policy_features_ = obs_feat;
      last_policy_chunk_raw_ = flat;
      last_policy_horizon_ = static_cast<std::size_t>(flat.size() / static_cast<Eigen::Index>(nu_));
      if (last_policy_horizon_ == 0) {
        throw std::runtime_error("PolicyChunk: ONNX output horizon is zero");
      }
      warm_start_N_.actions.resize(N_);
      const std::size_t fill_h = std::min<std::size_t>(N_, last_policy_horizon_);
      for (std::size_t i = 0; i < fill_h; ++i) {
        Eigen::VectorXd a = flat.segment(static_cast<Eigen::Index>(i * nu_), static_cast<Eigen::Index>(nu_));
        // Policy predicts in normalized space [-1, 1].
        for (std::size_t j = 0; j < nu_; ++j) {
          const Eigen::Index jj = static_cast<Eigen::Index>(j);
          a(jj) = std::min(std::max(a(jj), -1.0), 1.0);
        }
        // Map to planner control range [planner_act_low_, planner_act_high_].
        Eigen::VectorXd u = a.cwiseProduct(act_half_) + act_mid_;
        for (std::size_t j = 0; j < nu_; ++j) {
          u(static_cast<Eigen::Index>(j)) =
              std::min(std::max(u(static_cast<Eigen::Index>(j)), policy_u_clip_min_), policy_u_clip_max_);
        }
        warm_start_N_.actions[i] = u;
      }
      if (fill_h < N_) {
        const Eigen::VectorXd pad = warm_start_N_.actions[fill_h - 1];
        for (std::size_t i = fill_h; i < N_; ++i) {
          warm_start_N_.actions[i] = pad;
        }
      }
      // Rollout warm-start states from measured x_init_ with policy controls.
      rollout_warm_start_states_from_actions();
      if (mode_ == NmpcMode::TrackReferencePolicy) {
        ref_traj_N_ = warm_start_N_;
        track_reference_active_ = true;
      } else {
        ref_traj_N_.num_time_steps = N_;
        ref_traj_N_.states.assign(N_ + 1, problem_->goal);
        ref_traj_N_.actions.assign(N_, robot_->u_0);
        track_reference_active_ = false;
      }
      break;
    }
    case NmpcMode::TrackReferenceNmpcRefWarm: {
      last_policy_features_.resize(0);
      last_policy_chunk_raw_.resize(0);
      last_policy_horizon_ = 0;
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
      last_policy_features_.resize(0);
      last_policy_chunk_raw_.resize(0);
      last_policy_horizon_ = 0;
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
      last_policy_features_.resize(0);
      last_policy_chunk_raw_.resize(0);
      last_policy_horizon_ = 0;
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
  const Eigen::VectorXd x_before = x_init_;
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
    info.ddp_iters = static_cast<int>(ddp_solver_->get_iter());
    info.ddp_cost = ddp_solver_->get_cost();
    info.ddp_stop = ddp_solver_->get_stop();
  }
  if (!have_valid_window_) {
    throw std::runtime_error("No valid control window available after solve");
  }

  Eigen::VectorXd u = last_solved_window_.actions.at(planned_window_idx_);
  planned_window_idx_++;

  const double umax = 1.4;
  if (control_noise_ > 0.0) {
    u += control_noise_ * umax * Eigen::VectorXd::Random(nu_);
    if (dynamics_->u_lb.size() == static_cast<Eigen::Index>(nu_) &&
        dynamics_->u_ub.size() == static_cast<Eigen::Index>(nu_)) {
      u = u.cwiseMax(dynamics_->u_lb).cwiseMin(dynamics_->u_ub);
    }
  }

  Eigen::VectorXd xnext(nx_);
  robot_->step(xnext, x_init_.head(nx_), u.head(nu_), robot_->ref_dt);
  apply_payload_disturbance(xnext, k);

  sol_.states.push_back(xnext);
  sol_.actions.push_back(u);
  u_prev_exec_ = u;
  for (std::size_t j = 0; j < nu_; ++j) {
    const Eigen::Index jj = static_cast<Eigen::Index>(j);
    const double denom = std::max(1e-9, act_half_(jj));
    const double a = (u(jj) - act_mid_(jj)) / denom;
    prev_action_policy_(jj) = std::min(std::max(a, -1.0), 1.0);
  }
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
  if (debug_policy_loop_ && (info.did_solve || info.reached_goal || info.failed)) {
    write_policy_debug_step(k, info.did_solve, x_before, u, xnext, info.goal_distance);
  }
  return info;
}

void NmpcController::run() {
  sol_.states.clear();
  sol_.actions.clear();
  sol_.states.push_back(problem_->start);
  x_init_ = problem_->start;
  prev_action_policy_.setZero();
  last_policy_features_.resize(0);
  last_policy_chunk_raw_.resize(0);
  last_policy_horizon_ = 0;
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
  int total_ddp_iters = 0;
  int max_ddp_iters = 0;
  std::vector<double> step_hz_samples;
  std::vector<double> solve_hz_samples;
  std::vector<int> ddp_iters_samples;
  std::vector<double> ddp_cost_samples;
  std::vector<double> ddp_stop_samples;
  step_hz_samples.reserve(max_steps_);
  solve_hz_samples.reserve(max_steps_);
  ddp_iters_samples.reserve(max_steps_);
  ddp_cost_samples.reserve(max_steps_);
  ddp_stop_samples.reserve(max_steps_);
  const auto run_t0 = std::chrono::steady_clock::now();

  if (debug_policy_stream_.is_open()) {
    debug_policy_stream_.close();
  }
  if (debug_policy_loop_) {
    fs::path dbg_path = debug_policy_path_.empty()
                            ? (fs::path(results_path_).parent_path() / (fs::path(results_path_).stem().string() + "_policy_debug.jsonl"))
                            : fs::path(debug_policy_path_);
    if (dbg_path.has_parent_path()) {
      fs::create_directories(dbg_path.parent_path());
    }
    debug_policy_stream_.open(dbg_path, std::ios::out | std::ios::trunc);
    if (!debug_policy_stream_) {
      throw std::runtime_error("Failed to open debug_policy_path: " + dbg_path.string());
    }
    std::cout << "[nmpc] policy debug log: " << dbg_path.string() << "\n";
  }

  if (!do_optimize_) return;
  for (int k = 0; k < static_cast<int>(max_steps_); ++k) {
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
      total_ddp_iters += info.ddp_iters;
      max_ddp_iters = std::max(max_ddp_iters, info.ddp_iters);
      ddp_iters_samples.push_back(info.ddp_iters);
      ddp_cost_samples.push_back(info.ddp_cost);
      ddp_stop_samples.push_back(info.ddp_stop);
    }
    if (info.failed) {
      std::cout << "Tracking failed: distance (" << info.goal_distance
                << ") exceeded threshold (" << fail_threshold_ << ") at step " << k << "\n";
      break;
    }
    if (info.reached_goal) {
      std::cout << "Reached goal at step " << k
                << " with distance " << info.goal_distance << "\n";
      break;
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
  const double mean_ddp_iters = solve_count ? (static_cast<double>(total_ddp_iters) / static_cast<double>(solve_count)) : 0.0;
  auto mean_std_int = [](const std::vector<int>& v) -> std::pair<double, double> {
    if (v.empty()) return {0.0, 0.0};
    double mean = 0.0;
    for (int x : v) mean += static_cast<double>(x);
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (int x : v) { const double d = static_cast<double>(x) - mean; var += d * d; }
    var /= static_cast<double>(v.size());
    return {mean, std::sqrt(var)};
  };
  const auto [ddp_iters_mean, ddp_iters_std] = mean_std_int(ddp_iters_samples);
  const auto [ddp_cost_mean, ddp_cost_std] = mean_std(ddp_cost_samples);
  const auto [ddp_stop_mean, ddp_stop_std] = mean_std(ddp_stop_samples);

  fs::path timing_path = "nmpc_timing.json";
  if (!timing_output_path_override_.empty()) {
    timing_path = timing_output_path_override_;
  } else if (!results_path_.empty()) {
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
               << "  \"debug_policy_loop\": " << (debug_policy_loop_ ? "true" : "false") << ",\n"
               << "  \"control_noise\": " << control_noise_ << ",\n"
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
               << "  \"run_hz\": " << run_hz << ",\n"
               << "  \"total_ddp_iters\": " << total_ddp_iters << ",\n"
               << "  \"mean_ddp_iters\": " << mean_ddp_iters << ",\n"
               << "  \"max_ddp_iters\": " << max_ddp_iters << ",\n"
               << "  \"ddp_iters_mean\": " << ddp_iters_mean << ",\n"
               << "  \"ddp_iters_std\": " << ddp_iters_std << ",\n"
               << "  \"ddp_cost_mean\": " << ddp_cost_mean << ",\n"
               << "  \"ddp_cost_std\": " << ddp_cost_std << ",\n"
               << "  \"ddp_stop_mean\": " << ddp_stop_mean << ",\n"
               << "  \"ddp_stop_std\": " << ddp_stop_std << "\n"
               << "}\n";
    last_timing_output_path_ = timing_path.string();
    std::cout << "Saved NMPC timing summary to: " << timing_path.string() << "\n";
  }
  if (debug_policy_stream_.is_open()) {
    debug_policy_stream_.close();
  }
}

void NmpcController::write_policy_debug_step(int k, bool did_solve, const Eigen::VectorXd &x_before,
                                             const Eigen::VectorXd &u_applied, const Eigen::VectorXd &x_after,
                                             double goal_distance) {
  if (!debug_policy_loop_ || !debug_policy_stream_) {
    return;
  }
  debug_policy_stream_ << "{";
  debug_policy_stream_ << "\"k\":" << k << ",";
  debug_policy_stream_ << "\"mode\":\"" << options_trajopt_.nmpc_mode << "\",";
  debug_policy_stream_ << "\"did_solve\":" << (did_solve ? "true" : "false") << ",";
  debug_policy_stream_ << "\"nmpc_horizon\":" << N_ << ",";
  debug_policy_stream_ << "\"policy_horizon\":" << last_policy_horizon_ << ",";
  debug_policy_stream_ << "\"planned_window_idx\":" << planned_window_idx_ << ",";
  debug_policy_stream_ << "\"solve_every_k_steps\":" << solve_every_k_steps_ << ",";
  debug_policy_stream_ << "\"goal_distance\":" << goal_distance << ",";
  debug_policy_stream_ << "\"x_before\":"; json_write_vec(debug_policy_stream_, x_before); debug_policy_stream_ << ",";
  debug_policy_stream_ << "\"x_after\":"; json_write_vec(debug_policy_stream_, x_after); debug_policy_stream_ << ",";
  debug_policy_stream_ << "\"u_applied\":"; json_write_vec(debug_policy_stream_, u_applied); debug_policy_stream_ << ",";
  debug_policy_stream_ << "\"prev_action_policy\":"; json_write_vec(debug_policy_stream_, prev_action_policy_); debug_policy_stream_ << ",";
  debug_policy_stream_ << "\"policy_features\":"; json_write_vec(debug_policy_stream_, last_policy_features_); debug_policy_stream_ << ",";
  debug_policy_stream_ << "\"policy_chunk_raw\":"; json_write_vec(debug_policy_stream_, last_policy_chunk_raw_);
  if (!warm_start_N_.actions.empty()) {
    debug_policy_stream_ << ",\"warm_start_u0\":";
    json_write_vec(debug_policy_stream_, warm_start_N_.actions.front());
  }
  if (!last_solved_window_.actions.empty()) {
    debug_policy_stream_ << ",\"nmpc_u0\":";
    json_write_vec(debug_policy_stream_, last_solved_window_.actions.front());
  }
  if (did_solve) {
    debug_policy_stream_ << ",\"warm_start_actions_window\":";
    json_write_traj_vec(debug_policy_stream_, warm_start_N_.actions);
    debug_policy_stream_ << ",\"nmpc_actions_window\":";
    json_write_traj_vec(debug_policy_stream_, last_solved_window_.actions);
    debug_policy_stream_ << ",\"ref_actions_window\":";
    json_write_traj_vec(debug_policy_stream_, ref_traj_N_.actions);

    debug_policy_stream_ << ",\"warm_start_payload_pv_window\":";
    json_write_payload_pv(debug_policy_stream_, warm_start_N_.states);
    debug_policy_stream_ << ",\"nmpc_payload_pv_window\":";
    json_write_payload_pv(debug_policy_stream_, last_solved_window_.states);
    debug_policy_stream_ << ",\"ref_payload_pv_window\":";
    json_write_payload_pv(debug_policy_stream_, ref_traj_N_.states);
  }
  debug_policy_stream_ << "}\n";
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
