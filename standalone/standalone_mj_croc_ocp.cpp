#include <mujoco/mujoco.h>

#include <crocoddyl/core/action-base.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/states/euclidean.hpp>

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include <boost/make_shared.hpp>

#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CliOptions {
  std::string problem_yaml;
  std::string xml_file;
  std::string out_yaml = "standalone_ocp_result.yaml";
  std::size_t horizon = 180;
  std::size_t max_iter = 30;
  bool start_from_goal = false;
};

void print_usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " --problem-yaml <path> --xml-file <path> [--horizon N] [--max-iter N]"
      << " [--out-yaml path] [--start-from-goal]\n";
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto read_next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };
    if (a == "--problem-yaml") {
      o.problem_yaml = read_next("--problem-yaml");
    } else if (a == "--xml-file") {
      o.xml_file = read_next("--xml-file");
    } else if (a == "--out-yaml") {
      o.out_yaml = read_next("--out-yaml");
    } else if (a == "--horizon") {
      o.horizon = static_cast<std::size_t>(std::stoul(read_next("--horizon")));
    } else if (a == "--max-iter") {
      o.max_iter = static_cast<std::size_t>(std::stoul(read_next("--max-iter")));
    } else if (a == "--start-from-goal") {
      o.start_from_goal = true;
    } else if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }

  if (o.problem_yaml.empty() || o.xml_file.empty()) {
    print_usage(argv[0]);
    throw std::runtime_error("Both --problem-yaml and --xml-file are required");
  }
  return o;
}

std::vector<double> as_double_vector(const YAML::Node& n, const std::string& name) {
  if (!n || !n.IsSequence()) {
    throw std::runtime_error(name + " must be a sequence");
  }
  std::vector<double> out;
  out.reserve(n.size());
  for (const auto& v : n) {
    out.push_back(v.as<double>());
  }
  return out;
}

struct ProblemState {
  Eigen::VectorXd start;
  Eigen::VectorXd goal;
};

bool has_payload_layout(const int nq, const int nv) {
  if (nq <= 0 || nv <= 0) return false;
  if ((nq % 7) != 0 || (nv % 6) != 0) return false;
  return (nq / 7) == (nv / 6);
}

void convert_state_xyzw_to_mj_wxyz(Eigen::VectorXd* x, const int nq, const int nv) {
  if (!has_payload_layout(nq, nv)) return;
  const int n_bodies = nq / 7;
  for (int b = 0; b < n_bodies; ++b) {
    const int qbase = 7 * b + 3;
    const double qx = (*x)[qbase + 0];
    const double qy = (*x)[qbase + 1];
    const double qz = (*x)[qbase + 2];
    const double qw = (*x)[qbase + 3];
    (*x)[qbase + 0] = qw;
    (*x)[qbase + 1] = qx;
    (*x)[qbase + 2] = qy;
    (*x)[qbase + 3] = qz;
  }
}

void convert_state_mj_wxyz_to_xyzw(Eigen::VectorXd* x, const int nq, const int nv) {
  if (!has_payload_layout(nq, nv)) return;
  const int n_bodies = nq / 7;
  for (int b = 0; b < n_bodies; ++b) {
    const int qbase = 7 * b + 3;
    const double qw = (*x)[qbase + 0];
    const double qx = (*x)[qbase + 1];
    const double qy = (*x)[qbase + 2];
    const double qz = (*x)[qbase + 3];
    (*x)[qbase + 0] = qx;
    (*x)[qbase + 1] = qy;
    (*x)[qbase + 2] = qz;
    (*x)[qbase + 3] = qw;
  }
}

Eigen::Vector4d quat_xyzw_mul(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2) {
  const double x1 = q1(0), y1 = q1(1), z1 = q1(2), w1 = q1(3);
  const double x2 = q2(0), y2 = q2(1), z2 = q2(2), w2 = q2(3);
  Eigen::Vector4d out;
  out << (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2),
      (w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2),
      (w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2),
      (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2);
  return out;
}

Eigen::Vector4d quat_xyzw_inv(const Eigen::Vector4d& q) {
  const double n2 = q.squaredNorm();
  if (n2 <= 1e-12) return Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  return Eigen::Vector4d(-q(0), -q(1), -q(2), q(3)) / n2;
}

Eigen::Vector3d quat_error_vec_xyzw(const Eigen::Vector4d& q_ref, const Eigen::Vector4d& q_cur) {
  Eigen::Vector4d q_err = quat_xyzw_mul(quat_xyzw_inv(q_ref), q_cur);
  if (q_err(3) < 0.0) q_err = -q_err;
  return q_err.head<3>();
}

ProblemState load_problem_state(const std::string& yaml_path) {
  const YAML::Node root = YAML::LoadFile(yaml_path);
  const YAML::Node jr = root["joint_robot"];
  if (!jr || !jr.IsSequence() || jr.size() == 0) {
    throw std::runtime_error("joint_robot[0] missing in problem yaml");
  }
  const YAML::Node item = jr[0];
  auto start_v = as_double_vector(item["start"], "joint_robot[0].start");
  auto goal_v = as_double_vector(item["goal"], "joint_robot[0].goal");
  if (start_v.size() != goal_v.size()) {
    throw std::runtime_error("start and goal sizes mismatch");
  }
  ProblemState out;
  out.start = Eigen::Map<Eigen::VectorXd>(start_v.data(), static_cast<Eigen::Index>(start_v.size()));
  out.goal = Eigen::Map<Eigen::VectorXd>(goal_v.data(), static_cast<Eigen::Index>(goal_v.size()));
  return out;
}

class MujocoDiscreteActionData;

class MujocoDiscreteActionModel final : public crocoddyl::ActionModelAbstract {
 public:
  MujocoDiscreteActionModel(mjModel* model, const Eigen::VectorXd& xref, const Eigen::VectorXd& uref,
                            const Eigen::VectorXd& q_diag, const Eigen::VectorXd& r_diag,
                            bool terminal = false)
      : crocoddyl::ActionModelAbstract(
            boost::make_shared<crocoddyl::StateVector>(static_cast<std::size_t>(model->nq + model->nv)),
            static_cast<std::size_t>(model->nu),
            1,
            0,
            0),
        model_(model),
        xref_(xref),
        uref_(uref),
        q_diag_(q_diag),
        r_diag_(r_diag),
        terminal_(terminal),
        eps_fd_(1e-5) {
    if (!model_) throw std::runtime_error("MujocoDiscreteActionModel: model is null");
    if (xref_.size() != get_state()->get_nx()) throw std::runtime_error("xref size mismatch");
    if (!terminal_ && uref_.size() != static_cast<Eigen::Index>(nu_)) {
      throw std::runtime_error("uref size mismatch");
    }
    if (q_diag_.size() != get_state()->get_nx()) throw std::runtime_error("q_diag size mismatch");
    if (!terminal_ && r_diag_.size() != static_cast<Eigen::Index>(nu_)) {
      throw std::runtime_error("r_diag size mismatch");
    }
  }

  ~MujocoDiscreteActionModel() override = default;

  void calc(const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override;

  void calcDiff(const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) override;

  boost::shared_ptr<crocoddyl::ActionDataAbstract> createData() override;

 private:
  void rollout_step(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                    mjData* d, Eigen::VectorXd* xnext_out, double* cost_out) const {
    const int nq = model_->nq;
    const int nv = model_->nv;
    const int nu = model_->nu;

    Eigen::VectorXd x_mj = x;
    convert_state_xyzw_to_mj_wxyz(&x_mj, nq, nv);
    for (int i = 0; i < nq; ++i) d->qpos[i] = x_mj[i];
    for (int i = 0; i < nv; ++i) d->qvel[i] = x[nq + i];
    for (int i = 0; i < nu; ++i) d->ctrl[i] = terminal_ ? 0.0 : u[i];

    mj_forward(model_, d);
    mj_step(model_, d);

    xnext_out->resize(nq + nv);
    for (int i = 0; i < nq; ++i) (*xnext_out)[i] = d->qpos[i];
    for (int i = 0; i < nv; ++i) (*xnext_out)[nq + i] = d->qvel[i];
    convert_state_mj_wxyz_to_xyzw(xnext_out, nq, nv);

    double c = 0.0;
    if (has_payload_layout(nq, nv)) {
      const int n_bodies = nq / 7;
      for (int b = 0; b < n_bodies; ++b) {
        const int pbase = 7 * b;
        const int qbase = pbase + 3;
        for (int i = 0; i < 3; ++i) {
          const Eigen::Index idx = pbase + i;
          const double e = (*xnext_out)[idx] - xref_[idx];
          c += 0.5 * q_diag_[idx] * e * e;
        }
        const Eigen::Vector4d q_cur = xnext_out->segment<4>(qbase);
        const Eigen::Vector4d q_ref = xref_.segment<4>(qbase);
        const Eigen::Vector3d e_q = quat_error_vec_xyzw(q_ref, q_cur);
        const double wq = std::max(
            1e-6, 0.25 * (q_diag_[qbase + 0] + q_diag_[qbase + 1] + q_diag_[qbase + 2] + q_diag_[qbase + 3]));
        c += 0.5 * wq * e_q.squaredNorm();
      }
      for (int i = nq; i < nq + nv; ++i) {
        const Eigen::Index idx = i;
        const double e = (*xnext_out)[idx] - xref_[idx];
        c += 0.5 * q_diag_[idx] * e * e;
      }
    } else {
      const Eigen::VectorXd dx = (*xnext_out) - xref_;
      c = 0.5 * (q_diag_.array() * dx.array().square()).sum();
    }
    if (!terminal_) {
      const Eigen::VectorXd du = u - uref_;
      c += 0.5 * (r_diag_.array() * du.array().square()).sum();
    }
    *cost_out = c;
  }

  mjModel* model_;
  Eigen::VectorXd xref_;
  Eigen::VectorXd uref_;
  Eigen::VectorXd q_diag_;
  Eigen::VectorXd r_diag_;
  bool terminal_;
  double eps_fd_;
};

class MujocoDiscreteActionData final : public crocoddyl::ActionDataAbstract {
 public:
  explicit MujocoDiscreteActionData(crocoddyl::ActionModelAbstract* model, mjModel* m)
      : crocoddyl::ActionDataAbstract(model), mj_data(mj_makeData(m)) {
    if (!mj_data) throw std::runtime_error("mj_makeData failed");
  }
  ~MujocoDiscreteActionData() override {
    if (mj_data) mj_deleteData(mj_data);
  }
  mjData* mj_data;
};

boost::shared_ptr<crocoddyl::ActionDataAbstract> MujocoDiscreteActionModel::createData() {
  return boost::make_shared<MujocoDiscreteActionData>(this, model_);
}

void MujocoDiscreteActionModel::calc(const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                                     const Eigen::Ref<const Eigen::VectorXd>& x,
                                     const Eigen::Ref<const Eigen::VectorXd>& u) {
  auto* d = static_cast<MujocoDiscreteActionData*>(data.get());
  rollout_step(x, u, d->mj_data, &data->xnext, &data->cost);
}

void MujocoDiscreteActionModel::calcDiff(const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                                         const Eigen::Ref<const Eigen::VectorXd>& x,
                                         const Eigen::Ref<const Eigen::VectorXd>& u) {
  auto* d = static_cast<MujocoDiscreteActionData*>(data.get());
  Eigen::VectorXd xnext0;
  double c0 = 0.0;
  rollout_step(x, u, d->mj_data, &xnext0, &c0);

  data->Fx.setZero();
  data->Fu.setZero();
  data->Lx.setZero();
  data->Lu.setZero();
  data->Lxx.setZero();
  data->Luu.setZero();
  data->Lxu.setZero();

  Eigen::VectorXd xp = x;
  Eigen::VectorXd xm = x;
  Eigen::VectorXd xnext_p, xnext_m;
  for (std::size_t i = 0; i < get_state()->get_nx(); ++i) {
    xp = x;
    xm = x;
    xp[static_cast<Eigen::Index>(i)] += eps_fd_;
    xm[static_cast<Eigen::Index>(i)] -= eps_fd_;

    double cp = 0.0, cm = 0.0;
    rollout_step(xp, u, d->mj_data, &xnext_p, &cp);
    rollout_step(xm, u, d->mj_data, &xnext_m, &cm);

    data->Fx.col(static_cast<Eigen::Index>(i)) = (xnext_p - xnext_m) / (2.0 * eps_fd_);
    data->Lx[static_cast<Eigen::Index>(i)] = (cp - cm) / (2.0 * eps_fd_);
    data->Lxx(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) =
        (cp - 2.0 * c0 + cm) / (eps_fd_ * eps_fd_);
  }

  if (terminal_) return;

  Eigen::VectorXd up = u;
  Eigen::VectorXd um = u;
  for (std::size_t i = 0; i < nu_; ++i) {
    up = u;
    um = u;
    up[static_cast<Eigen::Index>(i)] += eps_fd_;
    um[static_cast<Eigen::Index>(i)] -= eps_fd_;

    double cp = 0.0, cm = 0.0;
    rollout_step(x, up, d->mj_data, &xnext_p, &cp);
    rollout_step(x, um, d->mj_data, &xnext_m, &cm);

    data->Fu.col(static_cast<Eigen::Index>(i)) = (xnext_p - xnext_m) / (2.0 * eps_fd_);
    data->Lu[static_cast<Eigen::Index>(i)] = (cp - cm) / (2.0 * eps_fd_);
    data->Luu(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) =
        (cp - 2.0 * c0 + cm) / (eps_fd_ * eps_fd_);
  }
}

void save_solution_yaml(const std::string& out_yaml, const crocoddyl::SolverFDDP& solver, bool solved) {
  YAML::Emitter em;
  em << YAML::BeginMap;
  em << YAML::Key << "solved" << YAML::Value << solved;
  em << YAML::Key << "iter" << YAML::Value << static_cast<int>(solver.get_iter());
  em << YAML::Key << "cost" << YAML::Value << solver.get_cost();
  em << YAML::Key << "xs" << YAML::Value << YAML::BeginSeq;
  for (const auto& x : solver.get_xs()) {
    em << YAML::Flow << YAML::BeginSeq;
    for (Eigen::Index i = 0; i < x.size(); ++i) em << x[i];
    em << YAML::EndSeq;
  }
  em << YAML::EndSeq;
  em << YAML::Key << "us" << YAML::Value << YAML::BeginSeq;
  for (const auto& u : solver.get_us()) {
    em << YAML::Flow << YAML::BeginSeq;
    for (Eigen::Index i = 0; i < u.size(); ++i) em << u[i];
    em << YAML::EndSeq;
  }
  em << YAML::EndSeq;
  em << YAML::EndMap;
  std::ofstream out(out_yaml);
  out << em.c_str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions opt = parse_cli(argc, argv);
    const ProblemState prob = load_problem_state(opt.problem_yaml);

    std::array<char, 1024> err{};
    mjModel* mjm = mj_loadXML(opt.xml_file.c_str(), nullptr, err.data(), err.size());
    if (!mjm) {
      throw std::runtime_error(std::string("mj_loadXML failed: ") + err.data());
    }
    std::unique_ptr<mjModel, void (*)(mjModel*)> model_guard(mjm, mj_deleteModel);

    const std::size_t nx = static_cast<std::size_t>(mjm->nq + mjm->nv);
    const std::size_t nu = static_cast<std::size_t>(mjm->nu);
    if (prob.start.size() != static_cast<Eigen::Index>(nx)) {
      throw std::runtime_error("Problem start size does not match MuJoCo model nq+nv");
    }
    if (prob.goal.size() != static_cast<Eigen::Index>(nx)) {
      throw std::runtime_error("Problem goal size does not match MuJoCo model nq+nv");
    }

    Eigen::VectorXd x0 = opt.start_from_goal ? prob.goal : prob.start;
    const Eigen::VectorXd x_goal = prob.goal;

    // Hover-like seed: midpoint of actuator control ranges.
    Eigen::VectorXd u_hover = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nu));
    for (std::size_t i = 0; i < nu; ++i) {
      const mjtNum low = mjm->actuator_ctrlrange[2 * i + 0];
      const mjtNum high = mjm->actuator_ctrlrange[2 * i + 1];
      u_hover[static_cast<Eigen::Index>(i)] = 0.5 * (low + high);
    }

    Eigen::VectorXd q_diag = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(nx));
    Eigen::VectorXd qf_diag = 20.0 * Eigen::VectorXd::Ones(static_cast<Eigen::Index>(nx));
    Eigen::VectorXd r_diag = 1e-2 * Eigen::VectorXd::Ones(static_cast<Eigen::Index>(nu));

    // Keep pose terms more important than velocity.
    const std::size_t nq = static_cast<std::size_t>(mjm->nq);
    for (std::size_t i = 0; i < nq; ++i) q_diag[static_cast<Eigen::Index>(i)] = 5.0;
    for (std::size_t i = nq; i < nx; ++i) q_diag[static_cast<Eigen::Index>(i)] = 0.5;
    for (std::size_t i = 0; i < nq; ++i) qf_diag[static_cast<Eigen::Index>(i)] = 50.0;
    for (std::size_t i = nq; i < nx; ++i) qf_diag[static_cast<Eigen::Index>(i)] = 2.0;

    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> running_models;
    running_models.reserve(opt.horizon);
    for (std::size_t k = 0; k < opt.horizon; ++k) {
      running_models.push_back(boost::make_shared<MujocoDiscreteActionModel>(mjm, x_goal, u_hover, q_diag, r_diag));
    }
    auto terminal_model =
        boost::make_shared<MujocoDiscreteActionModel>(mjm, x_goal, Eigen::VectorXd(), qf_diag, Eigen::VectorXd(), true);

    auto problem =
        boost::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
    crocoddyl::SolverFDDP solver(problem);

    std::vector<Eigen::VectorXd> xs_init(opt.horizon + 1, x0);
    std::vector<Eigen::VectorXd> us_init(opt.horizon, u_hover);

    const bool solved = solver.solve(xs_init, us_init, static_cast<unsigned int>(opt.max_iter), true, 1e-7);
    const double x0_err = (x0 - x_goal).norm();
    const double xf_err = (solver.get_xs().back() - x_goal).norm();

    std::cout << "Standalone OCP solved=" << (solved ? "true" : "false")
              << " iter=" << solver.get_iter()
              << " cost=" << solver.get_cost()
              << " x0_err=" << x0_err
              << " xf_err=" << xf_err << "\n";

    save_solution_yaml(opt.out_yaml, solver, solved);
    std::cout << "Saved solution: " << fs::absolute(opt.out_yaml) << "\n";
    return solved ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "[standalone_mj_croc_ocp] ERROR: " << e.what() << "\n";
    return 1;
  }
}
