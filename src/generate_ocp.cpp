

#include "generate_ocp.hpp"
#include "mujoco_quadrotors_payload.hpp"

namespace dynoplan {

using dynobench::check_equal;
using dynobench::FMT;
using Vxd = Eigen::VectorXd;
using V3d = Eigen::Vector3d;
using V4d = Eigen::Vector4d;

void Generate_params::print(std::ostream &out) const {
  auto pre = "";
  auto after = ": ";
  out << pre << STR(collisions, after) << std::endl;
  out << pre << STR(free_time, after) << std::endl;
  out << pre << STR(name, after) << std::endl;
  out << pre << STR(N, after) << std::endl;
  out << pre << STR(contour_control, after) << std::endl;
  out << pre << STR(max_alpha, after) << std::endl;
  out << STR(goal_cost, after) << std::endl;
  STRY(penalty, out, pre, after);

  out << pre << "goal" << after << goal.transpose() << std::endl;
  out << pre << "start" << after << start.transpose() << std::endl;
  out << pre << "states" << std::endl;
  // for (const auto &s : states)
  //   out << "  - " << s.format(FMT) << std::endl;
  // out << pre << "states_weights" << std::endl;
  // for (const auto &s : states_weights)
  //   out << "  - " << s.format(FMT) << std::endl;
  // out << pre << "actions" << std::endl;
  // for (const auto &s : actions)
  //   out << "  - " << s.format(FMT) << std::endl;
}

ptr<crocoddyl::ShootingProblem>
generate_problem(const Generate_params &gen_args,
                 const Options_trajopt &options_trajopt) {

  // std::cout << "**\nGENERATING PROBLEM\n**\nArgs:\n" << std::endl;
  std::cout << "\nGENERATING PROBLEM" << std::endl;
  // gen_args.print(std::cout);
  std::cout << "**\n" << std::endl;

  // std::cout << "**\nOpti Params\n**\n" << std::endl;
  // options_trajopt.print(std::cout);
  // std::cout << "**\n" << std::endl;

  std::vector<ptr<Cost>> feats_terminal;
  ptr<crocoddyl::ActionModelAbstract> am_terminal;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> amq_runs;


  std::map<std::string, double> additional_params;
  Control_Mode control_mode;
  control_mode = Control_Mode::default_mode;

  ptr<Dynamics> dyn =
      create_dynamics(gen_args.model_robot, control_mode, additional_params);


  CHECK(dyn, AT);

  // dyn->print_bounds(std::cout);

  size_t nu = dyn->nu;
  size_t nx = dyn->nx;

  ptr<Cost> control_feature =
      mk<Control_cost>(nx, nu, nu, dyn->u_weight, dyn->u_ref);


  bool use_hard_bounds = options_trajopt.control_bounds;

  if (options_trajopt.soft_control_bounds) {
    use_hard_bounds = false;
  }

  for (size_t t = 0; t < gen_args.N; t++) {

    std::vector<ptr<Cost>> feats_run;

    feats_run.push_back(control_feature);

    // feats_run.push_back(mk<State_bounds>(nx, nu, nx, v, -v);

    if (gen_args.collisions && gen_args.model_robot->env) {
      ptr<Cost> cl_feature = mk<Col_cost>(nx, nu, 1, gen_args.model_robot,
                                          options_trajopt.collision_weight);
      feats_run.push_back(cl_feature);
    }
    //

    if (startsWith(gen_args.name, "mujocoquadspayload")) {
      if ((control_mode == Control_Mode::default_mode)) {
        // std::cout << "adding regularization on the acceleration! " << std::endl;
        auto ptr_derived =
            std::dynamic_pointer_cast<dynobench::Model_MujocoQuadsPayload>(
                gen_args.model_robot);
        // Additionally, add regularization!!
        if (gen_args.name == "mujocoquadspayload_switch3") {
            Vxd state_weights = Vxd::Zero(nx);
            Vxd state_ref     = Vxd::Zero(nx);

          const int t_wp = 110;
          if (std::abs(int(t) - t_wp) <= 4) {
            V3d payload_pos(0.0, 0.0, 0.75);
            V3d quad1_pos(0.0 ,  0.25, 1.0); 
            V3d quad2_pos(0.0 , -0.25, 1.0); 
            V3d quad3_pos(-0.25,   0.0, 1.0); 

            state_weights.segment<3>(0)  = 1500.0 * V3d::Ones();
            state_weights.segment<3>(7)  = 1500.0 * V3d::Ones();
            state_weights.segment<3>(14) = 1500.0 * V3d::Ones();
            state_weights.segment<3>(21) = 0.0 * V3d::Ones();
            state_ref.segment<3>(0)      = payload_pos;
            state_ref.segment<3>(7)      = quad1_pos;
            state_ref.segment<3>(14)     = quad2_pos;
            state_ref.segment<3>(21)     = quad3_pos;
            std::cout << "adding waypoint constraints at t: " << t
            << "\nstate_weights: " << state_weights.transpose()
            << "\nstate_ref: " << state_ref.transpose() << std::endl;
            ptr<Cost> state_feature = mk<State_cost>(
              nx, nu, nx, state_weights, state_ref);
              feats_run.push_back(state_feature);
          } 
        } else if (gen_args.name == "mujocoquadspayload_switch2") {
            Vxd state_weights = Vxd::Zero(nx);
            Vxd state_ref     = Vxd::Zero(nx);

            const int t_wp = 175;
            if (std::abs(int(t) - t_wp) <= 4) {
              V3d payload_pos(0.25, 0.0, 0.8);
              V3d quad1_pos(0.25 ,  0.25, 1.2); 
              V3d quad2_pos(0.25 , -0.25, 1.2); 

              state_weights.segment<3>(0)  = 500.0 * V3d::Ones();
              state_weights.segment<3>(7)  = 500.0 * V3d::Ones();
              state_weights.segment<3>(14) = 500.0 * V3d::Ones();
              state_ref.segment<3>(0)      = payload_pos;
              state_ref.segment<3>(7)      = quad1_pos;
              state_ref.segment<3>(14)     = quad2_pos;
              std::cout << "adding waypoint constraints at t: " << t
              << "\nstate_weights: " << state_weights.transpose()
              << "\nstate_ref: " << state_ref.transpose() << std::endl;
              ptr<Cost> state_feature = mk<State_cost>(
                nx, nu, nx, state_weights, state_ref);
                // feats_run.push_back(state_feature);
            } 
        }
        // ptr<Cost> acc_cost = mk<mujoco_quads_payload_acc>(
        //     gen_args.model_robot, gen_args.model_robot->k_acc);
        // feats_run.push_back(acc_cost);
      } else {
        // QUIM TODO: Check if required!!
        NOT_IMPLEMENTED;
      }
    }
  
    if (dyn->x_lb.size() && dyn->x_weightb.sum() > 1e-10) {

      Eigen::VectorXd v = dyn->x_lb;
      feats_run.push_back(mk<State_bounds>(nx, nu, nx, v, -dyn->x_weightb));
    }
    if (dyn->x_ub.size() && dyn->x_weightb.sum() > 1e-10) {

      Eigen::VectorXd v = dyn->x_ub;
      feats_run.push_back(mk<State_bounds>(nx, nu, nx, v, dyn->x_weightb));
    }

    boost::shared_ptr<crocoddyl::ActionModelAbstract> am_run =
        to_am_base(mk<ActionModelDyno>(dyn, feats_run));

    am_run->set_u_lb(options_trajopt.u_bound_scale * dyn->u_lb);
    am_run->set_u_ub(options_trajopt.u_bound_scale * dyn->u_ub);
    amq_runs.push_back(am_run);
  }

  if (gen_args.goal_cost) {

    DYNO_CHECK_EQ(static_cast<size_t>(gen_args.goal.size()),
                  gen_args.model_robot->nx, AT);

    Eigen::VectorXd goal_weight = gen_args.model_robot->goal_weight;

    if (!goal_weight.size()) {
      goal_weight.resize(gen_args.model_robot->nx);
      goal_weight.setOnes();
    }

    // CSTR_V(goal_weight);

    ptr<Cost> state_feature = mk<State_cost_model>(
        gen_args.model_robot, nx, nu,
        gen_args.penalty * options_trajopt.weight_goal * goal_weight,
        gen_args.goal);

    feats_terminal.push_back(state_feature);
  }
  am_terminal = to_am_base(mk<ActionModelDyno>(dyn, feats_terminal));
  CHECK(am_terminal, AT);

  for (auto &a : amq_runs)
    CHECK(a, AT);

  ptr<crocoddyl::ShootingProblem> problem =
      mk<crocoddyl::ShootingProblem>(gen_args.start, amq_runs, am_terminal);

  return problem;
};

std::vector<ReportCost> report_problem(ptr<crocoddyl::ShootingProblem> problem,
                                       const std::vector<Vxd> &xs,
                                       const std::vector<Vxd> &us,
                                       const char *file_name) {
  std::vector<ReportCost> reports;

  for (size_t i = 0; i < problem->get_runningModels().size(); i++) {
    auto &x = xs.at(i);
    auto &u = us.at(i);
    auto p = boost::static_pointer_cast<ActionModelDyno>(
        problem->get_runningModels().at(i));
    std::vector<ReportCost> reports_i = get_report(
        p, [&](ptr<Cost> f, Eigen::Ref<Vxd> r) { f->calc(r, x, u); });

    for (auto &report_ii : reports_i)
      report_ii.time = i;
    reports.insert(reports.end(), reports_i.begin(), reports_i.end());
  }

  auto p =
      boost::static_pointer_cast<ActionModelDyno>(problem->get_terminalModel());
  std::vector<ReportCost> reports_t = get_report(
      p, [&](ptr<Cost> f, Eigen::Ref<Vxd> r) { f->calc(r, xs.back()); });

  for (auto &report_ti : reports_t)
    report_ti.time = xs.size() - 1;
  ;

  reports.insert(reports.begin(), reports_t.begin(), reports_t.end());

  // write down the reports.
  //

  std::string one_space = " ";
  std::string two_space = "  ";
  std::string four_space = "    ";

  create_dir_if_necessary(file_name);

  std::ofstream reports_file(file_name);
  for (auto &report : reports) {
    reports_file << "-" << one_space << "name: " << report.name << std::endl;
    reports_file << two_space << "time: " << report.time << std::endl;
    reports_file << two_space << "cost: " << report.cost << std::endl;
    reports_file << two_space << "type: " << static_cast<int>(report.type)
                 << std::endl;
    if (report.r.size()) {
      reports_file << two_space << "r: " << report.r.format(FMT) << std::endl;
    }
  }

  return reports;
}

bool check_problem(ptr<crocoddyl::ShootingProblem> problem,
                   ptr<crocoddyl::ShootingProblem> problem2,
                   const std::vector<Vxd> &xs, const std::vector<Vxd> &us) {

  bool equal = true;
  // for (auto &x : xs) {
  //   CSTR_V(x);
  //   CSTR_(x.size());
  // }
  // std::cout << "us" << std::endl;
  // for (auto &u : us) {
  //
  //   CSTR_(u.size());
  //   CSTR_V(u);
  // }

  problem->calc(xs, us);
  problem->calcDiff(xs, us);
  auto data_running = problem->get_runningDatas();
  auto data_terminal = problem->get_terminalData();

  // now with finite diff
  problem2->calc(xs, us);
  problem2->calcDiff(xs, us);
  auto data_running_diff = problem2->get_runningDatas();
  auto data_terminal_diff = problem2->get_terminalData();

  double tol = 1e-3;
  bool check;

  check = check_equal(data_terminal_diff->Lx, data_terminal->Lx, tol, tol);
  WARN(check, std::string("LxT:") + AT);
  if (!check)
    equal = false;

  check = check_equal(data_terminal_diff->Lxx, data_terminal->Lxx, tol, tol);
  if (!check)
    equal = false;
  WARN(check, std::string("LxxT:") + AT);

  DYNO_CHECK_EQ(data_running_diff.size(), data_running.size(), AT);
  for (size_t i = 0; i < data_running_diff.size(); i++) {
    auto &d = data_running.at(i);
    auto &d_diff = data_running_diff.at(i);
    CSTR_V(xs.at(i));
    CSTR_V(us.at(i));
    check = check_equal(d_diff->Fx, d->Fx, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Fx:") + AT);
    check = check_equal(d_diff->Fu, d->Fu, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Fu:") + AT);
    check = check_equal(d_diff->Lx, d->Lx, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Lx:") + AT);
    check = check_equal(d_diff->Lu, d->Lu, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Lu:") + AT);
    check = check_equal(d_diff->Fx, d->Fx, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Fx:") + AT);
    check = check_equal(d_diff->Fu, d->Fu, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Fu:") + AT);
    check = check_equal(d_diff->Lxx, d->Lxx, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Lxx:") + AT);
    check = check_equal(d_diff->Lxu, d->Lxu, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Lxu:") + AT);
    check = check_equal(d_diff->Luu, d->Luu, tol, tol);
    if (!check)
      equal = false;
    WARN(check, std::string("Luu:") + AT);
  }
  return equal;
}

} // namespace dynoplan
