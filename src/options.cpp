

#include "options.hpp"
#include "dyno_macros.hpp"
#include "general_utils.hpp"

namespace dynoplan {

void Options_trajopt::add_options(po::options_description &desc) {
  set_from_boostop(desc, VAR_WITH_NAME(u_bound_scale));
  set_from_boostop(desc, VAR_WITH_NAME(collision_weight));
  set_from_boostop(desc, VAR_WITH_NAME(th_acceptnegstep));
  set_from_boostop(desc, VAR_WITH_NAME(states_reg));
  set_from_boostop(desc, VAR_WITH_NAME(init_reg));
  set_from_boostop(desc, VAR_WITH_NAME(max_iter));
  set_from_boostop(desc, VAR_WITH_NAME(weight_goal));
  set_from_boostop(desc, VAR_WITH_NAME(th_stop));
  set_from_boostop(desc, VAR_WITH_NAME(policy_control_tracking_weight));
  set_from_boostop(desc, VAR_WITH_NAME(ref_state_tracking_weight));
  set_from_boostop(desc, VAR_WITH_NAME(planner_ref_control_tracking_weight));
  set_from_boostop(desc, VAR_WITH_NAME(policy_ref_control_tracking_weight));
  set_from_boostop(desc, VAR_WITH_NAME(goal_control_regularization_weight));
  set_from_boostop(desc, VAR_WITH_NAME(running_cost_goal_weight_mask));
  set_from_boostop(desc, VAR_WITH_NAME(solve_every_k_steps));
  set_from_boostop(desc, VAR_WITH_NAME(nmpc_mode));
  set_from_boostop(desc, VAR_WITH_NAME(disturbance_enable));
  set_from_boostop(desc, VAR_WITH_NAME(disturbance_start_s));
  set_from_boostop(desc, VAR_WITH_NAME(disturbance_duration_s));
  set_from_boostop(desc, VAR_WITH_NAME(disturbance_force_x));
  set_from_boostop(desc, VAR_WITH_NAME(disturbance_force_y));
  set_from_boostop(desc, VAR_WITH_NAME(disturbance_force_z));
  set_from_boostop(desc, VAR_WITH_NAME(disturbance_payload_mass));
}

void Options_trajopt::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Options_trajopt::__read_from_node(const YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(u_bound_scale));
  set_from_yaml(node, VAR_WITH_NAME(collision_weight));
  set_from_yaml(node, VAR_WITH_NAME(th_acceptnegstep));
  set_from_yaml(node, VAR_WITH_NAME(states_reg));
  set_from_yaml(node, VAR_WITH_NAME(init_reg));
  set_from_yaml(node, VAR_WITH_NAME(th_stop));
  set_from_yaml(node, VAR_WITH_NAME(max_iter));
  set_from_yaml(node, VAR_WITH_NAME(weight_goal));
  set_from_yaml(node, VAR_WITH_NAME(policy_control_tracking_weight));
  set_from_yaml(node, VAR_WITH_NAME(ref_state_tracking_weight));
  set_from_yaml(node, VAR_WITH_NAME(planner_ref_control_tracking_weight));
  set_from_yaml(node, VAR_WITH_NAME(policy_ref_control_tracking_weight));
  set_from_yaml(node, VAR_WITH_NAME(goal_control_regularization_weight));
  set_from_yaml(node, VAR_WITH_NAME(running_cost_goal_weight_mask));
  set_from_yaml(node, VAR_WITH_NAME(solve_every_k_steps));
  set_from_yaml(node, VAR_WITH_NAME(nmpc_mode));
  set_from_yaml(node, VAR_WITH_NAME(disturbance_enable));
  set_from_yaml(node, VAR_WITH_NAME(disturbance_start_s));
  set_from_yaml(node, VAR_WITH_NAME(disturbance_duration_s));
  set_from_yaml(node, VAR_WITH_NAME(disturbance_force_x));
  set_from_yaml(node, VAR_WITH_NAME(disturbance_force_y));
  set_from_yaml(node, VAR_WITH_NAME(disturbance_force_z));
  set_from_yaml(node, VAR_WITH_NAME(disturbance_payload_mass));
}

void Options_trajopt::read_from_yaml(YAML::Node &node) {

  if (node["options_trajopt"]) {
    __read_from_node(node["options_trajopt"]);
  } else {
    __read_from_node(node);
  }
}

void Options_trajopt::print(std::ostream &out, const std::string &be,
                            const std::string &af) const {
  out << be << STR(u_bound_scale, af) << std::endl;
  out << be << STR(collision_weight, af) << std::endl;
  out << be << STR(th_acceptnegstep, af) << std::endl;
  out << be << STR(states_reg, af) << std::endl;
  out << be << STR(th_stop, af) << std::endl;
  out << be << STR(init_reg, af) << std::endl;
  out << be << STR(max_iter, af) << std::endl;
  out << be << STR(weight_goal, af) << std::endl;
  out << be << STR(policy_control_tracking_weight, af) << std::endl;
  out << be << STR(ref_state_tracking_weight, af) << std::endl;
  out << be << STR(planner_ref_control_tracking_weight, af) << std::endl;
  out << be << STR(policy_ref_control_tracking_weight, af) << std::endl;
  out << be << STR(goal_control_regularization_weight, af) << std::endl;
  out << be << STR(running_cost_goal_weight_mask, af) << std::endl;
  out << be << STR(solve_every_k_steps, af) << std::endl;
  out << be << STR(nmpc_mode, af) << std::endl;
  out << be << STR(disturbance_enable, af) << std::endl;
  out << be << STR(disturbance_start_s, af) << std::endl;
  out << be << STR(disturbance_duration_s, af) << std::endl;
  out << be << STR(disturbance_force_x, af) << std::endl;
  out << be << STR(disturbance_force_y, af) << std::endl;
  out << be << STR(disturbance_force_z, af) << std::endl;
  out << be << STR(disturbance_payload_mass, af) << std::endl;
}

void PrintVariableMap(const boost::program_options::variables_map &vm,
                      std::ostream &out) {
  for (po::variables_map::const_iterator it = vm.cbegin(); it != vm.cend();
       it++) {
    out << "> " << it->first;
    if (((boost::any)it->second.value()).empty()) {
      out << "(empty)";
    }
    if (vm[it->first].defaulted() || it->second.defaulted()) {
      out << "(default)";
    }
    out << "=";

    bool is_char;
    try {
      boost::any_cast<const char *>(it->second.value());
      is_char = true;
    } catch (const boost::bad_any_cast &) {
      is_char = false;
    }
    bool is_str;
    try {
      boost::any_cast<std::string>(it->second.value());
      is_str = true;
    } catch (const boost::bad_any_cast &) {
      is_str = false;
    }

    auto &type = ((boost::any)it->second.value()).type();

    if (type == typeid(int)) {
      out << vm[it->first].as<int>() << std::endl;
    } else if (type == typeid(size_t)) {
      out << vm[it->first].as<size_t>() << std::endl;
    } else if (type == typeid(bool)) {
      out << vm[it->first].as<bool>() << std::endl;
    } else if (type == typeid(double)) {
      out << vm[it->first].as<double>() << std::endl;
    } else if (is_char) {
      out << vm[it->first].as<const char *>() << std::endl;
    } else if (is_str) {
      std::string temp = vm[it->first].as<std::string>();
      if (temp.size()) {
        out << temp << std::endl;
      } else {
        out << "true" << std::endl;
      }
    } else { // Assumes that the only remainder is vector<string>
      try {
        std::vector<std::string> vect =
            vm[it->first].as<std::vector<std::string>>();
        uint i = 0;
        for (std::vector<std::string>::iterator oit = vect.begin();
             oit != vect.end(); oit++, ++i) {
          out << "\r> " << it->first << "[" << i << "]=" << (*oit) << std::endl;
        }
      } catch (const boost::bad_any_cast &) {
        out << "UnknownType(" << ((boost::any)it->second.value()).type().name()
            << ")" << std::endl;
        assert(false);
      }
    }
  }
};

} // namespace dynoplan
