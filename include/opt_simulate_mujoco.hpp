#pragma once
#include "motions.hpp"
#include <string>

bool execute_optMujoco(std::string &env_file,
                       std::string &initial_guess_file,
                       std::string &output_file,
                       std::string &output_file_anytime,
                       dynobench::Trajectory &sol,
                       const std::string &dynobench_base,
                       bool sum_robots_cost,  dynobench::Trajectory &sol_broken, std::string cfg_file="");

void execute_simMujoco(std::string &env_file,
                       std::string &initial_guess_file,
                       dynobench::Trajectory &sol,
                       const std::string &dynobench_base,
                       const std::string &video_path,
                       const std::string &camera_view = "auto",
                       int num_repeats = 2, bool view_ghost = false, bool feasible = false);

void execute_nmpc_mujoco(dynobench::Problem &problem,
                       dynobench::Trajectory &init_guess,
                       dynobench::Trajectory &sol,
                        dynobench::Trajectory &sol_broken, std::string cfg_file);