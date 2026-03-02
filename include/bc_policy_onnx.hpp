#pragma once

#include <string>
#include <vector>
#include <array>

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>

namespace dynobench {
class Model_robot;  // forward decl
}

class BCPolicyOnnx {
public:
  // onnx_path should point to bc_policy.onnx (bc_policy.onnx.data must sit next to it)
  explicit BCPolicyOnnx(const std::string& onnx_path,
                        int intra_threads = 1);

  // Predict ONE action given (x, u_prev). Both in physical units.
  Eigen::VectorXd predict_one(const Eigen::VectorXd& x,
                              const Eigen::VectorXd& u_prev);

  // Rollout N steps:
  // - inputs: N, model (for step), x0, u_prev (in/out), optional clipping
  // - outputs: xs_out size N+1, us_out size N
  void predict_rollout(
      int N,
      dynobench::Model_robot* model,
      const Eigen::VectorXd& x0,
      Eigen::VectorXd& u_prev_io,
      std::vector<Eigen::VectorXd>* xs_out,
      std::vector<Eigen::VectorXd>* us_out,
      double u_clip_min = -1e30,
      double u_clip_max =  1e30);

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::string x_name_;
  std::string up_name_;
  std::string y_name_;
};
