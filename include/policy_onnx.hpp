#pragma once

#include <string>
#include <vector>
#include <array>

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>

namespace dynobench {
class Model_robot;  // forward decl
}

class PolicyOnnx {
public:
  // onnx_path should point to exported policy ONNX file.
  explicit PolicyOnnx(const std::string& onnx_path,
                        int intra_threads = 1);

  // Predict ONE action given (x, u_prev). Both in physical units.
  // Only valid for 2-input models with 1-step output.
  Eigen::VectorXd predict_one(const Eigen::VectorXd& x,
                              const Eigen::VectorXd& u_prev);

  // Predict a flattened chunk [H*nu] from observation x.
  // Works for single-input chunk models. Falls back to iterative rollout-style
  // prediction if the ONNX model is 2-input one-step policy.
  Eigen::VectorXd predict_chunk(const Eigen::VectorXd& x,
                                const Eigen::VectorXd& u_prev,
                                int horizon,
                                int nu);

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

  bool is_chunk_model() const { return input_count_ == 1; }
  bool is_autoregressive_model() const { return input_count_ >= 2; }
  int output_dim() const { return output_dim_; }
  int input_dim() const { return input_dim_; }

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::string x_name_;
  std::string up_name_;
  std::string y_name_;
  std::size_t input_count_ = 0;
  int input_dim_ = -1;
  int output_dim_ = -1;
};
