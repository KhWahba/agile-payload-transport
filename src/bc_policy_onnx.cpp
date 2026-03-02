#include "bc_policy_onnx.hpp"
#include "robot_models_base.hpp"  // for dynobench::Model_robot

#include <stdexcept>
#include <algorithm>

BCPolicyOnnx::BCPolicyOnnx(const std::string& onnx_path, int intra_threads)
: env_(ORT_LOGGING_LEVEL_WARNING, "bc_policy"),
  session_options_(),
  session_(nullptr),
  allocator_()
{
  session_options_.SetIntraOpNumThreads(intra_threads);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_ = Ort::Session(env_, onnx_path.c_str(), session_options_);

  // assumes ONNX inputs are [x, u_prev] and output is [u]
  x_name_  = session_.GetInputNameAllocated(0, allocator_).get();
  up_name_ = session_.GetInputNameAllocated(1, allocator_).get();
  y_name_  = session_.GetOutputNameAllocated(0, allocator_).get();
}

Eigen::VectorXd BCPolicyOnnx::predict_one(const Eigen::VectorXd& x,
                                         const Eigen::VectorXd& u_prev)
{
  const int nx = static_cast<int>(x.size());
  const int nu = static_cast<int>(u_prev.size());

  // batch=1
  std::array<int64_t,2> x_shape  {1, nx};
  std::array<int64_t,2> up_shape {1, nu};

  std::vector<float> x_f(nx), up_f(nu);
  for (int i=0;i<nx;i++) x_f[i] = static_cast<float>(x[i]);
  for (int i=0;i<nu;i++) up_f[i] = static_cast<float>(u_prev[i]);

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  Ort::Value x_tensor  = Ort::Value::CreateTensor<float>(
      mem, x_f.data(), x_f.size(), x_shape.data(), x_shape.size());

  Ort::Value up_tensor = Ort::Value::CreateTensor<float>(
      mem, up_f.data(), up_f.size(), up_shape.data(), up_shape.size());

  const char* in_names[]  = {x_name_.c_str(), up_name_.c_str()};
  const char* out_names[] = {y_name_.c_str()};
  Ort::Value inputs[] = {std::move(x_tensor), std::move(up_tensor)};

  auto outputs = session_.Run(Ort::RunOptions{nullptr},
                              in_names, inputs, 2,
                              out_names, 1);

  float* y = outputs[0].GetTensorMutableData<float>(); // shape (1,nu)

  Eigen::VectorXd u(nu);
  for (int i=0;i<nu;i++) u[i] = static_cast<double>(y[i]);
  return u;
}

void BCPolicyOnnx::predict_rollout(
    int N,
    dynobench::Model_robot* model,
    const Eigen::VectorXd& x0,
    Eigen::VectorXd& u_prev_io,
    std::vector<Eigen::VectorXd>* xs_out,
    std::vector<Eigen::VectorXd>* us_out,
    double u_clip_min,
    double u_clip_max)
{
  if (!model) throw std::runtime_error("BCPolicyOnnx::predict_rollout: model is null");
  if (!xs_out || !us_out) throw std::runtime_error("BCPolicyOnnx::predict_rollout: output ptr null");
  if (N <= 0) { xs_out->clear(); us_out->clear(); return; }

  const int nx = model->nx;
  const int nu = model->nu;

  if (x0.size() != nx) throw std::runtime_error("BCPolicyOnnx: x0 has wrong size");
  if (u_prev_io.size() != nu) u_prev_io = Eigen::VectorXd::Zero(nu);

  xs_out->clear();
  us_out->clear();
  xs_out->reserve(N + 1);
  us_out->reserve(N);

  Eigen::VectorXd x = x0;
  xs_out->push_back(x);

  for (int k = 0; k < N; ++k) {
    Eigen::VectorXd u = predict_one(x, u_prev_io);

    // optional clip
    for (int i = 0; i < nu; ++i) {
      u[i] = std::min(std::max(u[i], u_clip_min), u_clip_max);
    }

    us_out->push_back(u);

    Eigen::VectorXd xnext(nx);
    model->step(xnext, x, u, model->ref_dt);   // rollout with dynobench step()
    xs_out->push_back(xnext);

    x = xnext;
    u_prev_io = u;
  }
}
