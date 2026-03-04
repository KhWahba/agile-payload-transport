#include "policy_onnx.hpp"
#include "robot_models_base.hpp"  // for dynobench::Model_robot

#include <stdexcept>
#include <algorithm>
#include <limits>

PolicyOnnx::PolicyOnnx(const std::string& onnx_path, int intra_threads)
: env_(ORT_LOGGING_LEVEL_WARNING, "policy_onnx"),
  session_options_(),
  session_(nullptr),
  allocator_()
{
  session_options_.SetIntraOpNumThreads(intra_threads);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_ = Ort::Session(env_, onnx_path.c_str(), session_options_);
  input_count_ = session_.GetInputCount();
  if (input_count_ < 1) {
    throw std::runtime_error("PolicyOnnx: model must have at least one input");
  }
  x_name_ = session_.GetInputNameAllocated(0, allocator_).get();
  if (input_count_ >= 2) {
    up_name_ = session_.GetInputNameAllocated(1, allocator_).get();
  }
  y_name_ = session_.GetOutputNameAllocated(0, allocator_).get();

  try {
    auto in_info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto in_shape = in_info.GetShape();
    if (in_shape.size() >= 2 && in_shape.back() > 0 &&
        in_shape.back() <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
      input_dim_ = static_cast<int>(in_shape.back());
    }
  } catch (const std::exception&) {
    input_dim_ = -1;
  }

  try {
    auto out_info = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    if (out_shape.size() >= 2 && out_shape.back() > 0 &&
        out_shape.back() <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
      output_dim_ = static_cast<int>(out_shape.back());
    }
  } catch (const std::exception&) {
    // Some exported models have malformed/unsupported static shape metadata.
    // Fall back to runtime-driven horizon*nu sizing.
    output_dim_ = -1;
  }
}

Eigen::VectorXd PolicyOnnx::predict_one(const Eigen::VectorXd& x,
                                         const Eigen::VectorXd& u_prev)
{
  if (input_count_ < 2) {
    throw std::runtime_error("PolicyOnnx::predict_one requires 2-input autoregressive model");
  }
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

Eigen::VectorXd PolicyOnnx::predict_chunk(const Eigen::VectorXd& x,
                                            const Eigen::VectorXd& u_prev,
                                            int horizon,
                                            int nu) {
  if (input_count_ == 1) {
    const int nx = static_cast<int>(x.size());
    std::array<int64_t, 2> x_shape{1, nx};
    std::vector<float> x_f(nx);
    for (int i = 0; i < nx; ++i) x_f[i] = static_cast<float>(x[i]);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
        mem, x_f.data(), x_f.size(), x_shape.data(), x_shape.size());
    const char* in_names[] = {x_name_.c_str()};
    const char* out_names[] = {y_name_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &x_tensor, 1, out_names, 1);
    float* y = outputs[0].GetTensorMutableData<float>();
    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto elem_count = static_cast<int>(out_info.GetElementCount());
    if (elem_count <= 0) {
      throw std::runtime_error("PolicyOnnx::predict_chunk got empty output tensor");
    }
    const int dim = elem_count;
    Eigen::VectorXd out(dim);
    for (int i = 0; i < dim; ++i) out[i] = static_cast<double>(y[i]);
    return out;
  }

  // Fallback: old autoregressive one-step model. Build a pseudo chunk by
  // repeatedly querying with the same observation and updating u_prev only.
  Eigen::VectorXd up = u_prev;
  Eigen::VectorXd out(horizon * nu);
  for (int k = 0; k < horizon; ++k) {
    Eigen::VectorXd uk = predict_one(x, up);
    if (uk.size() != nu) {
      throw std::runtime_error("PolicyOnnx::predict_chunk fallback got wrong action size");
    }
    out.segment(k * nu, nu) = uk;
    up = uk;
  }
  return out;
}

void PolicyOnnx::predict_rollout(
    int N,
    dynobench::Model_robot* model,
    const Eigen::VectorXd& x0,
    Eigen::VectorXd& u_prev_io,
    std::vector<Eigen::VectorXd>* xs_out,
    std::vector<Eigen::VectorXd>* us_out,
    double u_clip_min,
    double u_clip_max)
{
  if (!model) throw std::runtime_error("PolicyOnnx::predict_rollout: model is null");
  if (!xs_out || !us_out) throw std::runtime_error("PolicyOnnx::predict_rollout: output ptr null");
  if (N <= 0) { xs_out->clear(); us_out->clear(); return; }

  const int nx = model->nx;
  const int nu = model->nu;

  if (x0.size() != nx) throw std::runtime_error("PolicyOnnx: x0 has wrong size");
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
