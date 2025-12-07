#include "dergachev_a_multistep_2d_parallel/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "dergachev_a_multistep_2d_parallel/common/include/common.hpp"

namespace dergachev_a_multistep_2d_parallel {

DergachevAMultistep2dParallelSEQ::DergachevAMultistep2dParallelSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool DergachevAMultistep2dParallelSEQ::ValidationImpl() {
  const auto &input = GetInput();
  bool valid = true;
  valid = valid && (input.func != nullptr);
  valid = valid && (input.x_min < input.x_max);
  valid = valid && (input.y_min < input.y_max);
  valid = valid && (input.epsilon > 0);
  valid = valid && (input.r_param > 1.0);
  valid = valid && (input.max_iterations > 0);
  return valid;
}

bool DergachevAMultistep2dParallelSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  trials_.clear();
  t_values_.clear();

  double t0 = 0.0;
  double t1 = 1.0;

  t_values_.push_back(t0);
  t_values_.push_back(t1);

  double x0 = PeanoToX(t0, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double y0 = PeanoToY(t0, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double z0 = input.func(x0, y0);

  double x1 = PeanoToX(t1, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double y1 = PeanoToY(t1, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double z1 = input.func(x1, y1);

  trials_.emplace_back(x0, y0, z0);
  trials_.emplace_back(x1, y1, z1);

  m_estimate_ = 1.0;

  return true;
}

bool DergachevAMultistep2dParallelSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  trials_.clear();
  t_values_.clear();

  double t0 = 0.0;
  double t1 = 1.0;

  t_values_.push_back(t0);
  t_values_.push_back(t1);

  double x0 = PeanoToX(t0, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double y0 = PeanoToY(t0, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double z0 = input.func(x0, y0);

  double x1 = PeanoToX(t1, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double y1 = PeanoToY(t1, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double z1 = input.func(x1, y1);

  trials_.emplace_back(x0, y0, z0);
  trials_.emplace_back(x1, y1, z1);

  m_estimate_ = 1.0;

  for (int iter = 0; iter < input.max_iterations; ++iter) {
    SortTrialsByT();

    m_estimate_ = ComputeLipschitzEstimate();
    if (m_estimate_ < 1e-10) {
      m_estimate_ = 1.0;
    }

    int best_idx = SelectBestInterval();

    double t_left = t_values_[best_idx];
    double t_right = t_values_[best_idx + 1];
    double z_left = trials_[best_idx].z;
    double z_right = trials_[best_idx + 1].z;

    double m_val = input.r_param * m_estimate_;
    double t_new = (0.5 * (t_left + t_right)) - ((z_right - z_left) / (2.0 * m_val));

    t_new = std::max(t_left + 1e-12, std::min(t_new, t_right - 1e-12));

    double delta = t_right - t_left;
    if (delta < input.epsilon) {
      output.converged = true;
      output.iterations = iter + 1;
      break;
    }

    double z_new = PerformTrial(t_new);
    double x_new = PeanoToX(t_new, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
    double y_new = PeanoToY(t_new, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);

    t_values_.push_back(t_new);
    trials_.emplace_back(x_new, y_new, z_new);

    output.iterations = iter + 1;
  }

  return true;
}

bool DergachevAMultistep2dParallelSEQ::PostProcessingImpl() {
  auto &output = GetOutput();

  double min_z = std::numeric_limits<double>::max();
  int min_idx = 0;

  for (std::size_t i = 0; i < trials_.size(); ++i) {
    if (trials_[i].z < min_z) {
      min_z = trials_[i].z;
      min_idx = static_cast<int>(i);
    }
  }

  output.x_opt = trials_[min_idx].x;
  output.y_opt = trials_[min_idx].y;
  output.func_min = min_z;

  return true;
}

void DergachevAMultistep2dParallelSEQ::SortTrialsByT() {
  std::vector<std::size_t> indices(t_values_.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }
  std::ranges::sort(indices, [this](std::size_t a, std::size_t b) { return t_values_[a] < t_values_[b]; });

  std::vector<double> sorted_t(t_values_.size());
  std::vector<TrialPoint> sorted_trials(trials_.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    sorted_t[i] = t_values_[indices[i]];
    sorted_trials[i] = trials_[indices[i]];
  }
  t_values_ = std::move(sorted_t);
  trials_ = std::move(sorted_trials);
}

double DergachevAMultistep2dParallelSEQ::ComputeLipschitzEstimate() {
  double max_slope = 0.0;

  for (std::size_t i = 1; i < t_values_.size(); ++i) {
    double dt = t_values_[i] - t_values_[i - 1];
    if (dt > 1e-15) {
      double dz = std::abs(trials_[i].z - trials_[i - 1].z);
      double slope = dz / dt;
      max_slope = std::max(slope, max_slope);
    }
  }

  return max_slope > 0.0 ? max_slope : 1.0;
}

double DergachevAMultistep2dParallelSEQ::ComputeCharacteristic(int idx, double m_val) {
  double t_i = t_values_[idx];
  double t_i1 = t_values_[idx + 1];
  double z_i = trials_[idx].z;
  double z_i1 = trials_[idx + 1].z;

  double delta = t_i1 - t_i;
  double diff = z_i1 - z_i;

  double r_val = (m_val * delta) + ((diff * diff) / (m_val * delta)) - (2.0 * (z_i1 + z_i));

  return r_val;
}

int DergachevAMultistep2dParallelSEQ::SelectBestInterval() {
  const auto &input = GetInput();
  double m_val = input.r_param * m_estimate_;

  double max_char = -std::numeric_limits<double>::max();
  int best_idx = 0;

  for (std::size_t i = 0; i + 1 < t_values_.size(); ++i) {
    double characteristic = ComputeCharacteristic(static_cast<int>(i), m_val);
    if (characteristic > max_char) {
      max_char = characteristic;
      best_idx = static_cast<int>(i);
    }
  }

  return best_idx;
}

double DergachevAMultistep2dParallelSEQ::PerformTrial(double t) {
  const auto &input = GetInput();
  double x = PeanoToX(t, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double y = PeanoToY(t, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  return input.func(x, y);
}

}  // namespace dergachev_a_multistep_2d_parallel
