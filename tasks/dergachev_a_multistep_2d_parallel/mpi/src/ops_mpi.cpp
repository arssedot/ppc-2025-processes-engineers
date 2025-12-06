#include "dergachev_a_multistep_2d_parallel/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "dergachev_a_multistep_2d_parallel/common/include/common.hpp"

namespace dergachev_a_multistep_2d_parallel {

DergachevAMultistep2dParallelMPI::DergachevAMultistep2dParallelMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}

bool DergachevAMultistep2dParallelMPI::ValidationImpl() {
  if (world_rank_ != 0) {
    return true;
  }

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

bool DergachevAMultistep2dParallelMPI::PreProcessingImpl() {
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

bool DergachevAMultistep2dParallelMPI::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  for (int iter = 0; iter < input.max_iterations; ++iter) {
    BroadcastTrialData();

    std::vector<std::size_t> indices(t_values_.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
      indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(),
              [this](std::size_t a, std::size_t b) { return t_values_[a] < t_values_[b]; });

    std::vector<double> sorted_t(t_values_.size());
    std::vector<TrialPoint> sorted_trials(trials_.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
      sorted_t[i] = t_values_[indices[i]];
      sorted_trials[i] = trials_[indices[i]];
    }

    t_values_ = sorted_t;
    trials_ = sorted_trials;

    m_estimate_ = ComputeLipschitzEstimate();
    if (m_estimate_ < 1e-10) {
      m_estimate_ = 1.0;
    }

    std::vector<double> all_characteristics;
    ComputeCharacteristicsParallel(input.r_param * m_estimate_, all_characteristics);

    int best_idx = SelectBestInterval(all_characteristics);

    double t_left = t_values_[best_idx];
    double t_right = t_values_[best_idx + 1];
    double z_left = trials_[best_idx].z;
    double z_right = trials_[best_idx + 1].z;

    double m_val = input.r_param * m_estimate_;
    double t_new = (0.5 * (t_left + t_right)) - ((z_right - z_left) / (2.0 * m_val));

    t_new = std::max(t_left + 1e-12, std::min(t_new, t_right - 1e-12));

    double delta = t_right - t_left;
    int converged_flag = (delta < input.epsilon) ? 1 : 0;

    MPI_Bcast(&converged_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (converged_flag != 0) {
      output.converged = true;
      output.iterations = iter + 1;
      break;
    }

    if (world_rank_ == 0) {
      double z_new = PerformTrial(t_new);
      double x_new = PeanoToX(t_new, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
      double y_new = PeanoToY(t_new, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);

      t_values_.push_back(t_new);
      trials_.emplace_back(x_new, y_new, z_new);
    }

    output.iterations = iter + 1;
  }

  BroadcastTrialData();

  return true;
}

bool DergachevAMultistep2dParallelMPI::PostProcessingImpl() {
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

double DergachevAMultistep2dParallelMPI::ComputeLipschitzEstimate() {
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

void DergachevAMultistep2dParallelMPI::ComputeCharacteristicsParallel(double m_val,
                                                                      std::vector<double> &characteristics) {
  int num_intervals = static_cast<int>(t_values_.size()) - 1;
  if (num_intervals <= 0) {
    characteristics.clear();
    return;
  }

  std::vector<int> counts(static_cast<std::size_t>(world_size_));
  std::vector<int> displs(static_cast<std::size_t>(world_size_));

  int base_count = num_intervals / world_size_;
  int remainder = num_intervals % world_size_;

  for (int i = 0; i < world_size_; ++i) {
    counts[static_cast<std::size_t>(i)] = base_count + (i < remainder ? 1 : 0);
    displs[static_cast<std::size_t>(i)] =
        (i == 0) ? 0 : (displs[static_cast<std::size_t>(i - 1)] + counts[static_cast<std::size_t>(i - 1)]);
  }

  std::vector<double> interval_data(static_cast<std::size_t>(num_intervals) * 4);
  if (world_rank_ == 0) {
    for (int i = 0; i < num_intervals; ++i) {
      auto idx = static_cast<std::size_t>(i);
      interval_data[(idx * 4)] = t_values_[idx];
      interval_data[(idx * 4) + 1] = t_values_[idx + 1];
      interval_data[(idx * 4) + 2] = trials_[idx].z;
      interval_data[(idx * 4) + 3] = trials_[idx + 1].z;
    }
  }

  std::vector<int> send_counts(static_cast<std::size_t>(world_size_));
  std::vector<int> send_displs(static_cast<std::size_t>(world_size_));
  for (int i = 0; i < world_size_; ++i) {
    auto idx = static_cast<std::size_t>(i);
    send_counts[idx] = counts[idx] * 4;
    send_displs[idx] = displs[idx] * 4;
  }

  int local_count = counts[static_cast<std::size_t>(world_rank_)];
  std::vector<double> local_interval_data(static_cast<std::size_t>(local_count) * 4);
  MPI_Scatterv(interval_data.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE, local_interval_data.data(),
               local_count * 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_chars(static_cast<std::size_t>(local_count));
  for (int i = 0; i < local_count; ++i) {
    auto idx = static_cast<std::size_t>(i);
    double t_i = local_interval_data[(idx * 4)];
    double t_i1 = local_interval_data[(idx * 4) + 1];
    double z_i = local_interval_data[(idx * 4) + 2];
    double z_i1 = local_interval_data[(idx * 4) + 3];

    double delta = t_i1 - t_i;
    double diff = z_i1 - z_i;
    local_chars[idx] = (m_val * delta) + ((diff * diff) / (m_val * delta)) - (2.0 * (z_i1 + z_i));
  }

  characteristics.resize(static_cast<std::size_t>(num_intervals));

  if (world_rank_ == 0) {
    int count0 = counts[0];
    for (int i = 0; i < count0; ++i) {
      characteristics[static_cast<std::size_t>(i)] = local_chars[static_cast<std::size_t>(i)];
    }

    for (int proc = 1; proc < world_size_; ++proc) {
      int proc_count = counts[static_cast<std::size_t>(proc)];
      if (proc_count > 0) {
        std::vector<double> recv_chars(static_cast<std::size_t>(proc_count));
        MPI_Recv(recv_chars.data(), proc_count, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int disp = displs[static_cast<std::size_t>(proc)];
        for (int i = 0; i < proc_count; ++i) {
          characteristics[static_cast<std::size_t>(disp + i)] = recv_chars[static_cast<std::size_t>(i)];
        }
      }
    }
  } else {
    if (local_count > 0) {
      MPI_Send(local_chars.data(), local_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Bcast(characteristics.data(), num_intervals, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int DergachevAMultistep2dParallelMPI::SelectBestInterval(const std::vector<double> &characteristics) {
  double max_char = -std::numeric_limits<double>::max();
  int best_idx = 0;

  for (std::size_t i = 0; i < characteristics.size(); ++i) {
    if (characteristics[i] > max_char) {
      max_char = characteristics[i];
      best_idx = static_cast<int>(i);
    }
  }

  return best_idx;
}

double DergachevAMultistep2dParallelMPI::PerformTrial(double t) {
  const auto &input = GetInput();
  double x = PeanoToX(t, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  double y = PeanoToY(t, input.x_min, input.x_max, input.y_min, input.y_max, peano_level_);
  return input.func(x, y);
}

void DergachevAMultistep2dParallelMPI::BroadcastTrialData() {
  int size = static_cast<int>(t_values_.size());
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank_ != 0) {
    t_values_.resize(static_cast<std::size_t>(size));
    trials_.resize(static_cast<std::size_t>(size));
  }

  MPI_Bcast(t_values_.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> trial_data(static_cast<std::size_t>(size) * 3);
  if (world_rank_ == 0) {
    for (int i = 0; i < size; ++i) {
      auto idx = static_cast<std::size_t>(i);
      trial_data[(idx * 3)] = trials_[idx].x;
      trial_data[(idx * 3) + 1] = trials_[idx].y;
      trial_data[(idx * 3) + 2] = trials_[idx].z;
    }
  }

  MPI_Bcast(trial_data.data(), size * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank_ != 0) {
    for (int i = 0; i < size; ++i) {
      auto idx = static_cast<std::size_t>(i);
      trials_[idx].x = trial_data[(idx * 3)];
      trials_[idx].y = trial_data[(idx * 3) + 1];
      trials_[idx].z = trial_data[(idx * 3) + 2];
    }
  }
}

}  // namespace dergachev_a_multistep_2d_parallel
