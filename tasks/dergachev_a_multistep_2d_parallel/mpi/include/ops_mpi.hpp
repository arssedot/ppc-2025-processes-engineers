#pragma once

#include <vector>

#include "dergachev_a_multistep_2d_parallel/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergachev_a_multistep_2d_parallel {

class DergachevAMultistep2dParallelMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DergachevAMultistep2dParallelMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SortTrialsByT();
  double ComputeLipschitzEstimate();
  void ComputeCharacteristicsParallel(double m_val, std::vector<double> &characteristics);
  static int SelectBestInterval(const std::vector<double> &characteristics);
  double PerformTrial(double t);
  void BroadcastTrialData();

  std::vector<TrialPoint> trials_;
  std::vector<double> t_values_;
  double m_estimate_{1.0};
  int peano_level_{10};
  int world_rank_{0};
  int world_size_{1};
};

}  // namespace dergachev_a_multistep_2d_parallel
