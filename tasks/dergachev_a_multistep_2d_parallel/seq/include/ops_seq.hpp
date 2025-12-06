#pragma once

#include <vector>

#include "dergachev_a_multistep_2d_parallel/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergachev_a_multistep_2d_parallel {

class DergachevAMultistep2dParallelSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DergachevAMultistep2dParallelSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SortTrialsByT();
  double ComputeLipschitzEstimate();
  double ComputeCharacteristic(int idx, double m_val);
  int SelectBestInterval();
  double PerformTrial(double t);

  std::vector<TrialPoint> trials_;
  std::vector<double> t_values_;
  double m_estimate_{1.0};
  int peano_level_{10};
};

}  // namespace dergachev_a_multistep_2d_parallel
