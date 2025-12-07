#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "dergachev_a_multistep_2d_parallel/common/include/common.hpp"
#include "dergachev_a_multistep_2d_parallel/mpi/include/ops_mpi.hpp"
#include "dergachev_a_multistep_2d_parallel/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dergachev_a_multistep_2d_parallel {

namespace {

double SpherePerfFunc(double x, double y) {
  return (x * x) + (y * y);
}

}  // namespace

class DergachevAMultistep2dParallelPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMaxIterations_ = 2000;
  InType input_data_;

  void SetUp() override {
    input_data_.func = SpherePerfFunc;
    input_data_.x_min = -5.0;
    input_data_.x_max = 5.0;
    input_data_.y_min = -5.0;
    input_data_.y_max = 5.0;
    input_data_.epsilon = 0.0001;
    input_data_.r_param = 2.5;
    input_data_.max_iterations = kMaxIterations_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.iterations >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DergachevAMultistep2dParallelPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DergachevAMultistep2dParallelMPI, DergachevAMultistep2dParallelSEQ>(
        PPC_SETTINGS_dergachev_a_multistep_2d_parallel);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DergachevAMultistep2dParallelPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DergachevAMultistep2dParallelPerfTests, kGtestValues, kPerfTestName);

}  // namespace dergachev_a_multistep_2d_parallel
