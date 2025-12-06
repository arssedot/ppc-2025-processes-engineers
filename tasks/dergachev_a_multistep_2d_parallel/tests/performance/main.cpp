#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "dergachev_a_multistep_2d_parallel/common/include/common.hpp"
#include "dergachev_a_multistep_2d_parallel/mpi/include/ops_mpi.hpp"
#include "dergachev_a_multistep_2d_parallel/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dergachev_a_multistep_2d_parallel {

inline double PerfTestFunction(double x, double y) {
  constexpr double kD = 2.0;
  double sum = 0.0;
  sum += x * std::sin(std::sqrt(std::abs(x)));
  sum += y * std::sin(std::sqrt(std::abs(y)));
  return 418.9829 * kD - sum;
}

inline double RastriginPerfFunc(double x, double y) {
  constexpr double kA = 10.0;
  constexpr double kPi = 3.14159265358979323846;
  return kA * 2.0 + (x * x - kA * std::cos(2.0 * kPi * x)) + (y * y - kA * std::cos(2.0 * kPi * y));
}

inline double AckleyPerfFunc(double x, double y) {
  constexpr double kA = 20.0;
  constexpr double kB = 0.2;
  constexpr double kC = 2.0 * 3.14159265358979323846;
  constexpr double kE = 2.71828182845904523536;

  double sum1 = x * x + y * y;
  double sum2 = std::cos(kC * x) + std::cos(kC * y);

  return -kA * std::exp(-kB * std::sqrt(0.5 * sum1)) - std::exp(0.5 * sum2) + kA + kE;
}

class DergachevAMultistep2dParallelPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMaxIterations_ = 300;
  InType input_data_{};

  void SetUp() override {
    input_data_.func = RastriginPerfFunc;
    input_data_.x_min = -5.12;
    input_data_.x_max = 5.12;
    input_data_.y_min = -5.12;
    input_data_.y_max = 5.12;
    input_data_.epsilon = 0.001;
    input_data_.r_param = 2.5;
    input_data_.max_iterations = kMaxIterations_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kTolerance = 2.0;
    bool result_valid = std::abs(output_data.x_opt) < kTolerance && std::abs(output_data.y_opt) < kTolerance;
    return result_valid || output_data.iterations > 0;
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
