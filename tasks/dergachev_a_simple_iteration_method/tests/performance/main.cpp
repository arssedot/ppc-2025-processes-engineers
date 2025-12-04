#include <gtest/gtest.h>

#include "dergachev_a_simple_iteration_method/common/include/common.hpp"
#include "dergachev_a_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "dergachev_a_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dergachev_a_simple_iteration_method {

class DergachevASimpleIterationMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 2500;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return input_data_ == output_data;
  }

  InType GetTestInputData() override {
    return input_data_;
  }
};

TEST_P(DergachevASimpleIterationMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DergachevASimpleIterationMethodMPI, DergachevASimpleIterationMethodSEQ>(
        PPC_SETTINGS_dergachev_a_simple_iteration_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DergachevASimpleIterationMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DergachevASimpleIterationMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace dergachev_a_simple_iteration_method
