#include <gtest/gtest.h>

#include "dergachev_a_max_elem_vec/common/include/common.hpp"
#include "dergachev_a_max_elem_vec/mpi/include/ops_mpi.hpp"
#include "dergachev_a_max_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dergachev_a_max_elem_vec {

class DergachevAMaxElemVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DergachevAMaxElemVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DergachevAMaxElemVecMPI, DergachevAMaxElemVecSEQ>(PPC_SETTINGS_dergachev_a_max_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DergachevAMaxElemVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DergachevAMaxElemVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace dergachev_a_max_elem_vec
