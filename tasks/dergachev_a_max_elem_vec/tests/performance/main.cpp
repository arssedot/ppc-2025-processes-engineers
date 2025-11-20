#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <limits>

#include "dergachev_a_max_elem_vec/common/include/common.hpp"
#include "dergachev_a_max_elem_vec/mpi/include/ops_mpi.hpp"
#include "dergachev_a_max_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dergachev_a_max_elem_vec {

class DergachevAMaxElemVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    // Используем меньший размер в CI окружении для более быстрого выполнения
    const char* ci_env = std::getenv("CI");
    if (ci_env != nullptr) {
      input_data_ = 100000000;  // 100 млн для CI
    } else {
      input_data_ = 800000000;  // 800 млн для локального запуска
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    InType expected_max = std::numeric_limits<InType>::min();
    const int limit = std::min<int>(input_data_, 2000);

    for (int idx = 0; idx < limit; ++idx) {
      const auto value = static_cast<InType>(((idx * 7) % 2000) - 1000);
      expected_max = std::max(expected_max, value);
      if (expected_max == 999) {
        break;
      }
    }

    if (input_data_ > limit) {
      expected_max = std::max(expected_max, static_cast<InType>(999));
    }

    return output_data == expected_max;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DergachevAMaxElemVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DergachevAMaxElemVecMPI, DergachevAMaxElemVecSEQ>(
    PPC_SETTINGS_dergachev_a_max_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DergachevAMaxElemVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DergachevAMaxElemVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace dergachev_a_max_elem_vec
