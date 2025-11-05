#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include "dergachev_a_max_elem_vec/common/include/common.hpp"
#include "dergachev_a_max_elem_vec/mpi/include/ops_mpi.hpp"
#include "dergachev_a_max_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dergachev_a_max_elem_vec {

class DergachevAMaxElemVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank != 0) {
        return true;
      }
    }

    if (input_data_ <= 0) {
      return true;
    }

    InType expected_max = std::numeric_limits<InType>::min();
    for (int idx = 0; idx < input_data_; ++idx) {
      const InType value = ((idx * 7) % 2000) - 1000;
      expected_max = std::max(value, expected_max);
    }
    return (expected_max == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(DergachevAMaxElemVecFuncTests, FindsExpectedMaximum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kFunctionalParams = {
    std::make_tuple(1, "size_1_unit"),       std::make_tuple(2, "size_2_pair"),
    std::make_tuple(3, "size_3_small"),      std::make_tuple(5, "size_5_fibonacci"),
    std::make_tuple(7, "size_7_prime"),      std::make_tuple(17, "size_17_prime"),
    std::make_tuple(31, "size_31_prime"),    std::make_tuple(64, "size_64_power2"),
    std::make_tuple(99, "size_99_odd"),      std::make_tuple(128, "size_128_even"),
    std::make_tuple(256, "size_256_power2"), std::make_tuple(50000, "size_50000_stress")};

const auto kTaskMatrix = std::tuple_cat(
    ppc::util::AddFuncTask<DergachevAMaxElemVecMPI, InType>(kFunctionalParams, PPC_SETTINGS_dergachev_a_max_elem_vec),
    ppc::util::AddFuncTask<DergachevAMaxElemVecSEQ, InType>(kFunctionalParams, PPC_SETTINGS_dergachev_a_max_elem_vec));

const auto kParameterizedValues = ppc::util::ExpandToValues(kTaskMatrix);

const auto kFunctionalTestName = DergachevAMaxElemVecFuncTests::PrintFuncTestName<DergachevAMaxElemVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(MaximumSearchSuite, DergachevAMaxElemVecFuncTests, kParameterizedValues, kFunctionalTestName);

TEST(DergachevAMaxElemVecValidation, RejectsNonPositiveInputSeq) {
  const std::array<InType, 3> invalid_values = {0, -1, -50};
  for (InType value : invalid_values) {
    DergachevAMaxElemVecSEQ task(value);
    EXPECT_FALSE(task.Validation());
    EXPECT_FALSE(task.PreProcessing());
  }
}

TEST(DergachevAMaxElemVecValidation, RejectsNonPositiveInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  const std::array<InType, 3> invalid_values = {0, -4, -128};
  for (InType value : invalid_values) {
    DergachevAMaxElemVecMPI task(value);
    EXPECT_FALSE(task.Validation());
    EXPECT_FALSE(task.PreProcessing());
  }
}

TEST(DergachevAMaxElemVecValidation, AcceptsPositiveInputSeq) {
  DergachevAMaxElemVecSEQ task(10);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(DergachevAMaxElemVecValidation, AcceptsPositiveInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  DergachevAMaxElemVecMPI task(10);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(DergachevAMaxElemVecRun, ReturnsFalseForInvalidSizeSeq) {
  DergachevAMaxElemVecSEQ task(0);
  ASSERT_FALSE(task.Validation());
  ASSERT_FALSE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
}

TEST(DergachevAMaxElemVecConsistency, SequentialMatchesFormula) {
  const InType input = 1234;
  DergachevAMaxElemVecSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  InType expected = std::numeric_limits<InType>::min();
  for (int idx = 0; idx < input; ++idx) {
    const InType value = ((idx * 7) % 2000) - 1000;
    expected = std::max(expected, value);
  }
  EXPECT_EQ(expected, task.GetOutput());
}

TEST(DergachevAMaxElemVecConsistency, SeqAndMpiProduceSameResult) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  const InType input = 4096;

  DergachevAMaxElemVecSEQ seq_task(input);
  ASSERT_TRUE(seq_task.Validation());
  ASSERT_TRUE(seq_task.PreProcessing());
  ASSERT_TRUE(seq_task.Run());
  ASSERT_TRUE(seq_task.PostProcessing());

  DergachevAMaxElemVecMPI mpi_task(input);
  ASSERT_TRUE(mpi_task.Validation());
  ASSERT_TRUE(mpi_task.PreProcessing());
  ASSERT_TRUE(mpi_task.Run());
  ASSERT_TRUE(mpi_task.PostProcessing());

  EXPECT_EQ(seq_task.GetOutput(), mpi_task.GetOutput());
}

TEST(DergachevAMaxElemVecPostProcessing, OutputIsWithinRangeSeq) {
  const InType input = 2048;
  DergachevAMaxElemVecSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_GE(task.GetOutput(), -1000);
  EXPECT_LE(task.GetOutput(), 1000);
}

TEST(DergachevAMaxElemVecPostProcessing, OutputIsWithinRangeMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  const InType input = 2048;
  DergachevAMaxElemVecMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_GE(task.GetOutput(), -1000);
  EXPECT_LE(task.GetOutput(), 1000);
}

TEST(DergachevAMaxElemVecLifecycle, RunRequiresPreprocessingSeq) {
  DergachevAMaxElemVecSEQ task(15);
  EXPECT_THROW(task.Run(), std::runtime_error);
}

TEST(DergachevAMaxElemVecLifecycle, PostProcessingRequiresRunSeq) {
  DergachevAMaxElemVecSEQ task(15);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  EXPECT_THROW(task.PostProcessing(), std::runtime_error);
}

TEST(DergachevAMaxElemVecLifecycle, FullPipelineSeq) {
  const InType input = 8192;
  DergachevAMaxElemVecSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 999);
}

TEST(DergachevAMaxElemVecLifecycle, FullPipelineMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  const InType input = 8192;
  DergachevAMaxElemVecMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 999);
}

}  // namespace

}  // namespace dergachev_a_max_elem_vec
