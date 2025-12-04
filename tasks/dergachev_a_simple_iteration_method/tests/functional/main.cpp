#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "dergachev_a_simple_iteration_method/common/include/common.hpp"
#include "dergachev_a_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "dergachev_a_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dergachev_a_simple_iteration_method {

class DergachevASimpleIterationMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(DergachevASimpleIterationMethodFuncTests, SimpleIterationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(1, "size_1"),   std::make_tuple(2, "size_2"),
                                             std::make_tuple(3, "size_3"),   std::make_tuple(5, "size_5"),
                                             std::make_tuple(7, "size_7"),   std::make_tuple(10, "size_10"),
                                             std::make_tuple(15, "size_15"), std::make_tuple(20, "size_20"),
                                             std::make_tuple(30, "size_30"), std::make_tuple(50, "size_50")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<DergachevASimpleIterationMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_dergachev_a_simple_iteration_method),
                                           ppc::util::AddFuncTask<DergachevASimpleIterationMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_dergachev_a_simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    DergachevASimpleIterationMethodFuncTests::PrintFuncTestName<DergachevASimpleIterationMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(BasicTests, DergachevASimpleIterationMethodFuncTests, kGtestValues, kFuncTestName);

TEST(DergachevASimpleIterationMethodEdgeCases, InvalidInputZeroSEQ) {
  DergachevASimpleIterationMethodSEQ task(0);
  EXPECT_FALSE(task.Validation());
}

TEST(DergachevASimpleIterationMethodEdgeCases, InvalidInputNegativeSEQ) {
  DergachevASimpleIterationMethodSEQ task(-5);
  EXPECT_FALSE(task.Validation());
}

TEST(DergachevASimpleIterationMethodEdgeCases, ValidInputPositiveSEQ) {
  DergachevASimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
}

TEST(DergachevASimpleIterationMethodEdgeCases, PreProcessingSEQ) {
  DergachevASimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(DergachevASimpleIterationMethodEdgeCases, FullExecutionSEQ) {
  DergachevASimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 5);
}

TEST(DergachevASimpleIterationMethodEdgeCases, MinimalSizeSEQ) {
  DergachevASimpleIterationMethodSEQ task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(DergachevASimpleIterationMethodEdgeCases, InvalidInputZeroMPI) {
  DergachevASimpleIterationMethodMPI task(0);
  EXPECT_FALSE(task.Validation());
}

TEST(DergachevASimpleIterationMethodEdgeCases, InvalidInputNegativeMPI) {
  DergachevASimpleIterationMethodMPI task(-5);
  EXPECT_FALSE(task.Validation());
}

TEST(DergachevASimpleIterationMethodEdgeCases, ValidInputPositiveMPI) {
  DergachevASimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
}

TEST(DergachevASimpleIterationMethodEdgeCases, PreProcessingMPI) {
  DergachevASimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(DergachevASimpleIterationMethodEdgeCases, FullExecutionMPI) {
  DergachevASimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 5);
}

TEST(DergachevASimpleIterationMethodEdgeCases, MinimalSizeMPI) {
  DergachevASimpleIterationMethodMPI task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

}  // namespace

}  // namespace dergachev_a_simple_iteration_method
