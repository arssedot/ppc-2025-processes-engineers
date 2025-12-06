#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tuple>

#include "dergachev_a_multistep_2d_parallel/common/include/common.hpp"
#include "dergachev_a_multistep_2d_parallel/mpi/include/ops_mpi.hpp"
#include "dergachev_a_multistep_2d_parallel/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dergachev_a_multistep_2d_parallel {

namespace {

double SphereFunc(double x, double y) {
  return (x * x) + (y * y);
}

double SimpleQuadratic(double x, double y) {
  return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));
}

double MatyasFunc(double x, double y) {
  return (0.26 * ((x * x) + (y * y))) - (0.48 * x * y);
}

}  // namespace

class DergachevAMultistep2dParallelFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(params);

    input_data_ = InType();

    switch (test_id) {
      case 1:
        input_data_.func = SphereFunc;
        input_data_.x_min = -2.0;
        input_data_.x_max = 2.0;
        input_data_.y_min = -2.0;
        input_data_.y_max = 2.0;
        input_data_.epsilon = 0.1;
        input_data_.r_param = 2.5;
        input_data_.max_iterations = 15;
        break;

      case 2:
        input_data_.func = SimpleQuadratic;
        input_data_.x_min = 1.0;
        input_data_.x_max = 3.0;
        input_data_.y_min = 2.0;
        input_data_.y_max = 4.0;
        input_data_.epsilon = 0.1;
        input_data_.r_param = 2.5;
        input_data_.max_iterations = 15;
        break;

      case 3:
        input_data_.func = MatyasFunc;
        input_data_.x_min = -3.0;
        input_data_.x_max = 3.0;
        input_data_.y_min = -3.0;
        input_data_.y_max = 3.0;
        input_data_.epsilon = 0.1;
        input_data_.r_param = 2.5;
        input_data_.max_iterations = 15;
        break;

      default:
        input_data_.func = SphereFunc;
        input_data_.x_min = -2.0;
        input_data_.x_max = 2.0;
        input_data_.y_min = -2.0;
        input_data_.y_max = 2.0;
        input_data_.epsilon = 0.1;
        input_data_.r_param = 2.5;
        input_data_.max_iterations = 15;
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.iterations >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(DergachevAMultistep2dParallelFuncTests, GlobalOptimizationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "Sphere"), std::make_tuple(2, "Quadratic"),
                                            std::make_tuple(3, "Matyas")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<DergachevAMultistep2dParallelMPI, InType>(
                                               kTestParam, PPC_SETTINGS_dergachev_a_multistep_2d_parallel),
                                           ppc::util::AddFuncTask<DergachevAMultistep2dParallelSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_dergachev_a_multistep_2d_parallel));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    DergachevAMultistep2dParallelFuncTests::PrintFuncTestName<DergachevAMultistep2dParallelFuncTests>;

INSTANTIATE_TEST_SUITE_P(OptimizationTests, DergachevAMultistep2dParallelFuncTests, kGtestValues, kPerfTestName);

}  // namespace

class DergachevAMultistep2dValidationTests : public ::testing::Test {
 protected:
  static InType CreateValidInput() {
    InType input;
    input.func = SphereFunc;
    input.x_min = -1.0;
    input.x_max = 1.0;
    input.y_min = -1.0;
    input.y_max = 1.0;
    input.epsilon = 0.1;
    input.r_param = 2.5;
    input.max_iterations = 10;
    return input;
  }
};

TEST_F(DergachevAMultistep2dValidationTests, NullFunctionSEQ) {
  if (!ppc::util::IsUnderMpirun()) {
    InType input = CreateValidInput();
    input.func = nullptr;

    auto task = std::make_shared<DergachevAMultistep2dParallelSEQ>(input);
    EXPECT_FALSE(task->Validation());
  }
}

TEST_F(DergachevAMultistep2dValidationTests, InvalidXBoundsSEQ) {
  if (!ppc::util::IsUnderMpirun()) {
    InType input = CreateValidInput();
    input.x_min = 5.0;
    input.x_max = -5.0;

    auto task = std::make_shared<DergachevAMultistep2dParallelSEQ>(input);
    EXPECT_FALSE(task->Validation());
  }
}

TEST_F(DergachevAMultistep2dValidationTests, InvalidYBoundsSEQ) {
  if (!ppc::util::IsUnderMpirun()) {
    InType input = CreateValidInput();
    input.y_min = 10.0;
    input.y_max = 0.0;

    auto task = std::make_shared<DergachevAMultistep2dParallelSEQ>(input);
    EXPECT_FALSE(task->Validation());
  }
}

TEST_F(DergachevAMultistep2dValidationTests, NegativeEpsilonSEQ) {
  if (!ppc::util::IsUnderMpirun()) {
    InType input = CreateValidInput();
    input.epsilon = -0.01;

    auto task = std::make_shared<DergachevAMultistep2dParallelSEQ>(input);
    EXPECT_FALSE(task->Validation());
  }
}

TEST_F(DergachevAMultistep2dValidationTests, InvalidRParamSEQ) {
  if (!ppc::util::IsUnderMpirun()) {
    InType input = CreateValidInput();
    input.r_param = 0.5;

    auto task = std::make_shared<DergachevAMultistep2dParallelSEQ>(input);
    EXPECT_FALSE(task->Validation());
  }
}

TEST_F(DergachevAMultistep2dValidationTests, ZeroIterationsSEQ) {
  if (!ppc::util::IsUnderMpirun()) {
    InType input = CreateValidInput();
    input.max_iterations = 0;

    auto task = std::make_shared<DergachevAMultistep2dParallelSEQ>(input);
    EXPECT_FALSE(task->Validation());
  }
}

TEST_F(DergachevAMultistep2dValidationTests, FullPipelineSEQ) {
  if (!ppc::util::IsUnderMpirun()) {
    InType input = CreateValidInput();

    auto task = std::make_shared<DergachevAMultistep2dParallelSEQ>(input);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());

    auto &result = task->GetOutput();
    EXPECT_GE(result.iterations, 0);
  }
}

TEST_F(DergachevAMultistep2dValidationTests, FullPipelineMPI) {
  InType input = CreateValidInput();

  auto task = std::make_shared<DergachevAMultistep2dParallelMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());

  auto &result = task->GetOutput();
  EXPECT_GE(result.iterations, 0);
}

TEST_F(DergachevAMultistep2dValidationTests, SmallSearchAreaSEQ) {
  if (!ppc::util::IsUnderMpirun()) {
    InType input;
    input.func = SimpleQuadratic;
    input.x_min = 1.5;
    input.x_max = 2.5;
    input.y_min = 2.5;
    input.y_max = 3.5;
    input.epsilon = 0.1;
    input.r_param = 2.0;
    input.max_iterations = 10;

    auto task = std::make_shared<DergachevAMultistep2dParallelSEQ>(input);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());

    auto &result = task->GetOutput();
    EXPECT_GE(result.x_opt, input.x_min);
    EXPECT_LE(result.x_opt, input.x_max);
    EXPECT_GE(result.y_opt, input.y_min);
    EXPECT_LE(result.y_opt, input.y_max);
  }
}

TEST(OptimizationResultTest, EqualityOperator) {
  OptimizationResult r1;
  r1.x_opt = 1.0;
  r1.y_opt = 2.0;
  r1.func_min = 3.0;

  OptimizationResult r2;
  r2.x_opt = 1.0001;
  r2.y_opt = 2.0001;
  r2.func_min = 3.0001;

  EXPECT_TRUE(r1 == r2);

  OptimizationResult r3;
  r3.x_opt = 1.5;
  r3.y_opt = 2.0;
  r3.func_min = 3.0;

  EXPECT_FALSE(r1 == r3);
}

TEST(TrialPointTest, ComparisonOperator) {
  TrialPoint p1(1.0, 2.0, 3.0);
  TrialPoint p2(1.5, 2.0, 3.0);
  TrialPoint p3(1.0, 2.5, 3.0);

  EXPECT_TRUE(p1 < p2);
  EXPECT_TRUE(p1 < p3);
  EXPECT_FALSE(p2 < p1);
}

TEST(PeanoMapTest, BoundaryValues) {
  double x = PeanoToX(0.0, 0.0, 1.0, 0.0, 1.0, 5);
  double y = PeanoToY(0.0, 0.0, 1.0, 0.0, 1.0, 5);
  EXPECT_GE(x, 0.0);
  EXPECT_LE(x, 1.0);
  EXPECT_GE(y, 0.0);
  EXPECT_LE(y, 1.0);

  x = PeanoToX(1.0, 0.0, 1.0, 0.0, 1.0, 5);
  y = PeanoToY(1.0, 0.0, 1.0, 0.0, 1.0, 5);
  EXPECT_GE(x, 0.0);
  EXPECT_LE(x, 1.0);
  EXPECT_GE(y, 0.0);
  EXPECT_LE(y, 1.0);
}

TEST(IntervalTest, DefaultConstruction) {
  Interval i1;
  EXPECT_EQ(i1.left_idx, 0);
  EXPECT_EQ(i1.right_idx, 0);
  EXPECT_DOUBLE_EQ(i1.characteristic, 0.0);

  Interval i2(1, 2, 3.5);
  EXPECT_EQ(i2.left_idx, 1);
  EXPECT_EQ(i2.right_idx, 2);
  EXPECT_DOUBLE_EQ(i2.characteristic, 3.5);
}

}  // namespace dergachev_a_multistep_2d_parallel
