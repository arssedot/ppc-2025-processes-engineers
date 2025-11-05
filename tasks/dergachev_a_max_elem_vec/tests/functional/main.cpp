#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <limits>
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
    InType expected_max = std::numeric_limits<InType>::min();
    for (int idx = 0; idx < input_data_; ++idx) {
      InType value = (idx * 7) % 2000 - 1000;
      if (value > expected_max) {
        expected_max = value;
      }
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

TEST_P(DergachevAMaxElemVecFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<DergachevAMaxElemVecMPI, InType>(kTestParam, PPC_SETTINGS_dergachev_a_max_elem_vec),
    ppc::util::AddFuncTask<DergachevAMaxElemVecSEQ, InType>(kTestParam, PPC_SETTINGS_dergachev_a_max_elem_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = DergachevAMaxElemVecFuncTests::PrintFuncTestName<DergachevAMaxElemVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, DergachevAMaxElemVecFuncTests, kGtestValues, kPerfTestName);

TEST(DergachevAMaxElemVecValidationTest, TestValidationWithValidInput_SEQ) {
  InType input = 10;
  DergachevAMaxElemVecSEQ task(input);
  ASSERT_TRUE(task.Validation());
}

TEST(DergachevAMaxElemVecValidationTest, TestValidationWithValidInput_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 10;
  DergachevAMaxElemVecMPI task(input);
  ASSERT_TRUE(task.Validation());
}

TEST(DergachevAMaxElemVecValidationTest, TestValidationWithZeroInput_SEQ) {
  InType input = 0;
  DergachevAMaxElemVecSEQ task(input);
  ASSERT_FALSE(task.Validation());
}

TEST(DergachevAMaxElemVecValidationTest, TestValidationWithNegativeInput_SEQ) {
  InType input = -5;
  DergachevAMaxElemVecSEQ task(input);
  ASSERT_FALSE(task.Validation());
}

TEST(DergachevAMaxElemVecBasicTest, TestSmallVector_SEQ) {
  InType input = 5;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();

  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecBasicTest, TestSmallVector_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 5;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();

    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecBasicTest, TestLargeVector_SEQ) {
  InType input = 1000;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecBasicTest, TestLargeVector_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 1000;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecConsistencyTest, TestSEQandMPIGiveSameResult) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 100;

  DergachevAMaxElemVecSEQ seq_task(input);
  ASSERT_TRUE(seq_task.Validation());
  ASSERT_TRUE(seq_task.PreProcessing());
  ASSERT_TRUE(seq_task.Run());
  ASSERT_TRUE(seq_task.PostProcessing());
  OutType seq_result = seq_task.GetOutput();

  DergachevAMaxElemVecMPI mpi_task(input);
  ASSERT_TRUE(mpi_task.Validation());
  ASSERT_TRUE(mpi_task.PreProcessing());
  ASSERT_TRUE(mpi_task.Run());
  ASSERT_TRUE(mpi_task.PostProcessing());

  if (rank == 0) {
    OutType mpi_result = mpi_task.GetOutput();

    ASSERT_EQ(seq_result, mpi_result);
  }
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestSingleElement_SEQ) {
  InType input = 1;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestSingleElement_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 1;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestTwoElements_SEQ) {
  InType input = 2;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestTwoElements_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 2;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestThreeElements_SEQ) {
  InType input = 3;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestThreeElements_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 3;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestPrimeNumberSize_SEQ) {
  InType input = 17;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestPrimeNumberSize_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 17;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestOddNumberSize_SEQ) {
  InType input = 99;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestOddNumberSize_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 99;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestEvenNumberSize_SEQ) {
  InType input = 128;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestEvenNumberSize_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 128;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestPowerOfTwo_SEQ) {
  InType input = 256;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecEdgeCasesTest, TestPowerOfTwo_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 256;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

TEST(DergachevAMaxElemVecConsistencyTest, TestConsistencySmallVector) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 7;

  DergachevAMaxElemVecSEQ seq_task(input);
  ASSERT_TRUE(seq_task.Validation());
  ASSERT_TRUE(seq_task.PreProcessing());
  ASSERT_TRUE(seq_task.Run());
  ASSERT_TRUE(seq_task.PostProcessing());
  OutType seq_result = seq_task.GetOutput();

  DergachevAMaxElemVecMPI mpi_task(input);
  ASSERT_TRUE(mpi_task.Validation());
  ASSERT_TRUE(mpi_task.PreProcessing());
  ASSERT_TRUE(mpi_task.Run());
  ASSERT_TRUE(mpi_task.PostProcessing());

  if (rank == 0) {
    OutType mpi_result = mpi_task.GetOutput();
    ASSERT_EQ(seq_result, mpi_result);
  }
}

TEST(DergachevAMaxElemVecConsistencyTest, TestConsistencyMediumVector) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 500;

  DergachevAMaxElemVecSEQ seq_task(input);
  ASSERT_TRUE(seq_task.Validation());
  ASSERT_TRUE(seq_task.PreProcessing());
  ASSERT_TRUE(seq_task.Run());
  ASSERT_TRUE(seq_task.PostProcessing());
  OutType seq_result = seq_task.GetOutput();

  DergachevAMaxElemVecMPI mpi_task(input);
  ASSERT_TRUE(mpi_task.Validation());
  ASSERT_TRUE(mpi_task.PreProcessing());
  ASSERT_TRUE(mpi_task.Run());
  ASSERT_TRUE(mpi_task.PostProcessing());

  if (rank == 0) {
    OutType mpi_result = mpi_task.GetOutput();
    ASSERT_EQ(seq_result, mpi_result);
  }
}

TEST(DergachevAMaxElemVecConsistencyTest, TestConsistencyLargeVector) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 5000;

  DergachevAMaxElemVecSEQ seq_task(input);
  ASSERT_TRUE(seq_task.Validation());
  ASSERT_TRUE(seq_task.PreProcessing());
  ASSERT_TRUE(seq_task.Run());
  ASSERT_TRUE(seq_task.PostProcessing());
  OutType seq_result = seq_task.GetOutput();

  DergachevAMaxElemVecMPI mpi_task(input);
  ASSERT_TRUE(mpi_task.Validation());
  ASSERT_TRUE(mpi_task.PreProcessing());
  ASSERT_TRUE(mpi_task.Run());
  ASSERT_TRUE(mpi_task.PostProcessing());

  if (rank == 0) {
    OutType mpi_result = mpi_task.GetOutput();
    ASSERT_EQ(seq_result, mpi_result);
  }
}

TEST(DergachevAMaxElemVecStressTest, TestVeryLargeVector_SEQ) {
  InType input = 50000;
  DergachevAMaxElemVecSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  OutType result = task.GetOutput();
  ASSERT_GE(result, -1000);
  ASSERT_LE(result, 1000);
}

TEST(DergachevAMaxElemVecStressTest, TestVeryLargeVector_MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = 50000;
  DergachevAMaxElemVecMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    OutType result = task.GetOutput();
    ASSERT_GE(result, -1000);
    ASSERT_LE(result, 1000);
  }
}

}  // namespace

}  // namespace dergachev_a_max_elem_vec
