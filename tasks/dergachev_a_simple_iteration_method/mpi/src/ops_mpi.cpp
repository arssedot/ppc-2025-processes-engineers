#include "dergachev_a_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "dergachev_a_simple_iteration_method/common/include/common.hpp"

namespace dergachev_a_simple_iteration_method {

namespace {

void ComputeRowDistribution(int n, int size, std::vector<int> &recvcounts, std::vector<int> &displs) {
  int row_count = 0;
  for (int proc = 0; proc < size; proc++) {
    int rows_per_proc = n / size;
    int remainder = n % size;
    int proc_local_rows = rows_per_proc + (proc < remainder ? 1 : 0);
    recvcounts[proc] = proc_local_rows;
    displs[proc] = row_count;
    row_count += proc_local_rows;
  }
}

int ComputeFinalResult(const std::vector<double> &x, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return static_cast<int>(std::round(sum));
}

void InitializeIdentityMatrix(std::vector<std::vector<double>> &a, int n) {
  for (int i = 0; i < n; i++) {
    a[i][i] = 1.0;
  }
}

double ComputeVectorNorm(const std::vector<double> &x_new, const std::vector<double> &x, int n) {
  double diff = 0.0;
  for (int i = 0; i < n; i++) {
    double d = x_new[i] - x[i];
    diff += d * d;
  }
  return std::sqrt(diff);
}

}  // namespace

DergachevASimpleIterationMethodMPI::DergachevASimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DergachevASimpleIterationMethodMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool DergachevASimpleIterationMethodMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool DergachevASimpleIterationMethodMPI::RunImpl() {
  int n = GetInput();
  if (n <= 0) {
    return false;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> b(n, 1.0);
  std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0));

  InitializeIdentityMatrix(a, n);

  int rows_per_proc = n / size;
  int remainder = n % size;
  int start_row = (rank * rows_per_proc) + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
  int local_rows = end_row - start_row;

  std::vector<double> local_ax(local_rows, 0.0);
  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);
  ComputeRowDistribution(n, size, recvcounts, displs);

  const double tau = 0.5;
  const double epsilon = 1e-6;
  const int max_iterations = 1000;

  for (int iteration = 0; iteration < max_iterations; iteration++) {
    for (int i = 0; i < local_rows; i++) {
      int global_i = start_row + i;
      double ax_i = 0.0;
      for (int j = 0; j < n; j++) {
        ax_i += a[global_i][j] * x[j];
      }
      local_ax[i] = ax_i;
    }

    std::vector<double> ax_global(n, 0.0);
    MPI_Allgatherv(local_ax.data(), local_rows, MPI_DOUBLE, ax_global.data(), recvcounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < n; i++) {
      x_new[i] = x[i] - (tau * (ax_global[i] - b[i]));
    }

    double diff = ComputeVectorNorm(x_new, x, n);
    x = x_new;

    if (diff < epsilon) {
      break;
    }
  }

  if (rank == 0) {
    GetOutput() = ComputeFinalResult(x, n);
  }

  MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool DergachevASimpleIterationMethodMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace dergachev_a_simple_iteration_method
