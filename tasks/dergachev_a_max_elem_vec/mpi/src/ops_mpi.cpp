#include "dergachev_a_max_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "dergachev_a_max_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace dergachev_a_max_elem_vec {

DergachevAMaxElemVecMPI::DergachevAMaxElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::numeric_limits<InType>::min();
}

bool DergachevAMaxElemVecMPI::ValidationImpl() {
  int process_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  if (process_rank == 0) {
    return (GetInput() > 0);
  }
  return true;
}

bool DergachevAMaxElemVecMPI::PreProcessingImpl() {
  int process_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  if (process_rank == 0) {
    return GetInput() > 0;
  }

  return true;
}

bool DergachevAMaxElemVecMPI::RunImpl() {
  int process_rank = 0;
  int total_processes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

  int vector_size = 0;

  if (process_rank == 0) {
    vector_size = GetInput();
    if (vector_size <= 0) {
      return false;
    }
  }

  MPI_Bcast(&vector_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int base_chunk_size = vector_size / total_processes;
  const int remainder_elements = vector_size % total_processes;

  const int start_index = process_rank * base_chunk_size + std::min(process_rank, remainder_elements);
  const int end_index = start_index + base_chunk_size + (process_rank < remainder_elements ? 1 : 0);

  InType local_maximum = std::numeric_limits<InType>::min();

  for (int idx = start_index; idx < end_index; ++idx) {
    const InType element_value = (idx * 7) % 2000 - 1000;

    if (element_value > local_maximum) {
      local_maximum = element_value;
    }
  }

  InType global_maximum = std::numeric_limits<InType>::min();
  MPI_Allreduce(&local_maximum, &global_maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_maximum;

  return true;
}

bool DergachevAMaxElemVecMPI::PostProcessingImpl() {
  int process_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  if (process_rank == 0) {
    return GetOutput() >= std::numeric_limits<InType>::min();
  }
  return true;
}

}  // namespace dergachev_a_max_elem_vec
