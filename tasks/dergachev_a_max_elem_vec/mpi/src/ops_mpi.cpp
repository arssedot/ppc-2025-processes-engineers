#include "dergachev_a_max_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "dergachev_a_max_elem_vec/common/include/common.hpp"

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
  int total_processes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

  if (process_rank == 0) {
    vector_size_ = GetInput();
    if (vector_size_ <= 0) {
      return false;
    }
  }

  MPI_Bcast(&vector_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int base_chunk_size = vector_size_ / total_processes;
  const int remainder = vector_size_ % total_processes;

  std::vector<int> send_counts(total_processes);
  std::vector<int> displacements(total_processes);

  for (int i = 0; i < total_processes; ++i) {
    send_counts[i] = base_chunk_size + (i < remainder ? 1 : 0);
    displacements[i] = (i * base_chunk_size) + std::min(i, remainder);
  }

  std::vector<InType> full_data;
  if (process_rank == 0) {
    full_data.resize(vector_size_);
    for (int idx = 0; idx < vector_size_; ++idx) {
      full_data[idx] = ((idx * 7) % 2000) - 1000;
    }
  }

  local_data_.resize(send_counts[process_rank]);

  MPI_Scatterv(full_data.data(), send_counts.data(), displacements.data(), MPI_INT, local_data_.data(),
               send_counts[process_rank], MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool DergachevAMaxElemVecMPI::RunImpl() {
  InType local_maximum = std::numeric_limits<InType>::min();

  for (const auto &value : local_data_) {
    local_maximum = std::max(value, local_maximum);
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
