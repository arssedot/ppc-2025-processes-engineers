#include "dergachev_a_max_elem_vec/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "dergachev_a_max_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace dergachev_a_max_elem_vec {

DergachevAMaxElemVecSEQ::DergachevAMaxElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::numeric_limits<InType>::min();
}

bool DergachevAMaxElemVecSEQ::ValidationImpl() {
  return (GetInput() > 0);
}

bool DergachevAMaxElemVecSEQ::PreProcessingImpl() {
  return GetInput() > 0;
}

bool DergachevAMaxElemVecSEQ::RunImpl() {
  if (GetInput() <= 0) {
    return false;
  }

  const int vector_size = GetInput();
  InType current_max = std::numeric_limits<InType>::min();

  for (int idx = 0; idx < vector_size; ++idx) {
    const InType current_value = (idx * 7) % 2000 - 1000;

    if (current_value > current_max) {
      current_max = current_value;
    }
  }

  GetOutput() = current_max;
  return true;
}

bool DergachevAMaxElemVecSEQ::PostProcessingImpl() {
  return GetOutput() >= std::numeric_limits<InType>::min();
}

}  // namespace dergachev_a_max_elem_vec
