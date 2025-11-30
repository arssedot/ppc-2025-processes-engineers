#pragma once

#include <vector>

#include "dergachev_a_max_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergachev_a_max_elem_vec {

class DergachevAMaxElemVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DergachevAMaxElemVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int vector_size_ = 0;
  std::vector<InType> local_data_;
};

}  // namespace dergachev_a_max_elem_vec
