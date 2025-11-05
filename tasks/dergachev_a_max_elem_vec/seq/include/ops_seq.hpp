#pragma once

#include "dergachev_a_max_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergachev_a_max_elem_vec {

class DergachevAMaxElemVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DergachevAMaxElemVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dergachev_a_max_elem_vec
