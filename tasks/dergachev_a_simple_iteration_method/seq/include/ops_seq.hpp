#pragma once

#include "dergachev_a_simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergachev_a_simple_iteration_method {

class DergachevASimpleIterationMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DergachevASimpleIterationMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dergachev_a_simple_iteration_method
