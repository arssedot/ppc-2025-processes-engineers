#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace dergachev_a_multistep_2d_parallel {

struct OptimizationInput {
  std::function<double(double, double)> func;
  double x_min{0.0};
  double x_max{1.0};
  double y_min{0.0};
  double y_max{1.0};
  double epsilon{0.01};
  double r_param{2.0};
  int max_iterations{1000};

  OptimizationInput() : func(nullptr) {}
};

struct OptimizationResult {
  double x_opt{0.0};
  double y_opt{0.0};
  double func_min{0.0};
  int iterations{0};
  bool converged{false};

  OptimizationResult() = default;

  bool operator==(const OptimizationResult &other) const {
    const double tol = 1e-3;
    return std::abs(x_opt - other.x_opt) < tol && std::abs(y_opt - other.y_opt) < tol &&
           std::abs(func_min - other.func_min) < tol;
  }
};

struct TrialPoint {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  TrialPoint() = default;
  TrialPoint(double px, double py, double pz) : x(px), y(py), z(pz) {}

  bool operator<(const TrialPoint &other) const {
    if (x != other.x) {
      return x < other.x;
    }
    return y < other.y;
  }
};

struct Interval {
  int left_idx{0};
  int right_idx{0};
  double characteristic{0.0};

  Interval() = default;
  Interval(int l, int r, double c) : left_idx(l), right_idx(r), characteristic(c) {}
};

using InType = OptimizationInput;
using OutType = OptimizationResult;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline double PeanoToX(double t, double x_min, double x_max, double y_min, double y_max, int level);
inline double PeanoToY(double t, double x_min, double x_max, double y_min, double y_max, int level);

namespace detail {

inline void PeanoMap(double t, int level, double &x, double &y) {
  x = 0.0;
  y = 0.0;
  double scale = 0.5;

  for (int i = 0; i < level; ++i) {
    int quadrant = static_cast<int>(t * 4.0);
    t = (t * 4.0) - quadrant;

    double tx = 0.0;
    double ty = 0.0;
    switch (quadrant) {
      case 0:
        tx = 0.0;
        ty = 0.0;
        break;
      case 1:
        tx = 0.0;
        ty = 1.0;
        break;
      case 2:
        tx = 1.0;
        ty = 1.0;
        break;
      case 3:
        tx = 1.0;
        ty = 0.0;
        break;
      default:
        tx = 0.0;
        ty = 0.0;
        break;
    }
    x += tx * scale;
    y += ty * scale;
    scale *= 0.5;
  }
}

}  // namespace detail

inline double PeanoToX(double t, double x_min, double x_max, double /*y_min*/, double /*y_max*/, int level) {
  double x = 0.0;
  double y = 0.0;
  detail::PeanoMap(t, level, x, y);
  return x_min + (x * (x_max - x_min));
}

inline double PeanoToY(double t, double /*x_min*/, double /*x_max*/, double y_min, double y_max, int level) {
  double x = 0.0;
  double y = 0.0;
  detail::PeanoMap(t, level, x, y);
  return y_min + (y * (y_max - y_min));
}

}  // namespace dergachev_a_multistep_2d_parallel
