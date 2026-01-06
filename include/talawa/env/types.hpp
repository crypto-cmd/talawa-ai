#pragma once
#include <memory>
#include <optional>
#include <string>

#include "talawa/core/Matrix.hpp"

namespace talawa::env {
enum class SpaceType { DISCRETE, CONTINUOUS };

struct Space {
 private:
  std::vector<float> low_;
  std::vector<float> high_;

 public:
  SpaceType type;
  std::vector<int64_t> shape;

  static Space Continuous(std::vector<int64_t> dims, std::vector<float> low,
                          std::vector<float> high) {
    Space s;
    s.type = SpaceType::CONTINUOUS;
    s.shape = std::move(dims);
    s.low_ = std::move(low);
    s.high_ = std::move(high);
    return s;
  }

  static Space Discrete(int n) {
    Space s;
    s.type = SpaceType::DISCRETE;
    s.shape = {1};
    s.low_ = {0.0f};
    s.high_ = {static_cast<float>(n)};
    return s;
  }
  float low(size_t i) const {
    // This branch is highly predictable by CPU
    if (low_.size() == 1) return low_[0];
    return low_[i];
  }

  float high(size_t i) const {
    if (high_.size() == 1) return high_[0];
    return high_[i];
  }
  int n() const {
    if (type != SpaceType::DISCRETE) throw std::runtime_error("Not Discrete!");
    return static_cast<int>(high_[0]);
  }

  const std::vector<float>& get_raw_low() const { return low_; }
  const std::vector<float>& get_raw_high() const { return high_; }
};

using AgentID = size_t;
using Observation = core::Matrix;
struct Action : public core::Matrix {
  Action() : core::Matrix() {}
  Action(const core::Matrix& m) : core::Matrix(m) {}
  static Action None;
};
inline Action Action::None = core::Matrix::Empty;
using ActionMask = core::Matrix;

enum class EpisodeStatus {
  Running,     // Episode is ongoing
  Terminated,  // Episode has ended naturally
  Truncated    // Episode has been cut short (e.g., time limit)
};

struct Transition {
  Observation state;       // s_t
  Action action;           // a_t
  float reward;            // r_{t+1}
  Observation next_state;  // s_{t+1}
  EpisodeStatus status;    // Status of the episode after the action
};

struct StepResult {
  Transition transition;
  Observation observation;
};

}  // namespace talawa::env
