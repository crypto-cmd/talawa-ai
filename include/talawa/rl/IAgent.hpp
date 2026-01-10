#pragma once

#include <iostream>
#include <optional>
#include <random>
#include <string>

#include "talawa/core/Matrix.hpp"
#include "talawa/env/types.hpp"

namespace talawa::rl::agent {

class IAgent {
 public:
  virtual ~IAgent() = default;

  // --- Agentic Methods ---
  virtual env::Action act(
      const env::Observation& observation,
      const std::optional<core::Matrix>& mask = std::nullopt,
      bool training = false) = 0;
  virtual void update(const env::Transition& transition) = 0;

  // --- Debugging Methods ---
  virtual void print() const = 0;
};

class ILearnable {
 protected:
  float learning_rate_;

 public:
  virtual ~ILearnable() = default;
  virtual void set_learning_rate(float lr) { learning_rate_ = lr; }
  virtual float get_learning_rate() const { return learning_rate_; }
};

class IExplorable {
 protected:
  float epsilon_;

 public:
  enum class Explorability { EXPLORE, EXPLOIT };
  virtual ~IExplorable() = default;
  virtual void set_epsilon(float eps) { epsilon_ = eps; }
  virtual float get_epsilon() const { return epsilon_; }
  Explorability explorability() const {
    return rand() / static_cast<float>(RAND_MAX) < epsilon_
               ? Explorability::EXPLORE
               : Explorability::EXPLOIT;
  }
};
}  // namespace talawa::rl::agent
