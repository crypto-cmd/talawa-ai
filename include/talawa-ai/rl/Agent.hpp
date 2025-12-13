#pragma once
#include <iostream>
#include <optional>
#include <string>

#include "talawa-ai/core/Matrix.hpp"
#include "talawa-ai/env/RLTypes.hpp"

namespace talawa_ai::rl::agent {

class Agent {
 public:
  virtual ~Agent() = default;

  virtual env::Action act(
      const env::GameState& state,
      const std::optional<core::Matrix>& mask = std::nullopt,
      bool training = false) = 0;

  virtual std::string name() const = 0;

  virtual void observe(env::Transition transition) = 0;
  virtual void learn() = 0;
  virtual bool ready_to_learn() const = 0;

  virtual void print() const {};  // Empty default implementation

  // Save agent state to file (optional)
  virtual void save(const std::string& filename) const {
    std::cout << "Save not implemented for this agent.\n";
  };
  // Load agent state from file (optional)
  virtual void load(const std::string& filename) {
    std::cout << "Load not implemented for this agent.\n";
  }
};

}  // namespace talawa_ai::rl::agent
