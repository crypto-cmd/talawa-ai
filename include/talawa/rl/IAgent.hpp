#pragma once

#include <iostream>
#include <optional>
#include <string>

#include "talawa/core/Matrix.hpp"
#include "talawa/env/interfaces/IEnvironment.hpp"
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

}  // namespace talawa::rl::agent
