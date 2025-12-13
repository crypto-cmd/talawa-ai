#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "talawa-ai/core/Matrix.hpp"
#include "talawa-ai/env/GameState.hpp"
#include "talawa-ai/env/RLTypes.hpp"
namespace talawa_ai::env {
class Environment {
 public:
  virtual ~Environment() = default;

  virtual void reset() = 0;
  virtual Observation observe() = 0;
  virtual Transition step(const Action& action) = 0;

  virtual std::unique_ptr<GameState> snapshot() const = 0;
  virtual void restore(const GameState& state) = 0;

  virtual bool is_done() const = 0;

  virtual ActionType get_action_type() const = 0;
  virtual std::string name() const = 0;
  virtual std::vector<int> get_observation_shape() const = 0;
  virtual int get_action_space_size() const = 0;

  virtual Environment* clone() const = 0;

  virtual std::optional<Action> get_legal_mask() {
    return std::nullopt;  // By default, no mask
  }

  virtual void render() const {
    // Default: no rendering
  }
};

};  // namespace talawa_ai::env
