#pragma once

#include <memory>

#include "talawa-ai/core/Matrix.hpp"
#include "talawa-ai/env/GameState.hpp"
namespace talawa_ai::env {

using Observation = core::Matrix;
using Action = core::Matrix;

struct Transition {
  std::unique_ptr<GameState> state = nullptr;
  Action action;
  float reward;
  std::unique_ptr<GameState> next_state = nullptr;
  bool terminated;
};

enum class ActionType { DISCRETE, CONTINUOUS };
};  // namespace talawa_ai::env
